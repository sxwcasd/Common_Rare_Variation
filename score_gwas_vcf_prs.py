"""Harmonize one GWAS Catalog analysis to a GRCh38 VCF and score its samples.

Uses GT-derived allele frequencies to check the reported effect allele and
resolve informative palindromic SNPs. Optional Hail GRCh38 sequence validation
checks VCF REF before allele inference; ALT is always supplied by the VCF.
This constructs an exploratory score;
it does not perform LD clumping or reproduce a validated published PRS.
See GWAS_VCF_PRS.md. Requires numpy and pandas; reference validation also needs Hail.
"""
import argparse
from contextlib import nullcontext
import gzip
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


COMPLEMENT = str.maketrans('ACGT', 'TGCA')
GRCH38_FASTA = 'gs://hail-common/references/Homo_sapiens_assembly38.fasta.gz'
GRCH38_FAI = 'gs://hail-common/references/Homo_sapiens_assembly38.fasta.fai'


class IncompleteGWASScoreError(ValueError):
    """Unresolved associations: inspect .report and .reference before scoring."""

    def __init__(self, message, report, reference):
        super().__init__(message)
        self.report = report
        self.reference = reference


def _text(value):
    return '' if pd.isna(value) else str(value).strip()


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _chrom(value):
    value = re.sub(r'\.0+$', '', _text(value).removeprefix('chr')).upper()
    return 'MT' if value == 'M' else value


def _weight(value, ci, effect_type):
    """Return signed weight/type; ambiguous automatic interpretations fail."""
    raw = _number(value)
    ci = _text(ci).lower()
    increase = bool(re.search(r'\bincrease\b', ci))
    decrease = bool(re.search(r'\bdecrease\b', ci))
    if not np.isfinite(raw) or (increase and decrease):
        return np.nan, 'invalid_effect'
    kind = effect_type
    if kind == 'auto':
        if increase or decrease:
            kind = 'beta'
        elif re.fullmatch(r'\[[\d.e+\-\s,]+\]', ci):
            kind = 'or'
        else:
            return np.nan, 'unknown_effect_type'
    if kind == 'or':
        if raw <= 0 or increase or decrease:
            return np.nan, 'invalid_or'
        return np.log(raw), kind
    if increase and raw < 0:
        return np.nan, 'conflicting_beta_direction'
    return -abs(raw) if decrease else raw, kind


def _prepare_catalog(associations, trait, pvalue_text, effect_type):
    d = associations.copy() if isinstance(associations, pd.DataFrame) else pd.read_csv(
        associations, sep='\t', dtype='string')
    required = {'DISEASE/TRAIT', 'CHR_ID', 'CHR_POS',
                'STRONGEST SNP-RISK ALLELE', 'OR or BETA'}
    if not required.issubset(d):
        raise ValueError(f'Missing Catalog columns: {sorted(required - set(d))}')
    d['source_row'] = np.arange(1, len(d) + 1)
    if trait is not None:
        d = d.loc[d['DISEASE/TRAIT'].eq(trait)].copy()
    if pvalue_text is not None:
        if 'P-VALUE (TEXT)' not in d:
            raise ValueError('Catalog lacks P-VALUE (TEXT) for the requested filter.')
        d = d.loc[d['P-VALUE (TEXT)'].fillna('').eq(pvalue_text)].copy()
    if d.empty:
        raise ValueError('No associations match the trait/p-value annotation selection.')
    rows = []
    for row in d.to_dict('records'):
        row.update(status='pending', hm_chr=_chrom(row['CHR_ID']), hm_pos=np.nan,
                   reported_effect_allele='', rsID='', catalog_eaf=np.nan,
                   effect_allele='', other_allele='', effect_weight=np.nan,
                   orientation='', frequency_check='unavailable', vcf_effect_af=np.nan)
        pos = _number(row['CHR_POS'])
        if np.isfinite(pos) and pos > 0 and pos.is_integer():
            row['hm_pos'] = int(pos)
        else:
            row['status'] = 'invalid_coordinate'
        if not re.fullmatch(r'(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)', row['hm_chr']):
            row['status'] = 'invalid_coordinate'
        match = re.fullmatch(r'(rs\d+)-([ACGT])', _text(row['STRONGEST SNP-RISK ALLELE']))
        if match:
            row['rsID'], row['reported_effect_allele'] = match.groups()
        else:
            row['status'] = 'unsupported_or_unknown_risk_allele'
        current = re.sub(r'\.0+$', '', _text(row.get('SNP_ID_CURRENT', '')))
        if re.fullmatch(r'(?:rs)?\d+', current):
            row['current_rsID'] = 'rs' + current.removeprefix('rs')
        else:
            row['current_rsID'] = ''
        af = _number(row.get('RISK ALLELE FREQUENCY'))
        if 0 <= af <= 1:
            row['catalog_eaf'] = af
        weight, kind = _weight(row['OR or BETA'], row.get('95% CI (TEXT)'), effect_type)
        row['effect_weight'], row['effect_type'] = weight, kind
        if not np.isfinite(weight) and row['status'] == 'pending':
            row['status'] = kind
        rows.append(row)
    report = pd.DataFrame(rows)
    valid_pos = report['hm_pos'].notna()
    duplicated = report.loc[valid_pos].duplicated(['hm_chr', 'hm_pos'], keep=False)
    report.loc[duplicated.index[duplicated], 'status'] = 'duplicate_association_locus'
    return report


def _open_vcf(vcf):
    if hasattr(vcf, 'read'):
        return nullcontext(vcf)
    if str(vcf) == '-':
        return nullcontext(sys.stdin)
    if str(vcf).startswith('gs://'):
        raise ValueError('Pass an open text stream for gs://; e.g. hl.hadoop_open(uri, "r").')
    with open(vcf, 'rb') as handle:
        compressed = handle.read(2) == b'\x1f\x8b'
    return gzip.open(vcf, 'rt') if compressed else open(vcf, encoding='utf-8')


def _alt_dosage(fields, n_samples):
    fmt = fields[8].split(':')
    if 'GT' not in fmt:
        return None
    gt_index = fmt.index('GT')
    dosage = np.full(n_samples, np.nan, dtype=np.float32)
    for i, field in enumerate(fields[9:]):
        parts = field.split(':')
        gt = parts[gt_index] if gt_index < len(parts) else '.'
        alleles = re.split(r'[/|]', gt)
        if any(a not in {'0', '1', '.'} for a in alleles):
            raise ValueError('Invalid GT allele index at a biallelic target locus.')
        if len(alleles) == 2 and '.' not in alleles:
            dosage[i] = alleles.count('1')
    return dosage


def _read_targets(vcf, targets, frequency_samples):
    """Stream VCF once; retain GT arrays only for selected target coordinates."""
    samples, frequency_indices = None, None
    records = {}
    seen, conflicting = set(), set()
    with _open_vcf(vcf) as handle:
        for line in handle:
            if line.startswith(('##reference=', '##contig=')) and re.search(
                    r'GRCh3[67]|hg1[89]', line, re.I):
                raise ValueError('VCF header indicates an incompatible build; GRCh38 is required.')
            if line.startswith('#CHROM\t'):
                if samples is not None:
                    raise ValueError('VCF has more than one sample header.')
                samples = line.rstrip('\r\n').split('\t')[9:]
                if not samples or any(not s for s in samples) or len(set(samples)) != len(samples):
                    raise ValueError('VCF requires nonempty, unique sample IDs.')
                if frequency_samples is None:
                    frequency_indices = np.arange(len(samples))
                else:
                    wanted = {str(s) for s in frequency_samples}
                    if not wanted or not wanted.issubset(samples):
                        raise ValueError('Frequency sample list is empty or includes IDs absent from VCF.')
                    frequency_indices = np.array([i for i, s in enumerate(samples) if s in wanted])
                continue
            if line.startswith('#') or not line.strip():
                continue
            if samples is None:
                raise ValueError('VCF is missing its #CHROM sample header.')
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) < 8:
                raise ValueError('Malformed VCF record.')
            key = (_chrom(fields[0]), int(fields[1]))
            if key not in targets:
                continue
            seen.add(key)
            if len(fields) != 9 + len(samples):
                raise ValueError('Target VCF record has an inconsistent sample count.')
            if fields[6] not in {'PASS', '.'}:
                continue
            if key in conflicting:
                continue
            ref, alt = fields[3:5]
            status = 'pending'
            if ',' in alt:
                status = 'multiallelic_vcf'
            elif ref not in 'ACGT' or alt not in 'ACGT' or len(ref) != 1 or len(alt) != 1 or ref == alt:
                status = 'non_snv_vcf'
            dosage = _alt_dosage(fields, len(samples)) if status == 'pending' else None
            if status == 'pending' and dosage is None:
                status = 'missing_GT'
            ids = {x for x in fields[2].split(';') if re.fullmatch(r'rs\d+', x)}
            new = dict(ref=ref, alt=alt, dosage=dosage, status=status, ids=ids)
            if key in records:
                old = records[key]
                same = old['ref'] == ref and old['alt'] == alt and old['status'] == status
                same = same and ((old['dosage'] is None and dosage is None) or
                                 (old['dosage'] is not None and dosage is not None and
                                  np.array_equal(old['dosage'], dosage, equal_nan=True)))
                if not same or (ids and old['ids'] and ids != old['ids']):
                    conflicting.add(key)
                    records[key] = dict(status='conflicting_vcf_records')
                else:
                    old['ids'].update(ids)
            else:
                records[key] = new
    if samples is None:
        raise ValueError('VCF is missing its #CHROM sample header.')
    return samples, frequency_indices, records, seen


def _hail_reference_bases(targets):
    """Batch-query 1-based loci using the configured Hail GRCh38 sequence.

    Reuses the current Hail backend and any registered GRCh38 sequence. If no
    sequence is attached, register Hail's documented public GRCh38 FASTA/index.
    Configure Hail/requester-pays before calling this in an All of Us notebook.
    No VCF genotypes or participant IDs are sent to the reference lookup.
    """
    if not targets:
        return {}
    try:
        import hail as hl
    except ImportError as error:
        raise ValueError('reference_source="hail" requires Hail in the current runtime.') from error
    rg = hl.get_reference('GRCh38')
    if not rg.has_sequence():
        rg.add_sequence(GRCH38_FASTA, GRCH38_FAI)
    rows = []
    for chrom, pos in sorted(targets):
        contig = 'chrM' if chrom == 'MT' else 'chr' + chrom
        if contig in rg.lengths and 1 <= pos <= rg.lengths[contig]:
            rows.append(hl.Struct(chrom=chrom, pos=pos, contig=contig))
    if not rows:
        return {}
    ht = hl.Table.parallelize(
        rows, schema=hl.tstruct(chrom=hl.tstr, pos=hl.tint32, contig=hl.tstr), n_partitions=1)
    ht = ht.annotate(reference_base=hl.get_sequence(ht.contig, ht.pos, reference_genome='GRCh38'))
    return {(row.chrom, row.pos): row.reference_base for row in ht.collect()}


def _orient(row, record, frequency_indices, *, max_af_diff, min_af_gap,
            min_af_samples, palindromic_maf_max, min_af_call_rate):
    ref, alt, dosage = record['ref'], record['alt'], record['dosage']
    freq_dosage = dosage[frequency_indices]
    n_called = int(np.isfinite(freq_dosage).sum())
    call_rate = n_called / len(freq_dosage)
    alt_af = float(np.nanmean(freq_dosage, dtype=np.float64) / 2) if n_called else np.nan
    result = dict(vcf_ref=ref, vcf_alt=alt, vcf_alt_af=alt_af,
                  vcf_ref_af=1 - alt_af, af_n_called=n_called,
                  af_n_samples=len(freq_dosage), af_call_rate=call_rate,
                  vcf_major_allele=(alt if alt_af > .5 else ref if alt_af < .5 else ''),
                  vcf_effect_af=np.nan, af_difference=np.nan, direct_af_difference=np.nan,
                  complement_af_difference=np.nan, major_consistent=pd.NA)
    if not np.isfinite(dosage).any():
        return dict(result, status='all_calls_missing')
    reported = row['reported_effect_allele']
    candidates = []
    for orientation, effect in [('direct', reported), ('complement', reported.translate(COMPLEMENT))]:
        if effect in (ref, alt):
            af = alt_af if effect == alt else 1 - alt_af
            diff = abs(af - row['catalog_eaf'])
            result[orientation + '_af_difference'] = diff
            candidates.append((orientation, effect, af, diff))
    if not candidates:
        return dict(result, status='allele_mismatch')
    reliable_af = n_called >= min_af_samples and call_rate >= min_af_call_rate
    has_catalog_af = np.isfinite(row['catalog_eaf'])
    if len(candidates) == 2:
        if not has_catalog_af:
            return dict(result, status='palindromic_missing_catalog_af')
        if not reliable_af:
            return dict(result, status='palindromic_insufficient_calls')
        if (min(alt_af, 1 - alt_af) > palindromic_maf_max or
                min(row['catalog_eaf'], 1 - row['catalog_eaf']) > palindromic_maf_max):
            return dict(result, status='palindromic_near_half')
        candidates.sort(key=lambda c: c[3])
        if candidates[0][3] > max_af_diff:
            return dict(result, status='frequency_mismatch', frequency_check='mismatch')
        if candidates[1][3] - candidates[0][3] < min_af_gap or candidates[0][3] == candidates[1][3]:
            return dict(result, status='palindromic_ambiguous')
        check = 'consistent'
    elif has_catalog_af and reliable_af:
        if candidates[0][3] > max_af_diff:
            return dict(result, status='frequency_mismatch', frequency_check='mismatch')
        check = 'consistent'
    else:
        check = 'missing_catalog_af' if not has_catalog_af else 'insufficient_calls'
    orientation, effect, af, difference = candidates[0]
    if has_catalog_af and np.isfinite(af) and af != .5 and row['catalog_eaf'] != .5:
        result['major_consistent'] = bool((af > .5) == (row['catalog_eaf'] > .5))
    return dict(result, status='accepted', effect_allele=effect,
                other_allele=ref if effect == alt else alt, orientation=orientation,
                frequency_check=check, vcf_effect_af=af, af_difference=difference)


def score_gwas_vcf(vcf, associations, *, genome_build, trait=None,
                   pvalue_text=None, effect_type='auto', frequency_samples=None,
                   allow_partial=False, max_af_diff=.10, min_af_gap=.10,
                   min_af_samples=20, palindromic_maf_max=.42, min_af_call_rate=.9,
                   reference_source='vcf'):
    """Return (scores, harmonized_reference, audit_report) for a selected trait.

    Frequency inference assumes Catalog EAF and the selected frequency cohort
    are comparable. frequency_samples contains VCF sample IDs; every VCF sample
    is still scored. Unknown or ambiguous alleles are never assigned by majority
    alone. Weights stay unchanged under strand complementation. Missing individual
    GTs use the scoring-cohort mean; missing loci require allow_partial=True.
    reference_source='hail' first verifies VCF REF against Hail GRCh38 sequence.
    Missing/non-ACGT reference bases and REF mismatches are excluded, not repaired.
    The default 'vcf' retains the original behavior without a FASTA lookup.
    FASTA verification alone does not resolve the reported effect's strand at
    palindromic SNPs; the existing frequency checks still apply.
    """
    if genome_build != 'GRCh38':
        raise ValueError('Only GRCh38 is supported; no implicit liftover.')
    if reference_source not in {'vcf', 'hail'}:
        raise ValueError('reference_source must be vcf or hail.')
    for name, value in [('max_af_diff', max_af_diff), ('min_af_gap', min_af_gap),
                        ('min_af_call_rate', min_af_call_rate)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')
    if (not 0 <= palindromic_maf_max < .5 or not np.isfinite(min_af_samples)
            or min_af_samples < 1 or int(min_af_samples) != min_af_samples):
        raise ValueError('Require 0 <= palindromic_maf_max < 0.5 and positive integer min_af_samples.')
    if isinstance(frequency_samples, (str, bytes)):
        raise ValueError('frequency_samples must be a collection of IDs, not one string.')
    report = _prepare_catalog(associations, trait, pvalue_text, effect_type)
    targets = {(row.hm_chr, int(row.hm_pos)) for row in report.itertuples()
               if row.status == 'pending'}
    samples, freq_indices, records, seen = _read_targets(vcf, targets, frequency_samples)
    reference_bases = (_hail_reference_bases({key for key, record in records.items()
                                            if record['status'] == 'pending'})
                       if reference_source == 'hail' else {})
    report['reference_source'] = reference_source
    report['reference_base'] = ''
    report['reference_check'] = 'not_checked' if reference_source == 'hail' else 'not_requested'
    report['vcf_ref'], report['vcf_alt'] = '', ''
    accepted_dosages = []
    for index, row in report.iterrows():
        if row['status'] != 'pending':
            continue
        key = (row['hm_chr'], int(row['hm_pos']))
        if key not in records:
            report.at[index, 'status'] = 'filtered_only' if key in seen else 'absent'
            continue
        record = records[key]
        if record['status'] != 'pending':
            report.at[index, 'status'] = record['status']
            continue
        report.at[index, 'vcf_ref'], report.at[index, 'vcf_alt'] = record['ref'], record['alt']
        if reference_source == 'hail':
            base = _text(reference_bases.get(key)).upper()
            report.at[index, 'reference_base'] = base
            if len(base) != 1 or base not in 'ACGT':
                report.at[index, 'status'] = 'reference_unavailable'
                report.at[index, 'reference_check'] = 'missing_or_non_acgt'
                continue
            if base != record['ref']:
                report.at[index, 'status'] = 'reference_mismatch'
                report.at[index, 'reference_check'] = 'mismatch'
                continue
            report.at[index, 'reference_check'] = 'match'
        catalog_ids = {row['rsID'], row['current_rsID']} - {''}
        if record['ids'] and not catalog_ids.intersection(record['ids']):
            report.at[index, 'status'] = 'rsid_mismatch'
            continue
        details = _orient(row, record, freq_indices, max_af_diff=max_af_diff,
                          min_af_gap=min_af_gap, min_af_samples=min_af_samples,
                          palindromic_maf_max=palindromic_maf_max, min_af_call_rate=min_af_call_rate)
        for field, value in details.items():
            if field not in report:
                report[field] = pd.Series([pd.NA] * len(report), dtype='object')
            report.at[index, field] = value
        if details['status'] == 'accepted':
            dosage = record['dosage']
            accepted_dosages.append(dosage if details['effect_allele'] == record['alt'] else 2 - dosage)
    report['hm_pos'] = report['hm_pos'].astype('Int64')
    for field in ['vcf_alt_af', 'vcf_ref_af', 'vcf_effect_af', 'af_difference',
                  'direct_af_difference', 'complement_af_difference', 'af_call_rate']:
        report[field] = pd.to_numeric(report.get(field, pd.Series(np.nan, index=report.index)), errors='coerce').astype(float)
    for field in ['af_n_called', 'af_n_samples']:
        report[field] = pd.to_numeric(report.get(field, pd.Series(np.nan, index=report.index)), errors='coerce').astype('Int64')
    report['major_consistent'] = report.get('major_consistent', pd.Series(pd.NA, index=report.index)).astype('boolean')
    reference = report.loc[report['status'].eq('accepted')].copy().reset_index(drop=True)
    reference['allelefrequency_effect'] = reference['catalog_eaf']
    reference.attrs.update(HmPOS_build='GRCh38', 
                           score_type='exploratory_GWAS_associations', reference_source=reference_source)
    if reference.empty or (len(reference) != len(report) and not allow_partial):
        raise IncompleteGWASScoreError(
            f'{len(reference)}/{len(report)} selected associations are scorable. '
            'Inspect the report; allow_partial=True explicitly permits a reduced score.', report, reference)
    x = np.asarray(accepted_dosages, dtype=np.float64)
    weights = reference['effect_weight'].to_numpy(dtype=float)
    means = np.nanmean(x, axis=1)
    scores = pd.DataFrame({'person_id': samples,
                           'PRS_GWAS': weights @ np.where(np.isnan(x), means[:, None], x),
                           'PRS_GWAS_noimp': weights @ np.nan_to_num(x, nan=0.),
                           'n_called_variants': np.isfinite(x).sum(axis=0),
                           'n_imputed_variants': np.isnan(x).sum(axis=0)})
    scores['n_selected_associations'] = len(report)
    scores['n_scored_variants'] = len(reference)
    scores['complete_model'] = len(reference) == len(report)
    scores['reference_source'] = reference_source
    return scores, reference, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcf', required=True, help='Local VCF/VCF.gz/VCF.bgz or - for text stdin.')
    parser.add_argument('--gwas', required=True, help='GWAS Catalog associations TSV.')
    parser.add_argument('--trait', help='Optional exact DISEASE/TRAIT filter.')
    parser.add_argument('--pvalue-text', help='Optional exact P-VALUE (TEXT) filter; empty string selects missing.')
    parser.add_argument('--genome-build', choices=['GRCh38'], required=True)
    parser.add_argument('--reference-source', choices=['vcf', 'hail'], default='vcf',
                        help='hail: verify REF against Hail GRCh38 FASTA before inference; ALT comes from VCF.')
    parser.add_argument('--effect-type', choices=['auto', 'or', 'beta'], default='auto',
                        help='auto requires CI text to identify OR versus beta; verify the study model.')
    parser.add_argument('--frequency-samples', help='One VCF ID per line, no header; ideally ancestry-matched controls.')
    parser.add_argument('--max-af-diff', type=float, default=.10)
    parser.add_argument('--min-af-gap', type=float, default=.10)
    parser.add_argument('--min-af-samples', type=int, default=20)
    parser.add_argument('--min-af-call-rate', type=float, default=.9)
    parser.add_argument('--palindromic-maf-max', type=float, default=.42)
    parser.add_argument('--allow-partial', action='store_true', help='Permit a reduced exploratory score after exclusions.')
    parser.add_argument('--out-prefix', required=True)
    args = parser.parse_args(argv)
    paths = {kind: Path(args.out_prefix + suffix) for kind, suffix in
             [('scores', '.scores.tsv'), ('reference', '.harmonized.tsv'), ('report', '.variants.tsv')]}
    if any(path.exists() for path in paths.values()):
        parser.error('Output exists; choose a new --out-prefix.')
    frequency_samples = None
    if args.frequency_samples:
        frequency_samples = [s.strip() for s in Path(args.frequency_samples).read_text().splitlines() if s.strip()]
    try:
        scores, reference, report = score_gwas_vcf(
            args.vcf, args.gwas, genome_build=args.genome_build,
            trait=args.trait, pvalue_text=args.pvalue_text, effect_type=args.effect_type,
            frequency_samples=frequency_samples, allow_partial=args.allow_partial,
            max_af_diff=args.max_af_diff, min_af_gap=args.min_af_gap,
            min_af_samples=args.min_af_samples, min_af_call_rate=args.min_af_call_rate,
            palindromic_maf_max=args.palindromic_maf_max, reference_source=args.reference_source)
        exit_code = 0
    except IncompleteGWASScoreError as error:
        reference, report = error.reference, error.report
        scores, exit_code = None, 2
        print(str(error), file=sys.stderr)
    except ValueError as error:
        parser.error(str(error))
    paths['report'].parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(paths['report'], sep='\t', index=False, mode='x')
    with paths['reference'].open('x') as handle:
        handle.write('#HmPOS_build=GRCh38\n#score_type=exploratory_GWAS_associations\n')
        handle.write(f'#selected_associations={len(report)}\n#effect_type_mode={args.effect_type}\n')
        handle.write(f'#reference_source={args.reference_source}\n')
        handle.write(f'#max_af_diff={args.max_af_diff}\n#min_af_gap={args.min_af_gap}\n')
        handle.write(f'#palindromic_maf_max={args.palindromic_maf_max}\n')
        handle.write(f'#min_af_samples={args.min_af_samples}\n#min_af_call_rate={args.min_af_call_rate}\n')
        handle.write('#frequency_sample_source=' + ('provided_list' if frequency_samples is not None else 'all_vcf_samples') + '\n')
        reference.to_csv(handle, sep='\t', index=False)
    if scores is not None:
        scores.to_csv(paths['scores'], sep='\t', index=False, mode='x')
        print(f'Saved scores for {len(scores)} samples using {len(reference)}/{len(report)} associations.')
    else:
        print('Saved harmonized candidates and audit report; no score file was written.', file=sys.stderr)
    print(report['status'].value_counts().to_string())
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
