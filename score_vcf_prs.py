"""Score individual samples in a GRCh38 VCF using stage 03's PRS methods.

Accepts local VCF/gzip/BGZF files or open text streams (including Hail GCS
streams). Uses diploid GT hard calls, exact forward allele pairs and the
public PGS000004 GRCh38 definition. See VCF_PRS.md for examples and limits.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import gzip
import io
from pathlib import Path
import re
import sys
from urllib.request import urlopen

import numpy as np
import pandas as pd

from load_stage import load_stage

prs_method = load_stage('prs')


class IncompleteScoreError(ValueError):
    """Scoring failed completeness checks; report contains all model loci."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def load_definition(path=None):
    """Read the official harmonized PGS000004 file, retaining original loci."""
    source = str(path or prs_method.PGS_URL)
    if source.startswith(('https://', 'http://')):
        with urlopen(source, timeout=60) as handle:
            raw = handle.read()
    else:
        raw = Path(source).read_bytes()
    if raw.startswith(b'\x1f\x8b'):
        raw = gzip.decompress(raw)
    text = raw.decode('utf-8-sig')
    metadata = dict(line[1:].split('=', 1) for line in text.splitlines()
                    if line.startswith('#') and '=' in line)
    if metadata.get('pgs_id') != 'PGS000004' or metadata.get('HmPOS_build') != 'GRCh38':
        raise ValueError('Use the PGS000004 harmonized GRCh38 scoring file.')
    return pd.read_csv(io.StringIO(text), sep='\t', comment='#')


def _prepare_definition(definition):
    required = {'chr_name', 'chr_position', 'hm_chr', 'hm_pos',
                'effect_allele', 'other_allele', 'effect_weight',
                'allelefrequency_effect'}
    if not required.issubset(definition.columns):
        raise ValueError('Definition is missing required PGS Catalog columns.')
    d = definition.copy()
    for col in ['chr_name', 'hm_chr']:
        d[col] = d[col].astype(str).str.removeprefix('chr')
    for col in ['chr_position', 'hm_pos']:
        values = pd.to_numeric(d[col], errors='raise')
        if values.isna().any() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError('Definition positions must be positive integers.')
        d[col] = values.astype('int64')
    for col in ['effect_allele', 'other_allele']:
        d[col] = d[col].astype(str).str.upper()
        if not d[col].str.fullmatch('[ACGT]+').all():
            raise ValueError('Definition requires explicit A/C/G/T alleles.')
    d['effect_weight'] = pd.to_numeric(d.effect_weight, errors='raise')
    d['allelefrequency_effect'] = pd.to_numeric(d.allelefrequency_effect, errors='coerce')
    if not np.isfinite(d.effect_weight).all() or d.effect_allele.eq(d.other_allele).any():
        raise ValueError('Invalid weights or identical model alleles.')
    if d.duplicated(['hm_chr', 'hm_pos', 'effect_allele', 'other_allele']).any():
        raise ValueError('Duplicated model variant.')
    return prs_method.prepare_definition(d)


def _open_vcf(vcf):
    if hasattr(vcf, 'read'):
        return nullcontext(vcf)
    if str(vcf) == '-':
        return nullcontext(sys.stdin)
    if str(vcf).startswith('gs://'):
        raise ValueError('For gs://, pass an open hl.hadoop_open(uri, "r") text stream.')
    with open(vcf, 'rb') as handle:
        compressed = handle.read(2) == b'\x1f\x8b'
    return gzip.open(vcf, 'rt') if compressed else open(vcf, encoding='utf-8')


def _effect_dosages(fields, effect, other):
    alleles = [fields[3], *fields[4].split(',')]
    fmt = fields[8].split(':')
    if 'GT' not in fmt:
        raise ValueError('A matching VCF record has no GT field; DS-only input is unsupported.')
    gt_index = fmt.index('GT')
    effect_index, other_index = alleles.index(effect), alleles.index(other)
    allowed = {effect_index, other_index}
    dosage = np.full(len(fields) - 9, np.nan)
    for sample_index, sample in enumerate(fields[9:]):
        parts = sample.split(':')
        gt = parts[gt_index] if gt_index < len(parts) else '.'
        calls = re.split(r'[/|]', gt)
        if len(calls) != 2 or '.' in calls:
            continue
        if not all(x.isdigit() for x in calls):
            raise ValueError('Malformed GT in a matching VCF record.')
        indices = [int(x) for x in calls]
        if any(x >= len(alleles) for x in indices):
            raise ValueError('GT allele index exceeds the VCF allele count.')
        # A third allele is not silently treated as the model's other allele.
        if set(indices).issubset(allowed):
            dosage[sample_index] = indices.count(effect_index)
    return dosage


def score_vcf(vcf, definition=None, *, genome_build, allow_partial=False,
              resolve_by_af=False):
    """Return (per-sample scores, per-model-variant matching report).

    definition may be a PGS file path/URL or its DataFrame. Callers supplying a
    DataFrame must provide correctly harmonized GRCh38 hm_chr/hm_pos columns.
    Missing loci never imply reference genotypes. Missing individual calls use
    the stage-03 scoring cohort mean; entirely missing loci cannot be imputed.
    AF resolution is opt-in because the published reference AF is European.
    """
    if genome_build != 'GRCh38':
        raise ValueError('This wrapper supports GRCh38 VCFs only; no implicit liftover.')
    d = _prepare_definition(definition if isinstance(definition, pd.DataFrame)
                            else load_definition(definition))
    targets = defaultdict(list)
    for row in d.itertuples(index=False):
        targets[(row.hm_chr, row.hm_pos)].append(row)
    candidates = defaultdict(list)
    observed, passing = set(), set()
    samples = None
    with _open_vcf(vcf) as handle:
        for line in handle:
            if line.startswith('##reference=') and re.search(r'GRCh37|hg19', line, re.I):
                raise ValueError('VCF reference header indicates GRCh37/hg19, not GRCh38.')
            if line.startswith('#CHROM\t'):
                if samples is not None:
                    raise ValueError('VCF has more than one sample header.')
                samples = line.rstrip('\r\n').split('\t')[9:]
                if not samples or any(not s for s in samples) or len(set(samples)) != len(samples):
                    raise ValueError('VCF requires nonempty, unique sample IDs.')
                continue
            if line.startswith('#') or not line.strip():
                continue
            if samples is None:
                raise ValueError('VCF is missing its #CHROM sample header.')
            fields = line.rstrip('\r\n').split('\t')
            key = (fields[0].removeprefix('chr'), int(fields[1]))
            if key not in targets:
                continue
            if len(fields) != 9 + len(samples):
                raise ValueError('Matching VCF record has an inconsistent sample count.')
            for target in targets[key]:
                uid = target.variant_uid
                observed.add(uid)
                if fields[6] not in {'PASS', '.'}:
                    continue
                passing.add(uid)
                pair = {target.effect_allele, target.other_allele}
                alleles = [fields[3], *fields[4].split(',')]
                if fields[3] not in pair or not pair.issubset(alleles):
                    continue
                values = _effect_dosages(fields, target.effect_allele, target.other_allele)
                representation = (fields[3], fields[4])
                if not any(rep == representation and np.array_equal(x, values, equal_nan=True)
                           for rep, x in candidates[uid]):
                    candidates[uid].append((representation, values))
    if samples is None:
        raise ValueError('VCF is missing its #CHROM sample header.')

    report = d[['variant_uid', 'hm_chr', 'hm_pos', 'effect_allele', 'other_allele',
                'effect_weight', 'is_chek2_tagsnp']].copy()
    statuses, counts, n_called, flags, accepted = [], [], [], [], []
    for row in d.itertuples(index=False):
        choices = candidates[row.variant_uid]
        count, chosen = len(choices), None
        status = ('allele_mismatch' if row.variant_uid in passing else
                  'filtered_only' if row.variant_uid in observed else 'absent')
        if count == 1:
            chosen, status = 0, 'matched'
        elif count > 1:
            status = 'ambiguous'
            # Conflicting duplicate records cannot be resolved by cohort AF.
            representations = [rep for rep, _ in choices]
            if resolve_by_af and len(set(representations)) == count:
                chosen, _ = prs_method.resolve_af(
                    [values for _, values in choices], row.allelefrequency_effect)
                if chosen is not None:
                    status = 'af_resolved'
        called, flagged = 0, False
        if chosen is not None:
            values = choices[chosen][1]
            called = int(np.isfinite(values).sum())
            if called:
                accepted.append((row.variant_uid, values, status == 'matched'))
                flagged = (1 - called / len(samples)) > prs_method.AF_GATES['max_missing_rate_flag']
            else:
                status = 'all_calls_missing'
        statuses.append(status)
        counts.append(count)
        n_called.append(called)
        flags.append(flagged)
    report['status'] = statuses
    report['n_candidate_records'] = counts
    report['n_called_samples'] = n_called
    report['high_missing_rate_flag'] = flags
    if not accepted or (len(accepted) != len(d) and not allow_partial):
        raise IncompleteScoreError(
            f'{len(accepted)}/{len(d)} model loci are scorable. Inspect the matching report; '
            'allow_partial=True explicitly permits a reduced, unrescaled score.', report)

    indices = [uid for uid, _, _ in accepted]
    x = np.stack([values for _, values, _ in accepted])
    selected = d.iloc[indices]
    scores = prs_method.score_prs(x, selected.effect_weight, selected.is_chek2_tagsnp,
                                 direct_match=[direct for _, _, direct in accepted])
    scores.insert(0, 'sample_id', samples)
    scores['n_scored_loci'] = len(accepted)
    scores['n_scored_non_chek2_tag_loci'] = int((~selected.is_chek2_tagsnp).sum())
    scores['n_called_loci'] = np.isfinite(x).sum(axis=0)
    scores['n_imputed_loci'] = np.isnan(x).sum(axis=0)
    scores['n_missing_model_loci'] = len(d) - len(accepted)
    scores['complete_model'] = len(accepted) == len(d)
    return scores, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcf', required=True, help='Local VCF/VCF.gz/VCF.bgz, or - for text stdin.')
    parser.add_argument('--genome-build', required=True, choices=['GRCh38'])
    parser.add_argument('--definition', help='PGS000004 GRCh38 file/URL; defaults to the public PGS URL.')
    parser.add_argument('--out-prefix', required=True, help='Private output prefix, outside the source repository.')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--resolve-by-af', action='store_true', help='Opt into stage-03 reference-AF ambiguity resolution.')
    args = parser.parse_args(argv)
    scores_path = Path(args.out_prefix + '.scores.tsv')
    report_path = Path(args.out_prefix + '.variants.tsv')
    if any(p.exists() for p in [scores_path, report_path]):
        parser.error('Output already exists. Choose a new --out-prefix.')
    try:
        scores, report = score_vcf(args.vcf, args.definition, genome_build=args.genome_build,
                                   allow_partial=args.allow_partial, resolve_by_af=args.resolve_by_af)
    except IncompleteScoreError as error:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        error.report.to_csv(report_path, sep='\t', index=False)
        print(str(error), file=sys.stderr)
        print('Matching report saved. No score file was written.', file=sys.stderr)
        return 2
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, sep='\t', index=False)
    scores.to_csv(scores_path, sep='\t', index=False)
    print('Saved individual scores and the matching report to the requested output prefix.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
