"""GRCh38 matching, AF disambiguation and effect-allele dosage scoring.

Public PGS definition coordinates are retained; no observed match list is shipped.
Only exact forward allele pairs are considered. No implicit strand rescue.
PGS000004 remains the default model; score names use the full model size.
"""
import numpy as np
import pandas as pd
from functions.prs import _orient_effect_allele, _decode_bed_block

AF_GATES = {'d_min_max': 0.1, 'd_gap_min': 0.1, 'd_ratio_min': 3.0,
            'd_min_ratio_floor': 0.01, 'complementary_af_tol': 0.05,
            'degenerate_af_ref_low': 0.4, 'degenerate_af_ref_high': 0.6,
            'max_missing_rate_flag': 0.1}
PGS_URL = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000004/ScoringFiles/Harmonized/PGS000004_hmPOS_GRCh38.txt.gz'
CHEK2_MODES = ('both', 'include')


def prepare_definition(definition):
    d = definition.copy().reset_index(drop=True)
    #if len(d) != 313:
    #    raise ValueError('Unexpected public PGS definition size')
    d['variant_uid'] = np.arange(len(d))
    d['is_chek2_tagsnp'] = [(str(c), int(p)) in {('22', 29203724), ('22', 29551872)}
                           for c, p in zip(d.chr_name, d.chr_position)]
    #if d.is_chek2_tagsnp.sum() != 2:
    #    raise ValueError('Public CHEK2 tag definition mismatch')
    return d


def match_bim(definition, bim):
    """bim has chrom, pos, A1, A2, bim_id, row_idx in original BED order."""
    rows = []
    for target in definition.itertuples(index=False):
        at = bim.loc[bim.chrom.astype(str).str.removeprefix('chr').eq(str(target.hm_chr))
                     & bim.pos.eq(int(target.hm_pos))]
        for row in at.itertuples(index=False):
            expected = {f'chr{target.hm_chr}:{target.hm_pos}:{target.other_allele}:{target.effect_allele}',
                        f'chr{target.hm_chr}:{target.hm_pos}:{target.effect_allele}:{target.other_allele}'}
            if row.bim_id not in expected:
                continue
            orient = _orient_effect_allele({'effect_allele': target.effect_allele, 'A1': row.A1, 'A2': row.A2})
            if orient == 'MISMATCH':
                continue
            rows.append({**row._asdict(), 'variant_uid': target.variant_uid, 'effect_is': orient})
    return pd.DataFrame(rows, columns=[*bim.columns, 'variant_uid', 'effect_is']).drop_duplicates(['variant_uid', 'chrom', 'row_idx'])


def resolve_af(candidate_dosages, reference_af, gates=AF_GATES):
    """Rows=candidate allele representations, columns=cohort samples.

    Return the accepted row index (or None) and a high-missing flag. Cohort AF
    remains in memory. Missingness >0.1 is a flag, not an exclusion.
    """
    x = np.asarray(candidate_dosages, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(reference_af):
        return None, False
    count = np.isfinite(x).sum(axis=1)
    af = np.divide(np.nansum(x, axis=1), 2 * count,
                   out=np.full(len(x), np.nan), where=count > 0)
    distance = np.abs(af - reference_af)
    order = np.argsort(np.where(np.isfinite(distance), distance, np.inf), kind='stable')
    best, second = distance[order[:2]]
    if not np.isfinite(best) or not np.isfinite(second):
        return None, False
    mirror = (len(x) == 2 and abs(af.sum() - 1) <= gates['complementary_af_tol']
              and gates['degenerate_af_ref_low'] <= reference_af <= gates['degenerate_af_ref_high'])
    ratio = best < gates['d_min_ratio_floor'] or second >= gates['d_ratio_min'] * best
    if mirror or best > gates['d_min_max'] or second - best < gates['d_gap_min'] or not ratio:
        return None, False
    chosen = int(order[0])
    return chosen, bool(np.isnan(x[chosen]).mean() > gates['max_missing_rate_flag'])


def decode_effect_dosage(raw_bytes, sample_count, effect_is):
    if len(raw_bytes) != (sample_count + 3) // 4 or effect_is not in {'A1', 'A2'}:
        raise ValueError('Invalid BED record or orientation')
    a1 = _decode_bed_block(raw_bytes, sample_count)
    return a1 if effect_is == 'A1' else 2 - a1


def score_prs(effect_dosage, weights, is_chek2_tag, direct_match=None, *,
              n_model_loci=313, n_model_chek2_tag_loci=2, chek2_mode='both'):
    """Variants x samples; main mean imputation uses the scoring cohort.

    noimp means a zero contribution from missing calls, matching SCORE_SUM with
    no-mean-imputation. It does not imply complete-case sample removal.

    Model counts describe the full definition, before filtering to scorable
    loci. Defaults retain the original PRS313/PRS311 API. Other model sizes
    use PRS<N> and PRS<N>_no_CHEK2_tags, with the same score calculations.

    chek2_mode='both' returns full and CHEK2-tag-excluded scores (the default).
    chek2_mode='include' retains the original weights for all reported scores,
    including PRS<N>_direct_only, and omits the tag-excluded score columns.
    Direct-only scores still omit AF-resolved matches in either mode.
    """
    if chek2_mode not in CHEK2_MODES:
        raise ValueError(f'chek2_mode must be one of {CHEK2_MODES}')
    x = np.asarray(effect_dosage, dtype=float)
    w = np.asarray(weights, dtype=float)
    tags = np.asarray(is_chek2_tag, dtype=bool)
    if x.ndim != 2 or x.shape[0] != len(w) or len(tags) != len(w):
        raise ValueError('PRS dimensions differ')
    for name, value in [('n_model_loci', n_model_loci),
                        ('n_model_chek2_tag_loci', n_model_chek2_tag_loci)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer from the full model definition')
    if n_model_loci <= 0 or not 0 <= n_model_chek2_tag_loci <= n_model_loci:
        raise ValueError('Invalid full-model locus counts')
    if len(w) > n_model_loci or tags.sum() > n_model_chek2_tag_loci:
        raise ValueError('Scored loci exceed model counts; supply the full definition counts')
    if not np.isfinite(w).all() or not np.isin(x[~np.isnan(x)], [0, 1, 2]).all():
        raise ValueError('Invalid weights or diploid calls')
    if np.isnan(x).all(axis=1).any():
        raise ValueError('A scoring locus is entirely missing')
    imputed = np.where(np.isnan(x), np.nanmean(x, axis=1)[:, None], x)
    zero = np.nan_to_num(x, nan=0.0)
    full_name = f'PRS{n_model_loci}'
    result = {full_name: w @ imputed}
    direct_name, direct_weights = full_name, w
    if chek2_mode == 'both':
        reduced = np.where(tags, 0, w)
        reduced_name = ('PRS311' if (n_model_loci, n_model_chek2_tag_loci) == (313, 2)
                        else f'{full_name}_no_CHEK2_tags')
        result[reduced_name] = reduced @ imputed
        direct_name, direct_weights = reduced_name, reduced
    result[f'{full_name}_noimp'] = w @ zero
    if chek2_mode == 'both':
        result[f'{reduced_name}_noimp'] = reduced @ zero
    if direct_match is not None:
        mask = np.asarray(direct_match, dtype=bool)
        result[f'{direct_name}_direct_only'] = direct_weights[mask] @ imputed[mask]
    return pd.DataFrame(result)


def plink_score_command(plink2, bed_prefix, score_file, output_prefix, mean_impute=True):
    command = [plink2, '--bfile', str(bed_prefix), '--score', str(score_file),
               '1', '2', '3', 'header', 'cols=+scoresums']
    if not mean_impute:
        command.append('no-mean-imputation')
    return command + ['--out', str(output_prefix)]


def cross_check_scores(numpy_scores, plink_scores, tolerance=1e-8):
    """Privately compare aligned per-person SCORE_SUM outputs; never print scores."""
    if numpy_scores.index.has_duplicates or not numpy_scores.index.equals(plink_scores.index):
        raise ValueError('PLINK comparison membership/order differs')
    if set(numpy_scores.columns) != set(plink_scores.columns):
        raise ValueError('PLINK comparison score definitions differ')
    a = numpy_scores.to_numpy(dtype=float)
    b = plink_scores.loc[:, numpy_scores.columns].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite PLINK comparison input')
    if not np.allclose(a, b, rtol=0, atol=tolerance):
        raise ValueError('Independent PLINK score check failed')
    return True
