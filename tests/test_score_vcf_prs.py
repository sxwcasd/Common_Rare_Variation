"""Synthetic VCF checks; contains no participant data."""
import gzip
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_vcf_prs import IncompleteScoreError, load_definition, main, score_vcf


def definition():
    d = pd.DataFrame({'chr_name': ['1'] * 313, 'chr_position': np.arange(1000, 1313),
                      'hm_chr': ['1'] * 313, 'hm_pos': np.arange(1000, 1313),
                      'effect_allele': ['G'] * 313, 'other_allele': ['A'] * 313,
                      'effect_weight': np.zeros(313), 'allelefrequency_effect': [0.1] * 313})
    d.loc[:3, 'effect_weight'] = [0.2, -0.3, 0.4, 0.5]
    d.loc[1, ['other_allele', 'effect_allele']] = ['C', 'T']
    for i, pos, weight in [(311, 29203724, 0.7), (312, 29551872, 0.9)]:
        d.loc[i, ['chr_name', 'hm_chr', 'chr_position', 'hm_pos', 'effect_weight']] = ['22', '22', pos, pos, weight]
    return d


def record(d, i, gt=('0/0', '0/0'), ref=None, alt=None, filt='PASS', fmt='GT'):
    r = d.iloc[i]
    return '\t'.join([f'chr{r.hm_chr}', str(r.hm_pos), '.', ref or r.other_allele,
                      alt or r.effect_allele, '.', filt, '.', fmt, *gt])


def vcf(d, replacements=None, omit=(), extras=()):
    replacements = replacements or {}
    return io.StringIO('##fileformat=VCFv4.2\n##reference=GRCh38\n'
                      '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t001\t002\n'
                      + '\n'.join([replacements.get(i, record(d, i)) for i in range(313)
                                   if i not in omit] + list(extras)) + '\n')


class VCFScoringTests(unittest.TestCase):
    def setUp(self):
        self.d = definition()

    def score(self, stream, **kwargs):
        return score_vcf(stream, self.d, genome_build='GRCh38', **kwargs)

    def test_orientation_phase_multiallelic_and_imputation(self):
        changes = {
            0: record(self.d, 0, ('0/1', '1|1')),
            1: record(self.d, 1, ('0/1', '1/1'), ref='T', alt='C'),
            2: record(self.d, 2, ('0/2', '2/2'), alt='C,G'),
            3: record(self.d, 3, ('./.', '0/1')),
            311: record(self.d, 311, ('1/1', '0/0')),
            312: record(self.d, 312, ('0/1', '0/1')),
        }
        scores, report = self.score(vcf(self.d, changes))
        np.testing.assert_allclose(scores.PRS313, [3.1, 2.6])
        np.testing.assert_allclose(scores.PRS311, [0.8, 1.7])
        np.testing.assert_allclose(scores.PRS313_noimp, [2.6, 2.6])
        np.testing.assert_allclose(scores.PRS311_noimp, [0.3, 1.7])
        self.assertEqual(scores.sample_id.tolist(), ['001', '002'])
        self.assertEqual(scores.n_imputed_loci.tolist(), [1, 0])
        self.assertTrue(scores.complete_model.all())
        self.assertTrue(report.loc[3, 'high_missing_rate_flag'])

    def test_absent_is_not_homozygous_reference(self):
        with self.assertRaises(IncompleteScoreError) as error:
            self.score(vcf(self.d, omit=[0]))
        self.assertEqual(error.exception.report.loc[0, 'status'], 'absent')
        scores, _ = self.score(vcf(self.d, omit=[0]), allow_partial=True)
        self.assertTrue(scores.n_missing_model_loci.eq(1).all())
        self.assertFalse(scores.complete_model.any())

    def test_all_missing_locus_not_fabricated(self):
        with self.assertRaises(IncompleteScoreError) as error:
            self.score(vcf(self.d, {0: record(self.d, 0, ('./.', '0/.'))}))
        self.assertEqual(error.exception.report.loc[0, 'status'], 'all_calls_missing')

    def test_third_allele_is_missing_not_other_allele(self):
        scores, _ = self.score(vcf(self.d, {2: record(self.d, 2, ('1/2', '2/2'), alt='C,G')}))
        np.testing.assert_allclose(scores.PRS313, [0.8, 0.8])
        self.assertEqual(scores.n_imputed_loci.tolist(), [1, 0])

    def test_identical_duplicates_score_once(self):
        scores, report = self.score(vcf(self.d, extras=[record(self.d, 0)]))
        self.assertTrue(scores.n_scored_loci.eq(313).all())
        self.assertEqual(report.loc[0, 'n_candidate_records'], 1)

    def test_conflicting_duplicates_rejected_even_with_af(self):
        with self.assertRaises(IncompleteScoreError) as error:
            self.score(vcf(self.d, extras=[record(self.d, 0, ('1/1', '1/1'))]), resolve_by_af=True)
        self.assertEqual(error.exception.report.loc[0, 'status'], 'ambiguous')

    def test_reference_af_resolution_is_explicit(self):
        extra = record(self.d, 0, ('0/0', '0/0'), ref='G', alt='A')
        with self.assertRaises(IncompleteScoreError):
            self.score(vcf(self.d, extras=[extra]))
        scores, report = self.score(vcf(self.d, extras=[extra]), resolve_by_af=True)
        self.assertEqual(report.loc[0, 'status'], 'af_resolved')
        np.testing.assert_allclose(scores.PRS313, [0, 0])

    def test_filtered_and_mismatched_are_reported(self):
        with self.assertRaises(IncompleteScoreError) as error:
            self.score(vcf(self.d, {0: record(self.d, 0, filt='LowQual'),
                                    2: record(self.d, 2, alt='T')}))
        self.assertEqual(error.exception.report.loc[0, 'status'], 'filtered_only')
        self.assertEqual(error.exception.report.loc[2, 'status'], 'allele_mismatch')

    def test_ds_only_and_bad_allele_index_rejected(self):
        for replacement in [record(self.d, 0, ('1', '2'), fmt='DS'),
                            record(self.d, 0, ('0/2', '0/0'))]:
            with self.assertRaises(ValueError):
                self.score(vcf(self.d, {0: replacement}))

    def test_build_mismatch_and_duplicate_samples_rejected(self):
        original = vcf(self.d).getvalue()
        for text in [original.replace('GRCh38', 'GRCh37'),
                     original.replace('\t001\t002', '\t001\t001')]:
            with self.assertRaises(ValueError):
                self.score(io.StringIO(text))

    def test_compressed_file_and_open_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.vcf.bgz'
            path.write_bytes(gzip.compress(vcf(self.d).getvalue().encode()))
            scores, _ = self.score(path)
            self.assertEqual(len(scores), 2)
        stream = vcf(self.d)
        self.score(stream)
        self.assertFalse(stream.closed)

    def test_cli_writes_scores_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'definition.txt'
            model.write_text('#pgs_id=PGS000004\n#HmPOS_build=GRCh38\n' + self.d.to_csv(sep='\t', index=False))
            path = root / 'samples.vcf'
            path.write_text(vcf(self.d).getvalue())
            self.assertEqual(len(load_definition(model)), 313)
            args = ['--vcf', str(path), '--definition', str(model), '--genome-build', 'GRCh38',
                    '--out-prefix', str(root / 'result')]
            self.assertEqual(main(args), 0)
            result = pd.read_csv(root / 'result.scores.tsv', sep='\t', dtype={'sample_id': str})
            self.assertEqual(result.sample_id.tolist(), ['001', '002'])
            self.assertEqual(len(pd.read_csv(root / 'result.variants.tsv', sep='\t')), 313)
            path.write_text(vcf(self.d, omit=[0]).getvalue())
            args[-1] = str(root / 'incomplete')
            self.assertEqual(main(args), 2)
            self.assertTrue((root / 'incomplete.variants.tsv').exists())
            self.assertFalse((root / 'incomplete.scores.tsv').exists())


if __name__ == '__main__':
    unittest.main()
