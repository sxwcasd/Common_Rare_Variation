"""Synthetic VCF checks; contains no participant data."""
import gzip
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_vcf_prs import IncompleteScoreError, load_definition, main, score_vcf, prs_method


def definition(n=313):
    d = pd.DataFrame({'chr_name': ['1'] * n, 'chr_position': np.arange(1000, 1000 + n),
                      'hm_chr': ['1'] * n, 'hm_pos': np.arange(1000, 1000 + n),
                      'effect_allele': ['G'] * n, 'other_allele': ['A'] * n,
                      'effect_weight': np.zeros(n), 'allelefrequency_effect': [0.1] * n})
    d.loc[:3, 'effect_weight'] = [0.2, -0.3, 0.4, 0.5]
    d.loc[1, ['other_allele', 'effect_allele']] = ['C', 'T']
    for i, pos, weight in [(n - 2, 29203724, 0.7), (n - 1, 29551872, 0.9)]:
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
                      + '\n'.join([replacements.get(i, record(d, i)) for i in range(len(d))
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

    def test_configurable_stage03_names_preserve_score_calculations(self):
        scores = prs_method.score_prs(
            [[0, 1], [2, np.nan]], [0.2, 0.7], [False, True], [True, False],
            n_model_loci=5218, n_model_chek2_tag_loci=1)
        self.assertEqual(list(scores), ['PRS5218', 'PRS5218_no_CHEK2_tags',
            'PRS5218_noimp', 'PRS5218_no_CHEK2_tags_noimp',
            'PRS5218_no_CHEK2_tags_direct_only'])
        np.testing.assert_allclose(scores.PRS5218, [1.4, 1.6])
        np.testing.assert_allclose(scores.PRS5218_noimp, [1.4, 0.2])
        np.testing.assert_allclose(scores.PRS5218_no_CHEK2_tags, [0, 0.2])
        np.testing.assert_allclose(scores.PRS5218_no_CHEK2_tags_direct_only, [0, 0.2])

    def test_stage03_include_mode_retains_tags_in_all_scores(self):
        scores = prs_method.score_prs(
            [[0, 1], [2, np.nan]], [0.2, 0.7], [False, True], [True, True],
            n_model_loci=5218, n_model_chek2_tag_loci=1, chek2_mode='include')
        self.assertEqual(list(scores), ['PRS5218', 'PRS5218_noimp', 'PRS5218_direct_only'])
        np.testing.assert_allclose(scores.PRS5218, [1.4, 1.6])
        np.testing.assert_allclose(scores.PRS5218_noimp, [1.4, 0.2])
        np.testing.assert_allclose(scores.PRS5218_direct_only, [1.4, 1.6])
        without_direct = prs_method.score_prs(
            [[0, 1], [2, np.nan]], [0.2, 0.7], [False, True],
            n_model_loci=5218, n_model_chek2_tag_loci=1, chek2_mode='include')
        self.assertEqual(list(without_direct), ['PRS5218', 'PRS5218_noimp'])

    def test_vcf_include_mode_preserves_full_scores_report_and_qc(self):
        changes = {0: record(self.d, 0, ('0/1', '1/1')),
                   311: record(self.d, 311, ('./.', '1/1')),
                   312: record(self.d, 312, ('0/1', '0/0'))}
        both, both_report = self.score(vcf(self.d, changes))
        scores, report = self.score(vcf(self.d, changes), chek2_mode='include')
        self.assertEqual([c for c in scores if c.startswith('PRS')],
                         ['PRS313', 'PRS313_noimp', 'PRS313_direct_only'])
        np.testing.assert_allclose(scores.PRS313, [2.5, 1.8])
        np.testing.assert_allclose(scores.PRS313_noimp, [1.1, 1.8])
        np.testing.assert_allclose(scores.PRS313_direct_only, [2.5, 1.8])
        pd.testing.assert_frame_equal(scores[both.columns.intersection(scores.columns)],
                                      both[both.columns.intersection(scores.columns)])
        pd.testing.assert_frame_equal(report, both_report)
        self.assertEqual(int(report.is_chek2_tagsnp.sum()), 2)
        self.assertTrue(scores.n_model_chek2_tag_loci.eq(2).all())
        self.assertTrue(scores.n_scored_non_chek2_tag_loci.eq(311).all())

    def test_include_direct_only_omits_af_resolved_locus_but_retains_tag(self):
        self.d.loc[0, 'allelefrequency_effect'] = 0.25
        changes = {0: record(self.d, 0, ('0/1', '0/0')),
                   311: record(self.d, 311, ('1/1', '1/1'))}
        extra = record(self.d, 0, ('0/0', '0/0'), ref='G', alt='A')
        scores, report = self.score(vcf(self.d, changes, extras=[extra]),
                                     resolve_by_af=True, chek2_mode='include')
        self.assertEqual(report.loc[0, 'status'], 'af_resolved')
        self.assertEqual(report.loc[311, 'status'], 'matched')
        np.testing.assert_allclose(scores.PRS313, [1.6, 1.4])
        np.testing.assert_allclose(scores.PRS313_direct_only, [1.4, 1.4])

    def test_include_mode_does_not_bypass_missing_tag_checks(self):
        with self.assertRaises(IncompleteScoreError):
            self.score(vcf(self.d, omit=[311]), chek2_mode='include')
        scores, report = self.score(vcf(self.d, omit=[311]),
                                     chek2_mode='include', allow_partial=True)
        self.assertIn('PRS313', scores)
        self.assertNotIn('PRS311', scores)
        self.assertFalse(scores.complete_model.any())
        self.assertTrue(scores.n_missing_model_loci.eq(1).all())
        self.assertEqual(report.loc[311, 'status'], 'absent')

    def test_invalid_chek2_modes_are_rejected_by_both_apis_and_cli(self):
        for mode in ['invalid', None, True]:
            with self.assertRaisesRegex(ValueError, 'chek2_mode'):
                prs_method.score_prs([[0]], [0.2], [False], chek2_mode=mode)
            with self.assertRaisesRegex(ValueError, 'chek2_mode'):
                self.score(vcf(self.d), chek2_mode=mode)
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(['--vcf', 'unused.vcf', '--genome-build', 'GRCh38',
                  '--out-prefix', 'unused', '--chek2-mode', 'invalid'])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('invalid choice', stderr.getvalue())

    def test_partial_model_names_use_full_definition_not_matched_loci(self):
        d = definition(5218)
        # This synthetic 5218-locus model includes only one of the two tag sites.
        d.loc[5217, ['chr_name', 'hm_chr', 'chr_position', 'hm_pos']] = ['1', '1', 6217, 6217]
        d.attrs['pgs_id'] = 'PGS000015'
        stream = vcf(d, {0: record(d, 0, ('0/1', '1/1'))}, omit=range(2, len(d)))
        scores, report = score_vcf(stream, d, genome_build='GRCh38', allow_partial=True)
        self.assertIn('PRS5218', scores)
        self.assertIn('PRS5218_no_CHEK2_tags', scores)
        self.assertNotIn('PRS313', scores)
        self.assertNotIn('PRS311', scores)
        self.assertNotIn('PRS2', scores)
        self.assertTrue(scores.n_model_loci.eq(5218).all())
        self.assertTrue(scores.n_model_chek2_tag_loci.eq(1).all())
        self.assertTrue(scores.n_scored_loci.eq(2).all())
        self.assertTrue(scores.n_missing_model_loci.eq(5216).all())
        self.assertTrue(scores.pgs_id.eq('PGS000015').all())
        self.assertFalse(scores.complete_model.any())
        self.assertEqual(len(report), 5218)
        np.testing.assert_allclose(scores.PRS5218, [0.2, 0.4])

    def test_non313_model_without_tag_sites_has_explicit_names(self):
        d = self.d.iloc[:4].copy()
        scores, _ = score_vcf(vcf(d), d, genome_build='GRCh38')
        self.assertIn('PRS4_no_CHEK2_tags', scores)
        self.assertTrue(scores.n_model_chek2_tag_loci.eq(0).all())
        np.testing.assert_array_equal(scores.PRS4, scores.PRS4_no_CHEK2_tags)
        self.assertTrue(scores.complete_model.all())
        self.assertTrue(scores.pgs_id.isna().all())

    def test_invalid_model_counts_are_rejected(self):
        for total, tags in [(1, 0), (2, 3), (2.0, 1), (True, 1), (2, 0)]:
            with self.assertRaises(ValueError):
                prs_method.score_prs([[0], [1]], [0.2, 0.7], [False, True],
                    n_model_loci=total, n_model_chek2_tag_loci=tags)

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
            self.assertTrue(result.pgs_id.eq('PGS000004').all())
            self.assertTrue(result.n_model_loci.eq(313).all())
            self.assertIn('PRS311', result)
            self.assertEqual(len(pd.read_csv(root / 'result.variants.tsv', sep='\t')), 313)
            path.write_text(vcf(self.d, omit=[0]).getvalue())
            args[-1] = str(root / 'incomplete')
            self.assertEqual(main(args), 2)
            self.assertTrue((root / 'incomplete.variants.tsv').exists())
            self.assertFalse((root / 'incomplete.scores.tsv').exists())

    def test_custom_model_metadata_drives_cli_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = self.d.iloc[:4].copy()
            model = root / 'definition.txt.gz'
            model.write_bytes(gzip.compress(('#pgs_id=PGS_TEST\n#HmPOS_build=GRCh38\n'
                + d.to_csv(sep='\t', index=False)).encode()))
            self.assertEqual(load_definition(model).attrs['pgs_id'], 'PGS_TEST')
            path = root / 'samples.vcf'
            path.write_text(vcf(d).getvalue())
            self.assertEqual(main(['--vcf', str(path), '--definition', str(model),
                '--genome-build', 'GRCh38', '--out-prefix', str(root / 'custom')]), 0)
            scores = pd.read_csv(root / 'custom.scores.tsv', sep='\t')
            self.assertIn('PRS4', scores)
            self.assertNotIn('PRS313', scores)
            self.assertTrue(scores.pgs_id.eq('PGS_TEST').all())

    def test_cli_include_mode_writes_tag_inclusive_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'definition.txt'
            model.write_text('#pgs_id=PGS000004\n#HmPOS_build=GRCh38\n'
                             + self.d.to_csv(sep='\t', index=False))
            path = root / 'samples.vcf'
            path.write_text(vcf(self.d, {311: record(self.d, 311, ('1/1', '0/0'))}).getvalue())
            self.assertEqual(main(['--vcf', str(path), '--definition', str(model),
                '--genome-build', 'GRCh38', '--out-prefix', str(root / 'include'),
                '--chek2-mode', 'include']), 0)
            scores = pd.read_csv(root / 'include.scores.tsv', sep='\t')
            self.assertEqual([c for c in scores if c.startswith('PRS')],
                             ['PRS313', 'PRS313_noimp', 'PRS313_direct_only'])
            np.testing.assert_allclose(scores.PRS313_direct_only, [1.4, 0])
            self.assertTrue(scores.n_model_chek2_tag_loci.eq(2).all())
            report = pd.read_csv(root / 'include.variants.tsv', sep='\t')
            self.assertEqual(int(report.is_chek2_tagsnp.sum()), 2)


if __name__ == '__main__':
    unittest.main()
