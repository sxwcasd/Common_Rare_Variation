"""Independent synthetic checks for GWAS/VCF allele harmonization and scoring.

No participant data or external services are used.
"""
import gzip
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_gwas_vcf_prs import (IncompleteGWASScoreError, main, score_gwas_vcf,
                               _hail_reference_bases, GRCH38_FASTA, GRCH38_FAI)


def association(pos=100, effect="A", weight=0.2, frequency="NR", **overrides):
    row = {
        "STUDY ACCESSION": "GCST_TEST",
        "DISEASE/TRAIT": "Breast cancer",
        "CHR_ID": "1",
        "CHR_POS": str(pos),
        "STRONGEST SNP-RISK ALLELE": f"rs{pos}-{effect}",
        "SNP_ID_CURRENT": str(pos),
        "OR or BETA": str(weight),
        "95% CI (TEXT)": "[0.1-0.3] unit increase",
        "RISK ALLELE FREQUENCY": str(frequency),
        "P-VALUE (TEXT)": "(EA)",
    }
    row.update(overrides)
    return row


def record(pos=100, ref="C", alt="A", gt=("0/1", "0/0"), *,
           snp_id=".", filt="PASS", fmt="GT"):
    return "\t".join(["chr1", str(pos), snp_id, ref, alt, ".", filt, ".", fmt, *gt])


def vcf(records, samples=("001", "002"), build="GRCh38"):
    return io.StringIO(
        f"##fileformat=VCFv4.2\n##reference={build}\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples) + "\n" + "\n".join(records) + "\n"
    )


class GWASVCFTests(unittest.TestCase):
    def score(self, rows, records, *, samples=("001", "002"), **kwargs):
        options = dict(study="GCST_TEST", genome_build="GRCh38",
                       min_af_samples=1, min_af_call_rate=0)
        options.update(kwargs)
        return score_gwas_vcf(vcf(records, samples), pd.DataFrame(rows), **options)

    def assert_excluded(self, rows, records, **kwargs):
        with self.assertRaises(IncompleteGWASScoreError) as caught:
            self.score(rows, records, **kwargs)
        self.assertFalse(caught.exception.report["status"].eq("accepted").all())
        return caught.exception.report

    def test_ref_alt_complement_and_missing_genotype_scores(self):
        rows = [association(100, "A", 0.2), association(101, "C", 0.3),
                association(102, "T", 0.4), association(103, "A", 0.5)]
        records = [record(100, gt=("0/1", "1|1")), record(101, gt=("0/0", "0|1")),
                   record(102, gt=("1|0", "0/0")), record(103, gt=("./.", "0/1"))]
        scores, reference, report = self.score(rows, records)
        np.testing.assert_allclose(scores.PRS_GWAS, [1.7, 1.2])
        np.testing.assert_allclose(scores.PRS_GWAS_noimp, [1.2, 1.2])
        self.assertEqual(scores.person_id.tolist(), ["001", "002"])
        self.assertEqual(scores.n_called_variants.tolist(), [3, 4])
        self.assertEqual(scores.n_imputed_variants.tolist(), [1, 0])
        self.assertTrue(scores.complete_model.all())
        self.assertEqual(reference.effect_allele.tolist(), ["A", "C", "A", "A"])
        self.assertEqual(reference.other_allele.tolist(), ["C", "A", "C", "C"])
        self.assertEqual(report.orientation.tolist(), ["direct", "direct", "complement", "direct"])
        self.assertTrue(report.status.eq("accepted").all())

    def test_palindromic_frequency_resolves_both_orientations(self):
        rows = [association(100, "A", 0.2, frequency=0.875),
                association(101, "A", 0.3, frequency=0.125)]
        calls = ("0/1", "0/0", "0/0", "0/0")
        records = [record(p, "A", "T", calls) for p in (100, 101)]
        scores, reference, report = self.score(rows, records, samples=("a", "b", "c", "d"))
        self.assertEqual(reference.effect_allele.tolist(), ["A", "T"])
        self.assertEqual(reference.other_allele.tolist(), ["T", "A"])
        self.assertEqual(report.orientation.tolist(), ["direct", "complement"])
        np.testing.assert_allclose(report.vcf_alt_af.to_numpy(dtype=float), [0.125, 0.125])
        np.testing.assert_allclose(report.vcf_effect_af.to_numpy(dtype=float), [0.875, 0.125])
        self.assertEqual(report.vcf_major_allele.tolist(), ["A", "A"])
        # Complementing the strand must not negate the biological effect.
        np.testing.assert_allclose(scores.PRS_GWAS, [0.5, 0.4, 0.4, 0.4])

    def test_palindromic_missing_frequency_is_not_guessed_from_major(self):
        self.assert_excluded([association(effect="A")],
                             [record(ref="A", alt="T", gt=("0/0", "0/0"))])

    def test_palindromic_frequency_near_half_is_unresolved(self):
        self.assert_excluded([association(effect="A", frequency=0.5)],
                             [record(ref="A", alt="T", gt=("0/1", "1/0"))])

    def test_palindromic_insufficient_samples_and_call_rate(self):
        row = association(effect="A", frequency=1)
        calls = record(ref="A", alt="T", gt=("0/0", "./."))
        self.assert_excluded([row], [calls], min_af_samples=20)
        self.assert_excluded([row], [calls], min_af_samples=1, min_af_call_rate=0.9)

    def test_nonpalindromic_frequency_disagreement_cannot_force_swap(self):
        report = self.assert_excluded([association(effect="A", frequency=0.9)],
                                     [record(ref="C", alt="A", gt=("0/0", "0/1"))])
        self.assertFalse(report.status.eq("accepted").any())

    def test_nonpalindromic_missing_frequency_still_uses_sequence(self):
        _, reference, report = self.score([association(effect="T")], [record()])
        self.assertEqual(reference.effect_allele.tolist(), ["A"])
        self.assertEqual(report.orientation.tolist(), ["complement"])

    def test_frequency_subset_changes_alignment_without_dropping_scored_people(self):
        samples = ("e1", "e2", "e3", "e4", "a1", "a2", "a3", "a4")
        # First ancestry has T frequency 0.125; pooled frequency is 0.5625.
        calls = ("0/1", "0/0", "0/0", "0/0", "1/1", "1/1", "1/1", "1/1")
        rows = [association(effect="A", frequency=0.125)]
        records = [record(ref="A", alt="T", gt=calls)]
        self.assert_excluded(rows, records, samples=samples)
        scores, reference, report = self.score(
            rows, records, samples=samples, frequency_samples=list(samples[:4]))
        self.assertEqual(len(scores), 8)
        self.assertEqual(reference.effect_allele.tolist(), ["T"])
        self.assertAlmostEqual(float(report.vcf_alt_af.iloc[0]), 0.125)
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0, 0, 0, 0.4, 0.4, 0.4, 0.4])

    def test_unknown_frequency_sample_rejected(self):
        with self.assertRaises(ValueError):
            self.score([association()], [record()], frequency_samples=["not_a_sample"])

    def test_partial_missing_gt_is_not_reference(self):
        scores, _, _ = self.score([association()], [record(gt=("0/.", "1/1"))])
        np.testing.assert_allclose(scores.PRS_GWAS, [0.4, 0.4])
        np.testing.assert_allclose(scores.PRS_GWAS_noimp, [0, 0.4])
        self.assertEqual(scores.n_imputed_variants.tolist(), [1, 0])

    def test_entirely_missing_locus_is_not_imputed(self):
        self.assert_excluded([association()], [record(gt=("./.", "0/."))])

    def test_invalid_genotype_index_rejected(self):
        with self.assertRaises(ValueError):
            self.score([association()], [record(gt=("0/2", "0/0"))])

    def test_multiallelic_symbolic_and_indel_records_excluded(self):
        for ref, alt in [("C", "A,G"), ("C", "<DEL>"), ("CA", "C")]:
            with self.subTest(ref=ref, alt=alt):
                self.assert_excluded([association()], [record(ref=ref, alt=alt)])

    def test_split_multiallelic_alt_candidate_can_be_scored(self):
        scores, reference, report = self.score([association()], [record(), record(alt="G")])
        self.assertEqual(reference.vcf_alt.tolist(), ['A'])
        self.assertEqual(report.mapping_status.tolist(), ['matched'])
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0])

    def test_identical_vcf_duplicate_counts_once(self):
        scores, reference, _ = self.score([association()], [record(), record()])
        self.assertEqual(len(reference), 1)
        self.assertTrue(scores.n_scored_variants.eq(1).all())
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0])

    def test_conflicting_vcf_duplicate_excluded(self):
        self.assert_excluded([association()], [record(), record(gt=("1/1", "0/0"))])

    def test_duplicate_associations_are_not_first_row_selected(self):
        self.assert_excluded([association(weight=0.2), association(weight=0.8)], [record()])

    def test_unknown_risk_allele_and_haplotype_excluded(self):
        for risk in ["rs100-?", "rs100-A; rs101-T", "rs100-AT"]:
            with self.subTest(risk=risk):
                self.assert_excluded([association(**{"STRONGEST SNP-RISK ALLELE": risk})],
                                     [record()])

    def test_rsid_mismatch_excluded_but_current_merged_id_matches(self):
        self.assert_excluded([association()], [record(snp_id="rs999")])
        _, reference, _ = self.score(
            [association(**{"SNP_ID_CURRENT": "999"})], [record(snp_id="rs999")])
        self.assertEqual(len(reference), 1)

    def test_or_below_one_and_beta_decrease_have_negative_weights(self):
        rows = [association(weight=0.8, **{"95% CI (TEXT)": "[0.7-0.9]"}),
                association(101, weight=0.3, **{"95% CI (TEXT)": "[0.2-0.4] unit decrease"})]
        scores, reference, _ = self.score(rows, [record(gt=("0/1", "0/0")),
                                               record(101, gt=("1/1", "0/0"))])
        np.testing.assert_allclose(reference.effect_weight, [np.log(0.8), -0.3])
        np.testing.assert_allclose(scores.PRS_GWAS, [np.log(0.8) - 0.6, 0])

    def test_explicit_signed_beta_and_or_validation(self):
        _, reference, _ = self.score(
            [association(weight=-0.3, **{"95% CI (TEXT)": "NR"})],
            [record()], effect_type="beta")
        np.testing.assert_allclose(reference.effect_weight, [-0.3])
        self.assert_excluded([association()], [record()], effect_type="or")
        self.assert_excluded([association(weight=0, **{"95% CI (TEXT)": "[0-0]"})],
                             [record()], effect_type="or")

    def test_auto_missing_effect_type_is_not_guessed(self):
        self.assert_excluded([association(**{"95% CI (TEXT)": "NR"})], [record()])

    def test_study_and_trait_selection(self):
        rows = [association(), association(101, **{"STUDY ACCESSION": "OTHER"})]
        scores, reference, _ = self.score(rows, [record(), record(101)])
        self.assertEqual(len(reference), 1)
        self.assertTrue(scores.study_accession.eq("GCST_TEST").all())
        rows[1]["STUDY ACCESSION"] = "GCST_TEST"
        rows[1]["DISEASE/TRAIT"] = "Survival"
        with self.assertRaises(ValueError):
            self.score(rows, [record(), record(101)])
        _, reference, _ = self.score(rows, [record(), record(101)], trait="Breast cancer")
        self.assertEqual(len(reference), 1)

    def test_pvalue_annotation_selects_analysis_instead_of_deduplicating(self):
        rows = [association(weight=0.2),
                association(weight=0.8, **{"P-VALUE (TEXT)": "(other analysis)"})]
        scores, reference, _ = self.score(rows, [record()], pvalue_text="(EA)")
        self.assertEqual(len(reference), 1)
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0])

    def test_partial_scores_are_explicit_and_coverage_is_retained(self):
        rows = [association(), association(101)]
        self.assert_excluded(rows, [record()])
        scores, reference, report = self.score(rows, [record()], allow_partial=True)
        self.assertTrue(scores.n_selected_associations.eq(2).all())
        self.assertTrue(scores.n_scored_variants.eq(1).all())
        self.assertFalse(scores.complete_model.any())
        self.assertEqual(len(reference), 1)
        self.assertEqual(len(report), 2)
        self.assert_excluded([association()], [], allow_partial=True)

    def test_filtered_and_no_gt_records_are_not_scored(self):
        self.assert_excluded([association()], [record(filt="LowQual")])
        self.assert_excluded([association()], [record(fmt="DS", gt=("1.0", "0.0"))])

    def test_grch37_and_duplicate_sample_ids_rejected(self):
        with self.assertRaises(ValueError):
            score_gwas_vcf(vcf([record()], build="GRCh37"), pd.DataFrame([association()]),
                           study="GCST_TEST", genome_build="GRCh38")
        with self.assertRaises(ValueError):
            self.score([association()], [record()], samples=("same", "same"))

    def test_gzip_input_and_dataframe_not_mutated(self):
        source = pd.DataFrame([association()])
        original = source.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cohort.vcf.bgz"
            path.write_bytes(gzip.compress(vcf([record()]).getvalue().encode()))
            scores, _, _ = score_gwas_vcf(path, source, study="GCST_TEST", genome_build="GRCh38")
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0])
        pd.testing.assert_frame_equal(source, original)

    def test_cli_success_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, variants, prefix = root / "catalog.tsv", root / "input.vcf", root / "results"
            pd.DataFrame([association()]).to_csv(source, sep="\t", index=False)
            variants.write_text(vcf([record()]).getvalue())
            args = ["--gwas", str(source), "--vcf", str(variants), "--study", "GCST_TEST",
                    "--genome-build", "GRCh38", "--out-prefix", str(prefix)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            score_path = Path(str(prefix) + ".scores.tsv")
            report_path = Path(str(prefix) + ".variants.tsv")
            ref_path = Path(str(prefix) + ".harmonized.tsv")
            self.assertTrue(all(p.exists() for p in [score_path, report_path, ref_path]))
            reference = pd.read_csv(ref_path, sep="\t", comment="#")
            self.assertEqual(reference.effect_allele.tolist(), ["A"])
            before = {p: p.read_bytes() for p in [score_path, report_path, ref_path]}
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(args)
            self.assertEqual(caught.exception.code, 2)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_cli_strict_failure_writes_audit_without_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, variants, prefix = root / "catalog.tsv", root / "input.vcf", root / "results"
            pd.DataFrame([association(), association(101)]).to_csv(source, sep="\t", index=False)
            variants.write_text(vcf([record()]).getvalue())
            args = ["--gwas", str(source), "--vcf", str(variants), "--study", "GCST_TEST",
                    "--genome-build", "GRCh38", "--out-prefix", str(prefix)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
            self.assertFalse(Path(str(prefix) + ".scores.tsv").exists())
            report = pd.read_csv(str(prefix) + ".variants.tsv", sep="\t")
            reference = pd.read_csv(str(prefix) + ".harmonized.tsv", sep="\t", comment="#")
            self.assertEqual(len(report), 2)
            self.assertEqual(len(reference), 1)


class LowestPSelectionTests(unittest.TestCase):
    score = GWASVCFTests.score
    assert_excluded = GWASVCFTests.assert_excluded

    def test_mapping_retains_repeated_studies_before_weight_selection(self):
        rows = [association(weight=0.2, **{'P-VALUE': '1e-5'}),
                association(weight=0.8, **{'P-VALUE': '1e-9'})]
        scores, reference, report = self.score(rows, [record()], duplicate_policy='lowest_p')
        self.assertTrue(report.mapping_status.eq('matched').all())
        self.assertTrue(report.coordinate_match.all())
        self.assertTrue(report.sequence_match.all())
        self.assertEqual(report.status.tolist(), ['duplicate_association_not_selected', 'accepted'])
        self.assertEqual(reference.effect_weight.tolist(), [0.8])
        self.assertEqual(scores.n_scored_variants.tolist(), [1, 1])
        self.assertTrue(scores.complete_model.all())
        np.testing.assert_allclose(scores.PRS_GWAS, [0.8, 0])

    def test_selection_uses_only_eligible_weights(self):
        rows = [association(weight=0.2, **{'P-VALUE': '1e-30', '95% CI (TEXT)': 'NR'}),
                association(weight=0.8, **{'P-VALUE': '1e-9'})]
        scores, reference, report = self.score(rows, [record()], duplicate_policy='lowest_p', allow_partial=True)
        self.assertTrue(report.mapping_status.eq('matched').all())
        self.assertEqual(report.status.tolist(), ['unknown_effect_type', 'accepted'])
        self.assertEqual(reference.effect_weight.tolist(), [0.8])
        np.testing.assert_allclose(scores.PRS_GWAS, [0.8, 0])

    def test_different_effect_alleles_do_not_double_count_one_variant(self):
        rows = [association(effect='A', weight=0.2, **{'P-VALUE': '1e-5'}),
                association(effect='C', weight=0.8, **{'P-VALUE': '1e-9'})]
        scores, reference, _ = self.score(rows, [record()], duplicate_policy='lowest_p')
        self.assertEqual(reference.effect_allele.tolist(), ['C'])
        np.testing.assert_allclose(scores.PRS_GWAS, [0.8, 1.6])

    def test_lowest_p_order_is_numeric_and_does_not_underflow(self):
        rows = [association(weight=0.2, **{'P-VALUE': '1e-400'}),
                association(weight=0.8, **{'P-VALUE': '1e-500'})]
        _, reference, _ = self.score(rows, [record()], duplicate_policy='lowest_p')
        self.assertEqual(reference.effect_weight.tolist(), [0.8])

    def test_tied_p_with_different_weights_remains_unresolved(self):
        rows = [association(weight=0.2, **{'P-VALUE': '1e-9'}),
                association(weight=0.8, **{'P-VALUE': '1e-9'})]
        report = self.assert_excluded(rows, [record()], duplicate_policy='lowest_p', allow_partial=True)
        self.assertTrue(report.status.eq('tied_pvalue_conflicting_weights').all())
        self.assertTrue(report.mapping_status.eq('matched').all())

    def test_identical_tied_estimates_contribute_once(self):
        rows = [association(weight=0.2, **{'P-VALUE': '1e-9'})] * 2
        scores, reference, _ = self.score(rows, [record()], duplicate_policy='lowest_p')
        self.assertEqual(len(reference), 1)
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0])

    def test_duplicate_weights_with_no_p_cannot_be_ranked(self):
        report = self.assert_excluded([association(weight=0.2), association(weight=0.8)],
                                     [record()], duplicate_policy='lowest_p')
        self.assertTrue(report.status.eq('missing_selection_pvalue').all())

    def test_unknown_weight_still_receives_allele_mapping(self):
        report = self.assert_excluded([association(**{'95% CI (TEXT)': 'NR'})], [record()])
        self.assertEqual(report.mapping_status.tolist(), ['matched'])
        self.assertEqual(report.weight_status.tolist(), ['unknown_effect_type'])

    def test_split_reference_effect_is_not_two_minus_one_split_alt(self):
        report = self.assert_excluded([association(effect='C')],
                                     [record(), record(alt='G')], duplicate_policy='lowest_p')
        self.assertEqual(report.mapping_status.tolist(), ['split_site_ref_effect_ambiguous'])
        self.assertTrue(report.sequence_match.all())

    def test_different_alt_candidates_without_af_remain_ambiguous(self):
        report = self.assert_excluded([association()], [record(), record(alt='T')])
        self.assertEqual(report.mapping_status.tolist(), ['ambiguous_vcf_candidates'])

    def test_frequency_can_select_between_two_alt_candidates(self):
        calls_a = ('0/1', '0/0', '0/0', '0/0')
        calls_t = ('0/1', '1/1', '1/1', '0/1')
        scores, reference, report = self.score(
            [association(frequency=0.125)], [record(gt=calls_a), record(alt='T', gt=calls_t)],
            samples=('a', 'b', 'c', 'd'))
        self.assertEqual(reference.effect_allele.tolist(), ['A'])
        self.assertEqual(report.n_sequence_candidates.tolist(), [2])
        np.testing.assert_allclose(scores.PRS_GWAS, [0.2, 0, 0, 0])

    def test_invalid_duplicate_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate_policy'):
            self.score([association()], [record()], duplicate_policy='first')

    def test_pooled_lowest_p_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            catalog, calls, prefix = path / 'gwas.tsv', path / 'samples.vcf', path / 'result'
            pd.DataFrame([
                association(weight=0.2, **{'P-VALUE': '1e-5'}),
                association(weight=0.8, **{'P-VALUE': '1e-9', 'STUDY ACCESSION': 'OTHER'}),
            ]).to_csv(catalog, sep='\t', index=False)
            calls.write_text(vcf([record()]).getvalue())
            with redirect_stdout(io.StringIO()):
                rc = main(['--gwas', str(catalog), '--vcf', str(calls), '--genome-build', 'GRCh38',
                           '--duplicate-policy', 'lowest_p', '--out-prefix', str(prefix)])
            self.assertEqual(rc, 0)
            ref = pd.read_csv(str(prefix) + '.harmonized.tsv', sep='\t', comment='#')
            self.assertEqual(ref['STUDY ACCESSION'].tolist(), ['OTHER'])
            self.assertEqual(ref.effect_weight.tolist(), [0.8])


class ReferenceGenomeTests(unittest.TestCase):
    score = GWASVCFTests.score
    assert_excluded = GWASVCFTests.assert_excluded

    def test_optional_reference_check_preserves_scores_and_dosage_orientation(self):
        rows = [association(effect="C"), association(101, effect="T")]
        records = [record(), record(101, gt=("1/1", "0/0"))]
        original, _, _ = self.score(rows, records)
        with patch('score_gwas_vcf_prs._hail_reference_bases',
                   return_value={("1", 100): "C", ("1", 101): "c"}) as lookup:
            scores, reference, report = self.score(rows, records, reference_source="hail")
        lookup.assert_called_once_with({("1", 100), ("1", 101)})
        np.testing.assert_allclose(scores.PRS_GWAS, original.PRS_GWAS)
        self.assertEqual(reference.effect_allele.tolist(), ["C", "A"])
        self.assertEqual(report.reference_base.tolist(), ["C", "C"])
        self.assertTrue(report.reference_check.eq("match").all())
        self.assertTrue(scores.reference_source.eq("hail").all())

    def test_reference_mismatch_is_not_repaired_by_swapping_ref_alt(self):
        with patch('score_gwas_vcf_prs._hail_reference_bases', return_value={("1", 100): "A"}):
            # Genome base equals ALT; reassigning GT indices would change the score.
            report = self.assert_excluded([association(frequency=0.9)], [record()], reference_source="hail")
        self.assertEqual(report.status.tolist(), ["reference_mismatch"])
        self.assertEqual(report.reference_check.tolist(), ["mismatch"])
        self.assertEqual(report.vcf_ref.tolist(), ["C"])
        self.assertEqual(report.vcf_alt.tolist(), ["A"])
        self.assertTrue(report.vcf_alt_af.isna().all())  # Reference validation precedes AF inference.

    def test_missing_or_ambiguous_reference_base_cannot_fall_back_silently(self):
        for base in [None, "N", "", "CA"]:
            with self.subTest(base=base), patch('score_gwas_vcf_prs._hail_reference_bases',
                                               return_value={("1", 100): base}):
                report = self.assert_excluded([association()], [record()], reference_source="hail")
                self.assertEqual(report.status.tolist(), ["reference_unavailable"])

    def test_reference_check_does_not_resolve_palindromic_effect_strand(self):
        with patch('score_gwas_vcf_prs._hail_reference_bases', return_value={("1", 100): "A"}):
            report = self.assert_excluded([association(effect="A")],
                                         [record(ref="A", alt="T")], reference_source="hail")
        self.assertEqual(report.reference_check.tolist(), ["match"])
        self.assertEqual(report.status.tolist(), ["palindromic_missing_catalog_af"])

    def test_vcf_default_does_not_query_reference(self):
        with patch('score_gwas_vcf_prs._hail_reference_bases', side_effect=AssertionError("unexpected lookup")):
            _, _, report = self.score([association()], [record()])
        self.assertEqual(report.reference_check.tolist(), ["not_requested"])

    def test_invalid_reference_source_fails_before_reading_vcf(self):
        with self.assertRaisesRegex(ValueError, "reference_source"):
            self.score([association()], [record()], reference_source="GRCh37")

    def test_reference_lookup_error_does_not_silently_use_vcf_only(self):
        with patch('score_gwas_vcf_prs._hail_reference_bases', side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                self.score([association()], [record()], reference_source="hail")

    def test_partial_reference_mismatch_is_visible_in_score_coverage(self):
        with patch('score_gwas_vcf_prs._hail_reference_bases',
                   return_value={("1", 100): "C", ("1", 101): "T"}):
            scores, _, report = self.score([association(), association(101)], [record(), record(101)],
                                          reference_source="hail", allow_partial=True)
        self.assertEqual(report.status.tolist(), ["accepted", "reference_mismatch"])
        self.assertTrue(scores.n_scored_variants.eq(1).all())
        self.assertFalse(scores.complete_model.any())

    def test_hail_batch_adapter_uses_one_based_positions_and_chrM(self):
        hl = MagicMock()
        rg = hl.get_reference.return_value
        rg.has_sequence.return_value = False
        rg.lengths = {"chr1": 1000, "chrM": 100}
        hl.Struct.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        ht = hl.Table.parallelize.return_value
        ht.annotate.return_value.collect.return_value = [
            SimpleNamespace(chrom="1", pos=1, reference_base="C"),
            SimpleNamespace(chrom="MT", pos=100, reference_base="A"),
        ]
        with patch.dict(sys.modules, {"hail": hl}):
            bases = _hail_reference_bases({("1", 1), ("MT", 100), ("MT", 101), ("Z", 1)})
        self.assertEqual(bases, {("1", 1): "C", ("MT", 100): "A"})
        hl.get_reference.assert_called_once_with("GRCh38")
        rg.add_sequence.assert_called_once_with(GRCH38_FASTA, GRCH38_FAI)
        queried = hl.Table.parallelize.call_args.args[0]
        self.assertEqual([(r.contig, r.pos) for r in queried], [("chr1", 1), ("chrM", 100)])
        hl.get_sequence.assert_called_once_with(ht.contig, ht.pos, reference_genome="GRCh38")

    def test_hail_adapter_reuses_registered_reference_sequence(self):
        hl = MagicMock()
        rg = hl.get_reference.return_value
        rg.has_sequence.return_value = True
        rg.lengths = {"chr1": 1000}
        hl.Table.parallelize.return_value.annotate.return_value.collect.return_value = []
        with patch.dict(sys.modules, {"hail": hl}):
            _hail_reference_bases({("1", 100)})
        rg.add_sequence.assert_not_called()
        hl.init.assert_not_called()

    def test_cli_hail_option_records_reference_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            catalog = path / "gwas.tsv"
            calls = path / "samples.vcf"
            prefix = path / "result"
            pd.DataFrame([association()]).to_csv(catalog, sep="\t", index=False)
            calls.write_text(vcf([record()]).getvalue())
            args = ["--gwas", str(catalog), "--vcf", str(calls), "--study", "GCST_TEST",
                    "--genome-build", "GRCh38", "--reference-source", "hail", "--out-prefix", str(prefix)]
            with patch('score_gwas_vcf_prs._hail_reference_bases', return_value={("1", 100): "C"}), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            text = Path(str(prefix) + ".harmonized.tsv").read_text()
            self.assertIn("#reference_source=hail\n", text)
            report = pd.read_csv(str(prefix) + ".variants.tsv", sep="\t")
            self.assertEqual(report.reference_check.tolist(), ["match"])


if __name__ == "__main__":
    unittest.main()
