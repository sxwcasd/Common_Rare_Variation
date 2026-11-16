# Individual PRS from a VCF

`score_vcf_prs.py` wraps `03_common_variants_prs.py`. It reuses
`prepare_definition`, `resolve_af` and `score_prs`, including model-specific column names.
Only Python, NumPy and pandas are required. No PLINK conversion is needed.

## Command line

From the repository directory, run:

```bash
python score_vcf_prs.py \
  --vcf /path/to/merged.vcf.bgz \
  --genome-build GRCh38 \
  --out-prefix /path/to/private_outputs/breast_prs
```

For another harmonized model, supply its definition:

```bash
python score_vcf_prs.py \
  --vcf /path/to/merged.vcf.bgz \
  --genome-build GRCh38 \
  --definition /path/to/PGS000015_hmPOS_GRCh38.txt.gz \
  --out-prefix /path/to/private_outputs/PGS000015
```
The default definition is the public [PGS000004 harmonized GRCh38 file](https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000004/ScoringFiles/Harmonized/PGS000004_hmPOS_GRCh38.txt.gz).
Use `--definition /path/to/PGS000004_hmPOS_GRCh38.txt.gz` for an offline copy.
The original `chr_name`/`chr_position` columns identify the two CHEK2 tags;
the harmonized `hm_chr`/`hm_pos` columns match the VCF.

Two files are written, with existing files protected from replacement:

- `breast_prs.scores.tsv`: one row per VCF sample, in original sample order.
- `breast_prs.variants.tsv`: all loci in the supplied definition and their matching/QC status.

Choose an output location inside your authorized analysis environment, outside
this source repository. No participant scores or identifiers are printed.

## Notebook

```python
import sys
sys.path.insert(0, "/home/jupyter/repos/Common_Rare_Variation")
from score_vcf_prs import score_vcf

individual_prs_df, prs_variant_report = score_vcf(
    "/path/to/merged.vcf.bgz",
    genome_build="GRCh38",
)
```

An already-open **text** stream is also accepted and remains open after scoring.
For cloud input, use a stream that decompresses the VCF before passing it to
`score_vcf`. For an uncompressed GCS VCF in an initialized Hail session:

```python
with hl.hadoop_open(vcf_uri, "r") as vcf_stream:
    individual_prs_df, prs_variant_report = score_vcf(
        vcf_stream, genome_build="GRCh38"
    )
```

Alternatively, stream a compressed GCS VCF to the CLI without saving a local copy:

```bash
set -o pipefail
gsutil -u YOUR_BILLING_PROJECT cat gs://YOUR_BUCKET/merged.vcf.bgz \
  | gzip -dc \
  | python score_vcf_prs.py --vcf - --genome-build GRCh38 \
      --out-prefix /path/to/private_outputs/breast_prs
```

## Score definitions and matching

The main score name uses the number of rows in the **full model definition**,
before matching/filtering. It does not change to `PRS2` when only two loci match.
The default 313-locus model with two CHEK2 tags retains `PRS313`/`PRS311` names
for compatibility with the numbered analysis modules. Other models name the
tag-excluded score explicitly rather than calling it `PRS311`. The following
table describes the default `chek2_mode="both"` output.

| Calculation | PGS000004 (313 model loci) | PGS000015 (5218 model loci) |
| --- | --- | --- |
| Weighted score, cohort mean imputation | `PRS313` | `PRS5218` |
| Exclude the stage-03 CHEK2 tag sites | `PRS311` | `PRS5218_no_CHEK2_tags` |
| Missing individual calls contribute zero | `PRS313_noimp` | `PRS5218_noimp` |
| Exclude tags; missing calls contribute zero | `PRS311_noimp` | `PRS5218_no_CHEK2_tags_noimp` |
| Exclude tags and AF-resolved matches; mean imputation | `PRS311_direct_only` | `PRS5218_no_CHEK2_tags_direct_only` |

The tag mask still uses the two original-coordinate sites defined in stage 03;
it does not remove every variant in the CHEK2 gene. A supplied model may contain
zero, one, or both sites. `n_model_chek2_tag_loci` records how many it contains.
If no such sites are present, the tag-excluded score equals the main score.

To retain CHEK2 tags in **every returned score**, set `chek2_mode="include"`
in either `score_prs` or `score_vcf`, or pass `--chek2-mode include` to the CLI:

```bash
python score_vcf_prs.py \
  --vcf /path/to/merged.vcf.bgz \
  --genome-build GRCh38 \
  --definition /path/to/PGS000004_hmPOS_GRCh38.txt.gz \
  --chek2-mode include \
  --out-prefix /path/to/private_outputs/breast_prs_with_chek2
```

```python
scores, report = score_vcf(
    "/path/to/merged.vcf.bgz",
    definition="/path/to/PGS000004_hmPOS_GRCh38.txt.gz",
    genome_build="GRCh38",
    chek2_mode="include",
)
```

This mode outputs `PRS<N>`, `PRS<N>_noimp`, and `PRS<N>_direct_only` (the last
requires `direct_match` for direct stage-03 calls). It uses original weights
for CHEK2-tagged loci and omits `PRS311`/`_no_CHEK2_tags` columns. Direct-only
still excludes AF-resolved matches. The main `PRS<N>` and `PRS<N>_noimp` already
included the tags in the default mode; their values do not change. The variant
report and tag-count columns continue to describe which loci are CHEK2 tags.
Matching, missing-call imputation, and completeness requirements are unchanged.

The default `chek2_mode="both"` / `--chek2-mode both` preserves existing full
and tag-excluded scores for compatibility.

`pgs_id` records `#pgs_id` from the scoring file (or `definition.attrs['pgs_id']`
for a DataFrame), and `n_model_loci` records the full definition size. These
identify the intended model; `n_scored_loci` and `complete_model` describe actual
coverage. With no supplied PGS metadata, `pgs_id` is missing rather than guessed.

Direct stage-03 callers scoring another model must supply its full counts:

```python
definition = prs_method.prepare_definition(definition)
scores = prs_method.score_prs(
    effect_dosage, weights, tag_mask,
    n_model_loci=len(definition),
    n_model_chek2_tag_loci=int(definition.is_chek2_tagsnp.sum()),
    chek2_mode="include",  # Omit this argument to return both score versions.
)
```

Already-written TSV files are not changed automatically. For an existing
5218-model result produced by the old script, this renames columns in memory
without changing any scores:

```python
prs_5218_results = prs_5218_results.rename(columns={
    'PRS313': 'PRS5218',
    'PRS311': 'PRS5218_no_CHEK2_tags',
    'PRS313_noimp': 'PRS5218_noimp',
    'PRS311_noimp': 'PRS5218_no_CHEK2_tags_noimp',
    'PRS311_direct_only': 'PRS5218_no_CHEK2_tags_direct_only',
})
```

These are **raw weighted scores**, not ancestry-standardized z-scores or absolute
cancer risks. Supplying a different scoring cohort can change mean-imputed scores.
For a single-sample VCF, a missing call means the entire locus is missing.

Input must contain diploid global `GT` calls. Phased and unphased calls are
supported. `DS` and VDS-local `LGT` are not used. For a multiallelic record, the
effect allele is located using the original ALT index; a genotype containing
an allele outside the model's allele pair is treated as missing. Only site
`FILTER=PASS` or `.` records are used. No additional GQ/DP/FT thresholds are applied.

Matching requires exact forward allele pairs at the same GRCh38 position, with
the VCF REF belonging to the model pair. The wrapper does not perform liftover,
strand complementation, reference validation or indel normalization. Equivalent
indels represented at different positions are reported as unmatched and require
reference-based harmonization before scoring. Missing VCF sites are never
assumed to be homozygous reference. Use a dense VCF, not sparse VDS variant data.

By default, all loci in the supplied model must be scorable. On incomplete matching the CLI writes
only the variant report and exits with code 2. In Python, catch
`IncompleteScoreError` and inspect its `.report` attribute.
`--allow-partial` (or `allow_partial=True`) explicitly allows scoring only accepted
loci, without rescaling. Check `complete_model`, `n_scored_loci`,
`n_scored_non_chek2_tag_loci`, `n_missing_model_loci`, `n_called_loci` and
`n_imputed_loci`; incomplete scores are not equivalent to the full intended model.
This wrapper does not enforce the paper's cohort-specific 304/302 matched loci.

Identical duplicate records are counted once. Conflicting duplicates are
ambiguous. Multiple allele representations are rejected unless
`--resolve-by-af` / `resolve_by_af=True` is explicitly selected and stage 03's
AF gates accept one. Its published reference AF is European; this option is not
appropriate to enable automatically for a mixed-ancestry or small cohort.

The reader streams the VCF once and retains candidate dosages only at PRS loci.
It does not use a tabix index. For whole-genome VCFs, a pre-extracted PRS-region
VCF avoids a full genome scan. Keep all samples you want to use for mean imputation.

Run synthetic verification with:

```bash
python -m unittest discover -s tests -p 'test_score_vcf_prs.py'
```
