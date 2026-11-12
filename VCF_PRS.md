# Individual PRS from a VCF

`score_vcf_prs.py` wraps `03_common_variants_prs.py`. It reuses
`prepare_definition`, `resolve_af` and `score_prs`; the numbered module is unchanged.
Only Python, NumPy and pandas are required. No PLINK conversion is needed.

## Command line

From the repository directory, run:

```bash
python score_vcf_prs.py \
  --vcf /path/to/merged.vcf.bgz \
  --genome-build GRCh38 \
  --out-prefix /path/to/private_outputs/breast_prs
```

The default definition is the public [PGS000004 harmonized GRCh38 file](https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000004/ScoringFiles/Harmonized/PGS000004_hmPOS_GRCh38.txt.gz).
Use `--definition /path/to/PGS000004_hmPOS_GRCh38.txt.gz` for an offline copy.
The original `chr_name`/`chr_position` columns identify the two CHEK2 tags;
the harmonized `hm_chr`/`hm_pos` columns match the VCF.

Two files are written, with existing files protected from replacement:

- `breast_prs.scores.tsv`: one row per VCF sample, in original sample order.
- `breast_prs.variants.tsv`: all 313 model loci and their matching/QC status.

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

- `PRS313`: weighted effect-allele counts, with missing individual calls replaced
  by that locus's mean among the samples in the input VCF.
- `PRS311`: the same calculation excluding the two CHEK2 tags defined in stage 03.
- `PRS313_noimp` and `PRS311_noimp`: missing calls contribute zero.
- `PRS311_direct_only`: excludes both CHEK2 tags and loci resolved using AF.

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

By default, all 313 loci must be scorable. On incomplete matching the CLI writes
only the variant report and exits with code 2. In Python, catch
`IncompleteScoreError` and inspect its `.report` attribute.
`--allow-partial` (or `allow_partial=True`) explicitly allows scoring only accepted
loci, without rescaling. Check `complete_model`, `n_scored_loci`,
`n_scored_non_chek2_tag_loci`, `n_missing_model_loci`, `n_called_loci` and
`n_imputed_loci`; incomplete scores are not equivalent to the full PRS313/311.
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
