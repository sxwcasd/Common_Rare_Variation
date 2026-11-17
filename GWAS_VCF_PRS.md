# GWAS Catalog associations → VCF alleles → individual scores

`score_gwas_vcf_prs.py` is a separate script requiring numpy and pandas. It reads
the Catalog association TSV used in this project and a merged GRCh38 VCF, derives
the other allele from the VCF, and computes an exploratory weighted score for
each VCF sample. Existing PRS scripts and notebooks are not changed.
Optional reference-genome validation uses Hail in your All of Us runtime.

## Notebook example

```python
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/home/jupyter/repos/Common_Rare_Variation")  # your checkout
from score_gwas_vcf_prs import score_gwas_vcf, IncompleteGWASScoreError

# Set these to the files in your runtime.
gwas_path = Path("/path/to/gwas-association-downloaded_2026-11-16-MONDO_0007254-withChildTraits.tsv")
vcf_path = Path("/path/to/merged_samples.vcf.bgz")

# Optional: actual VCF sample IDs of ancestry-matched controls for AF estimation.
# Everyone in the VCF is still scored. None uses all VCF samples.
frequency_ids = None
# frequency_ids = ancestry_df.loc[
#     ancestry_df["ancestry"].eq("EUR") & ancestry_df["is_control"], "sample_id"
# ].astype(str).tolist()

try:
    scores, harmonized_reference, audit = score_gwas_vcf(
        vcf_path,
        gwas_path,
        study="GCST90090980",  # example study in the supplied file; choose your analysis
        trait="Breast cancer",
        genome_build="GRCh38",
        effect_type="or",     # this example uses ORs; verify your selected analysis
        frequency_samples=frequency_ids,
    )
except IncompleteGWASScoreError as exc:
    audit, harmonized_reference = exc.report, exc.reference
    print(exc)

print(audit["status"].value_counts())
```

The initial strict run gives an audit even if some associations cannot be used.
After reviewing exclusions, repeat with `allow_partial=True` to calculate a
reduced score. If no association is usable, it still fails. A score created from
a subset is not equivalent to the original set, and no rescaling is performed.

For a **plain-text** cloud VCF in a Hail notebook, pass an open text stream:

```python
with hl.hadoop_open(vcf_uri, "r") as vcf_stream:
    scores, harmonized_reference, audit = score_gwas_vcf(
        vcf_stream, gwas_path, study="GCST90090980", trait="Breast cancer",
        genome_build="GRCh38", effect_type="or",
        frequency_samples=frequency_ids, allow_partial=True,
    )
```

Local `.vcf`, `.vcf.gz`, and `.vcf.bgz` are detected automatically. Open streams
must supply decompressed text. The VCF is scanned once, keeping genotype arrays
only for target loci. This is not an indexed query and a whole-genome VCF can
take time to scan.

## Verify REF against GRCh38 in All of Us

With your existing `hl.init(...)` already configured, enable the new option:

```python
scores, harmonized_reference, audit = score_gwas_vcf(
    vcf_path, gwas_path,
    study="GCST90090980", trait="Breast cancer",
    genome_build="GRCh38", effect_type="or",
    reference_source="hail",   # Verify REF before allele/frequency inference.
    frequency_samples=frequency_ids,
    allow_partial=True,         # Only after reviewing exclusions, as above.
)
```

The VCF is still required. This mode uses Hail's registered GRCh38 sequence to
look up the reference base at each target, then checks that it equals VCF REF.
ALT comes from the supplied VCF. A FASTA contains the reference sequence; it
cannot identify the alternate allele or independently resolve an A/T or C/G
effect allele's strand. Existing allele and frequency checks still apply.

If GRCh38 has no attached sequence, the script registers the public files
listed in the [Hail reference-genome documentation](https://hail.is/docs/0.2/genetics/hail.genetics.ReferenceGenome.html#hail.genetics.ReferenceGenome.add_sequence):

```text
gs://hail-common/references/Homo_sapiens_assembly38.fasta.gz
gs://hail-common/references/Homo_sapiens_assembly38.fasta.fai
```

It reuses an already registered sequence, so you can attach your workspace's
GRCh38 FASTA/index with `hl.get_reference("GRCh38").add_sequence(...)` beforehand.
The lookup is batched over target coordinates using 1-based positions. It does
not query participant genotypes from an external service. It uses the existing
Hail backend and requester-pays configuration; it does not restart Hail.
For the Python API, initialize Hail with your workspace settings before calling
the scorer. In a fresh CLI process, Hail may initialize with its default
configuration; configure the Hail backend/requester-pays project first if needed.

New audit columns are `reference_source`, `reference_base`, and
`reference_check`. Missing/non-ACGT reference bases give `reference_unavailable`;
VCF REF disagreement gives `reference_mismatch`. These are excluded, without
silently swapping REF/ALT or altering GT indices. A reference-service error
stops the run rather than silently disabling the requested check.

The default `reference_source="vcf"` keeps the original behavior without a FASTA
lookup. The CLI equivalent of the new option is `--reference-source hail`.

## Command line

Run from the repository directory, using paths in your own environment:

```bash
python score_gwas_vcf_prs.py \
  --gwas /path/to/gwas-associations.tsv \
  --vcf /path/to/merged_samples.vcf.bgz \
  --trait "Breast cancer" \
  --genome-build GRCh38 \
  --effect-type or \
  --frequency-samples /path/to/ancestry_matched_control_ids.txt \
  --out-prefix /path/to/results/gwas_breast_cancer
```

The optional frequency file has one exact VCF sample ID per line, without a
header. Omit the flag to use everyone for AF calculation. The script checks that
all requested IDs exist. Use a new output prefix for each run; outputs are not
overwritten. A reduced run additionally requires `--allow-partial`.

Outputs:

- `.variants.tsv`: every selected association, original fields, alignment result,
  frequency comparison, and exclusion reason.
- `.harmonized.tsv`: accepted rows with `hm_chr`, `hm_pos`, `effect_allele`,
  `other_allele`, and `effect_weight`, plus provenance and QC. Read with
  `pd.read_csv(path, sep="\t", comment="#")`.
- `.scores.tsv`: `person_id`, `PRS_GWAS`, `PRS_GWAS_noimp`, selected/scored counts,
  per-sample called/imputed counts, study accession, and `complete_model`.

Strict incomplete runs exit with code 2 and write the audit/reference, without
a scores file. `complete_model` means every selected Catalog row was scorable;
it does not mean every sample had every genotype, or that this is a validated PRS.
The harmonized file is the output of this new workflow; do not pass it directly
to the older wrapper, which also expects original PGS coordinates for CHEK2 tags.
This script includes all accepted variants without any CHEK2-specific removal.

## Allele and frequency decisions

1. Match Catalog chromosome/position to the VCF. Both must be GRCh38; there is no
   liftover. Check original/current rsIDs when the VCF supplies rsIDs. VCF `ID=.`
   is supported with coordinate matching. If `reference_source="hail"`, verify
   VCF REF against GRCh38 sequence before proceeding to allele inference.
2. Accept explicit single-base risk alleles and biallelic SNVs. The reported
   effect allele may match VCF REF or ALT, directly or by strand complement.
   The other allele is the remaining VCF allele, not necessarily REF.
3. Compute REF/ALT allele frequencies from fully called diploid GTs in the AF
   sample subset. Ignore missing/partial calls in the denominator. Do not use
   INFO/AF, which may describe a different cohort. A frequency tie has no major
   allele. Report `vcf_major_allele`, `vcf_ref_af`, `vcf_alt_af`, and AF call counts.
4. Compare each possible effect-allele frequency to `RISK ALLELE FREQUENCY`.
   `NR` or invalid values stay missing. Non-palindromic SNPs normally have one
   sequence-compatible orientation. When informative frequencies disagree,
   exclude it; never force the effect allele to be the major allele.
5. Palindromic A/T and C/G SNPs have two possible orientations. Choose only a
   sufficiently close, distinctly better frequency match. Missing reference AF,
   insufficient calls, or frequencies near 0.5 leave these unresolved.
6. Complementing the strand keeps the same biological effect and weight. It does
   not negate beta or invert an OR. Effect dosage is ALT dosage when effect=ALT,
   and `2 - ALT dosage` when effect=REF.

Defaults are configurable QC heuristics: maximum EAF difference 0.10, minimum
gap between the two candidate differences 0.10, at least 20 called AF samples,
AF call rate at least 0.90, and palindromic reference/cohort MAF at most 0.42.
The last threshold has precedent in TwoSampleMR; these choices are not a
guarantee of correct alignment. Frequency comparisons need a comparable
population, ideally ancestry-matched controls. With missing or inadequate AF,
unique non-palindromic sequence matches can still be retained and are marked
`missing_catalog_af` or `insufficient_calls` in `frequency_check`.

`major_consistent` compares whether the effect allele is common (>0.5) in both
datasets. It is diagnostic, not a standalone orientation rule. The script
excludes multiallelic sites (including conflicting split records at one
position), indels/symbolic alleles, unknown risk alleles, rsID conflicts,
non-PASS filtered sites, and unresolved duplicate associations at one locus.
Identical repeated VCF records are counted once.

## Selecting and weighting associations

Choose one `STUDY ACCESSION`; use `trait`/`--trait` and
`pvalue_text`/`--pvalue-text` to select a specific outcome/analysis if needed.
`--pvalue-text ''` selects missing annotations. Repeated loci are excluded
instead of arbitrarily taking the first reported weight. The file contains
survival, drug-response, subtype, and susceptibility associations, which should
not all be pooled into one susceptibility score.

- `effect_type="or"`: log-transform positive per-allele ORs, including OR<1.
  Rows annotated as beta increase/decrease are rejected in this mode.
- `effect_type="beta"`: use the signed beta; Catalog `decrease` makes the weight
  negative. Contradictory negative/increase annotations are rejected.
- `effect_type="auto"`: beta is recognized by increase/decrease text, and OR by
  a plain numeric CI. Missing or unclear CI text is excluded as unknown rather
  than guessed. Verify the original statistical model and units: text parsing
  cannot establish that all estimates are comparable additive log-odds effects.

GTs must be diploid (phased or unphased). Haploid/polyploid and partial calls are
missing. DS-only target records are excluded. Missing individual genotypes use
the mean dosage across all scored samples for that SNP; `PRS_GWAS_noimp` instead
gives missing calls zero contribution. Entirely missing loci cannot be imputed.

This script addresses allele matching and weighted summation. LD selection,
cohort overlap, ancestry transferability, and independent validation remain
analysis choices. Simply summing Catalog associations produces an exploratory
score, not a reproduction of BC313 or another published PRS.

## References and checks

- [GWAS Catalog field definitions](https://www.ebi.ac.uk/gwas/docs/fileheaders/)
- [Catalog allele orientation and effect-size FAQ](https://www.ebi.ac.uk/gwas/docs/faq/)
- [TwoSampleMR allele-frequency harmonization](https://mrcieu.github.io/TwoSampleMR/articles/harmonise.html)
- [Hail sequence lookup](https://hail.is/docs/0.2/functions/genetics.html#hail.expr.functions.get_sequence)

Synthetic regression checks (no participant data):

```bash
python -m unittest discover -s tests -p 'test_score_gwas_vcf_prs.py' -v
```
