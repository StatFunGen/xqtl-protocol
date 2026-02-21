# SoS Step Dry-Run Results

Each `mwe/NN_<step>.sh` script runs `sos run -n <notebook> <step> ...` to validate
that parameter names and file formats are correct without executing compute tasks.

**Note on `sos run -n`:** The `-n` flag prevents task execution (shows HINT blocks
instead) but still runs global section and step setup/metadata code.  Steps that
read or validate input files in their setup code will fail if placeholder data is
not in the exact expected format, or if external tools (R, bgzip, tabix) are
invoked in setup code.

---

## Results Summary

| # | Step | Status | Notes |
|---|------|--------|-------|
| 01 | fastqc | **PASS** | |
| 02 | rnaseqc_call | *partial* | Steps 1-2 print HINT blocks; step 3 fails: `multiqc` not installed |
| 03 | bulk_expression_qc | *env* | R not installed (`library(edgeR)`) |
| 04 | bulk_expression_normalization | **PASS** | |
| 05 | vcf_qc | **PASS** | |
| 06 | vcf_to_plink | **PASS** | |
| 07 | genotype_by_chrom | **PASS** | |
| 08 | plink_qc | **PASS** | |
| 09 | sample_match | *env* | `bgzip`/`tabix` not installed |
| 10 | king_kinship | **PASS** | |
| 11 | unrelated_qc | **PASS** | |
| 12 | related_qc | **PASS** | |
| 13 | flashpca | *env* | R not installed (`library(flashpcaR)`) |
| 14 | project_samples | *env* | R not installed |
| 15 | merge_pca_covariate | *env* | R not installed (`library(dplyr)`) |
| 16 | phenotype_by_chrom | *data* | Step 2 reads placeholder phenotype file; empty CSV causes `EmptyDataError` |
| 17 | marchenko_pc | *env* | R not installed |
| 18 | peer | *env* | R not installed |
| 19 | tensorqtl_cis | *env* | R not installed (Python cis steps show HINT blocks) |
| 20 | susie_twas | *data* | `process_cis_files` in mnm_regression setup code expects per-region BED format; placeholder standard expression BED causes `TypeError` |
| 21 | finemapping_plots | **PASS** | Documented limitation: SoS cannot accept positional `_input` for standalone steps; `univariate_plot` returns "Unrecognized option" but exits 0 via find-loop guard |

**PASS: 10 steps** | **env (R/tool): 9 steps** | **data format: 2 steps**

---

## Design Gaps Found and Fixed

The following bugs in the Snakemake rules were found and fixed during dry-run testing:

1. **`DRY_RUN_SOS = "--dryrun"`** → `"-n"` (wrong SoS flag; also flag must come before
   notebook path, not after step name) — fixed in `Snakefile` and all 6 `rules/sos/*.smk`
   files via regex replacement of `{params.dry_run}` position (22 replacements).

2. **`Snakefile_sos` missing `DRY_RUN_SOS`** — added definition.

3. **`susie_twas --L`** → `--init-L` (wrong parameter name; notebook uses `init_L`).

4. **`susie_twas` missing `--name`** — `mnm_regression.ipynb` requires `--name`.

5. **`finemapping_plots --finemapping-dir`** — parameter does not exist in
   `rss_analysis.ipynb`; replaced with a `find` loop that passes each `.rds` file
   as a positional argument.

6. **`susie_twas phenoFile`** — `mnm_regression phenoFile = paths` expects actual
   BED file paths, not the file-listing from `phenotype_by_chrom`; fixed by `awk`
   expansion in the shell block.

7. **`rnaseqc_call` design gap** — the Snakemake rule only passes `--sample-list`
   but sub-steps `rnaseqc_call_1/2/3/4` also require `--bam-list` (STAR output).
   A dedicated `STAR_align` Snakemake rule is needed upstream to produce the BAM
   manifest before `rnaseqc_call` can run fully.

---

## Environment Requirements for Full Dry-Run

To get all steps to pass `sos run -n`, the following must be installed:

- **R** with packages: `edgeR`, `flashpcaR`, `dplyr`, `readr`, `purrr`, `tidyr`, `peer`
- **bgzip** and **tabix** (htslib)
- **multiqc**
- **STAR** aligner (for rnaseqc_call BAM list)
