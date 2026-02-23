# SoS Step Dry-Run Results

Each `mwe/NN_<step>.sh` script runs `sos run notebook.ipynb step ... -n` to validate
that parameter names and file formats are correct without executing compute tasks.

**Note on `sos run ... -n`:** The `-n` flag prevents `task:` block execution (shows
HINT blocks instead) but still runs global section and step setup/metadata code that
is outside task blocks.  Steps that read or validate input files in their setup code
require properly-formatted placeholder data.  Additionally, SoS's `R:` action
validates that `Rscript` is available on the host even in dry-run mode, so R must be
installed for R-based steps to show HINT blocks rather than fail.

---

## Results Summary

| # | Step | Status | Notes |
|---|------|--------|-------|
| 01 | fastqc | **PASS** | |
| 02 | rnaseqc_call | *design* | Steps 1-2 PASS; step 3 fails: `report()` action runs outside `task:` block and requires real STAR BAM input (see design gap 9) |
| 03 | bulk_expression_qc | **PASS** | |
| 04 | bulk_expression_normalization | **PASS** | |
| 05 | vcf_qc | **PASS** | |
| 06 | vcf_to_plink | **PASS** | |
| 07 | genotype_by_chrom | **PASS** | |
| 08 | plink_qc | **PASS** | |
| 09 | sample_match | **PASS** | |
| 10 | king_kinship | **PASS** | |
| 11 | unrelated_qc | **PASS** | |
| 12 | related_qc | **PASS** | |
| 13 | flashpca | **PASS** | |
| 14 | project_samples | **PASS** | |
| 15 | merge_pca_covariate | **PASS** | |
| 16 | phenotype_by_chrom | **PASS** | |
| 17 | marchenko_pc | **PASS** | |
| 18 | peer | **PASS** | |
| 19 | tensorqtl_cis | **PASS** | |
| 20 | susie_twas | **PASS** | |
| 21 | finemapping_plots | **PASS** | |

**PASS: 20 steps** | **design gap: 1 step (02)**

---

## Design Gaps Found and Fixed

The following bugs in the Snakemake rules were found and fixed during dry-run testing:

1. **`DRY_RUN_SOS = "--dryrun"`** → `"-n"` placed after all parameters — wrong flag
   and wrong position; fixed in `Snakefile`, `Snakefile_sos`, and all 6
   `rules/sos/*.smk` files (22 replacements).

2. **`Snakefile_sos` missing `DRY_RUN_SOS`** — added definition.

3. **`susie_twas --L`** → `--init-L` (wrong parameter name; notebook uses `init_L`).

4. **`susie_twas` missing `--name`** — `mnm_regression.ipynb` requires `--name`.

5. **`finemapping_plots --finemapping-dir`** — parameter does not exist in
   `rss_analysis.ipynb`; replaced with a `find` loop that passes each `.rds` file
   as a positional argument.

6. **`susie_twas phenoFile` format** — `mnm_regression.get_analysis_regions` calls
   `process_cis_files()` which treats column 4 of each phenotype BED as a **file
   path** to per-region data, not an expression value.  Passing standard expression
   BEDs (col4 = float) causes `TypeError`.  The smk rule needs a data-prep step that
   produces mnm-format BEDs (col4 = path).  MWE uses correctly-formatted placeholders.

7. **`susie_twas` missing `--cis-window`** — default is `-1` which raises `ValueError`;
   added `--cis-window {params.cis_window}` from `config["association"]["cis_window"]`.

8. **`susie_twas --covFile`** — `mnm_regression` requires one covFile per phenotype
   file (`len(covFile) == len(phenoFile)`); fixed by repeating hidden_factors N_PHENO
   times in the shell block.

9. **`rnaseqc_call` design gap** — sub-steps require `--bam-list` (STAR BAM output);
   a dedicated `STAR_align` Snakemake rule is needed upstream.

---

## Environment Requirements for Full Dry-Run

The following tools must be installed to reproduce the results above (20/21 PASS).
All three are now present in the test environment:

- **R** (≥ 4.3) — SoS's `R:` action validates `Rscript` availability before printing
  HINT, even with `-n`; needed for steps 03, 13, 14, 15, 17, 18, 19, 20
- **bgzip** and **tabix** (htslib) — step 09 declares them via `input: executable()`
  which SoS resolves before `-n` takes effect
- **multiqc** — step 02 steps 1-2; step 3 still fails due to design gap (see above)

The one remaining failure (step 02 step 3) is a pipeline design gap, not an
environment issue: `RNA_calling.ipynb` calls `report()` outside a `task:` block,
so it executes even under `-n` and requires real STAR BAM input files that a
dedicated upstream `STAR_align` rule must produce.
