# pecotmr API migration — survey of legacy callers in xqtl-protocol

MWE inspected: `/Users/danielnachun/Downloads/fungen_xqtl/xqtl-protocol`.
Out of scope for this round: the four new wrappers under `code/SoS/pecotmr_integration/` (`qtl_dataset`, `fine_mapping`, `twas_weights`, `colocboost`).

## Notebooks examined

User-specified subset:

| Notebook | SoS steps with pecotmr work | Migration shape |
|---|---|---|
| `code/SoS/association_scan/quantile_models/qr_and_twas.ipynb` | `[quantile_qtl_twas_weight]` — inline R | new wrapper |
| `code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb` | `[susie_twas]` only (per user directive) — invokes `code/script/mnm_analysis/mnm_methods/susie_twas.R` | replace step + script |
| `code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb` | `[colocboost]` — inline R, calls `library(colocboost)` + `library(pecotmr)` (plus the deprecated multi-task loader pattern via the SoS `[get_analysis_regions]` step) | replace with new colocboost.ipynb pattern |
| `code/SoS/mnm_analysis/mnm_methods/rss_analysis.ipynb` | `[univariate_rss]` — invokes `code/script/pecotmr_integration/univariate_rss.R` | replace script + step |
| `code/SoS/pecotmr_integration/twas_ctwas.ipynb` | `[twas]`, `[ctwas_1]`, `[ctwas_2]`, `[ctwas_3]`, `[quantile_twas]` — all inline R | refactor; needs design |
| `code/SoS/pecotmr_integration/SuSiE_enloc.ipynb` | `[xqtl_gwas_enrichment]`, `[susie_coloc]` — both inline R | refactor; needs design |
| `code/SoS/pecotmr_integration/intact.ipynb` | `[intact]` — inline R (uses `fastenloc`, `run_intact`, etc.) | might be downstream-only (no pecotmr fitting) — needs check |

## Deprecated pecotmr symbols referenced per step

(Searched both snake_case and camelCase forms; both are in pecotmr's `R/deprecated.R`.)

- `qr_and_twas.ipynb [quantile_qtl_twas_weight]`: **`load_regional_association_data`** + quantile-regression-specific helpers.
- `mnm_regression.ipynb [susie_twas]` → `susie_twas.R`: **`load_regional_univariate_data`**, **`univariate_analysis_pipeline`**. The legacy script is ~567 lines and wraps both data loading and pipeline fitting into one CLI.
- `mnm_regression.ipynb [ensemble_twas_weight]`: **`ensemble_weights`** (+ `load_regional_univariate_data` for resume-cache reads). (Out of scope per user directive but logged here for context.)
- `colocboost.ipynb [colocboost]`: uses `library(colocboost)` and `library(pecotmr)`; relies on the SoS `[get_analysis_regions]` upstream which uses the deprecated multi-task data-loader convention (manifest-driven). The R block itself likely calls `pecotmr` helpers for data assembly and the `colocboost::colocboost()` solver directly.
- `rss_analysis.ipynb [univariate_rss]` → `univariate_rss.R`: **`load_LD_matrix`** (deprecated → `loadLdMatrix`). The script is ~265 lines.
- `twas_ctwas.ipynb`:
  - `[twas]` — **`twas_pipeline`**, **`load_twas_weights`**, **`batch_load_twas_weights`**, **`twas_z`** + various `mr_*` MR helpers.
  - `[ctwas_1]` — **`get_ctwas_meta_data`**, **`harmonize_gwas`**, **`load_LD_matrix`**, **`twas_analysis`**, plus internal `ctwas_*` data structures.
  - `[ctwas_2]` — internal `ctwas_param_*` flow.
  - `[ctwas_3]` — **`ctwas_bimfile_loader`**, **`get_ctwas_meta_data`**, plus `ctwas_*` plotting/postprocess helpers.
  - `[quantile_twas]` — `twas_pipeline`, `mr_result`.
- `SuSiE_enloc.ipynb`:
  - `[xqtl_gwas_enrichment]` — **`xqtl_enrichment_wrapper`** (deprecated → `qtlEnrichmentPipeline`).
  - `[susie_coloc]` — calls `pecotmr:::get_nested_element` (internal helper) on a per-context fine-mapped RDS; the per-context coloc loop is built around `library(coloc)` directly.
- `intact.ipynb [intact]` — `fastenloc`, `run_intact`, `intact_pip`, `intact_res`. No direct pecotmr-pipeline call observed; mostly a downstream synthesizer. Needs deeper check before concluding.

## MWE data conventions discovered

- **Genotype**: BOTH PLINK1 (e.g. `input/colocboost/example.chr22.bed/.bim/.fam`) AND VCF / bgzipped formats (`input/genotype/protocol_example.genotype.chr*.vcf.gz`, `*.bgz`). **No PLINK2 (pgen/pvar/psam) files present** in the MWE.
- **Phenotype BED**: standard QTLtools format with `#chr, start, end, ID, SAMPLE_001, SAMPLE_002, ...`. Matches what our new `qtl_dataset_construct.R` parses.
- **Covariate file**: **QTLtools convention** — `#id` header + rows like `sex`, `age`, `PC1`, ... with sample columns. **This is the opposite orientation from what the new wrapper expects** (we updated the script to take samples-as-rows since the user described `--phenotype-covariates` as "molecular trait PCs, same shape as genotype PCs").
- **Per-region manifest TSVs** (`fine_mapping_meta.tsv`, `pheno_manifest_multicontext.tsv`): columns `#chr, start, end, ID, path, cond, cov_path` — one row per (region, context). The same `cov_path` is reused across contexts in the MWE examples.
- **xQTL meta TSVs** (TWAS / cTWAS / enloc inputs): point at pre-computed per-(study, gene) `*.univariate_susie_twas_weights.rds` files. These workflows do NOT recompute fits; they consume the pre-built RDS files.
- **GWAS inputs** (`input/rss_analysis/`, `input/twas/`): tabix-indexed `.tsv.gz` plus a YAML column-mapping file.

## Migration sketch by notebook

1. **`mnm_regression.ipynb` `[susie_twas]` step**: delete the step and `susie_twas.R`; replace with the existing `fine_mapping.ipynb` + `twas_weights.ipynb` against a pre-built QtlDataset RDS.
2. **`rss_analysis.ipynb`**: new wrapper `gwas_finemap.R` that takes a QC'd `GwasSumStats` RDS and calls `fineMappingPipeline(methods = "susie")`. The legacy `[univariate_rss]` step's inputs (sumstats TSV + LD meta) feed a prior `summaryStatsQc` step that builds the GwasSumStats RDS.
3. **`colocboost.ipynb`** (mnm_analysis): the existing notebook covers individual-level cross-context coloc with `colocboost::colocboost()` invoked through pecotmr's old data assembly. The new `pecotmr_integration/colocboost.ipynb` we wrote covers the QtlDataset focal-gene case; multi-condition / GWAS-jointed colocboost can be added similarly via `colocboostPipeline(jointGwas = TRUE, ...)`.
4. **`qr_and_twas.ipynb`**: quantile-regression TWAS. Need to confirm whether pecotmr's S4 refactor exposes a new entry point for quantile TWAS or whether the legacy `load_regional_association_data` + a `quantile_*` worker is still the only path.
5. **`twas_ctwas.ipynb`**: maps to the new `ctwasPipeline` + `causalInferencePipeline` in pecotmr. Each step (`twas`, `ctwas_1/2/3`, `quantile_twas`) becomes a thin wrapper that takes the pre-computed TwasWeights RDS + GWAS RDS + LD meta and calls one pipeline function. The data flow stays the same; only the inline R block is replaced.
6. **`SuSiE_enloc.ipynb`**:
   - `[xqtl_gwas_enrichment]` → `qtlEnrichmentPipeline(gwasFineMappingResult, qtlFineMappingResult)`. Input meta TSV stays the same; the wrapper just calls the new pipeline.
   - `[susie_coloc]` → `colocPipeline(qtlFineMappingResult, gwasFineMappingResult)` or similar. The new pipeline handles the per-(study, region) coloc loop internally; no need to hand-write the loop in the SoS notebook.
7. **`intact.ipynb`**: downstream of TWAS + enloc; uses `fastenloc` and `run_intact`. These look like INTACT-specific functions, not pecotmr S4 pipeline targets. Likely the inline R only consumes RDS outputs from earlier steps and a thin wrapper that calls the same `run_intact` is enough. Confirm whether `run_intact` is a pecotmr export or an INTACT-package function.

## Status update (2026-06-22)

**colocboost extension done.** `colocboost.R` + `colocboost.ipynb` now accept `--gwas-sumstats <RDS>` plus the three pipeline-variant flags (`--no-xqtl-coloc`, `--joint-gwas`, `--separate-gwas`), with the matching SoS parameter validations. xqtl-only regression and the GWAS-variant validations all smoke-tested against the MWE. Joint/separate-gwas end-to-end testing still needs a QC'd GwasSumStats RDS.

**twas_ctwas.ipynb migration — wrappers + notebooks written.** Three new pairs under `code/SoS/pecotmr_integration/` + `code/script/pecotmr_integration/`:

- `gwas_sumstats_construct.R` + `gwas_sumstats.ipynb` — per-LD-block GwasSumStats builder. Reads a GWAS TSV, subsets to the block, resolves the LD-meta row, builds a PLINK1/PLINK2/GDS/VCF `GenotypeHandle`, calls `GwasSumStats(...)`. Has a `--skip-qc` flag escape hatch because `summaryStatsQc()` triggers MungeSumstats which requires the `SNPlocs.Hsapiens.dbSNP155.GRCh38` BioConductor package. **Smoke-tested against the MWE** — produces a valid `GwasSumStats` over chr22:10516173-17414263 (112 variants) with the PLINK2 LD sketch attached.
- `twas.R` + `twas.ipynb` — per-gene `causalInferencePipeline`. Notebook is now manifest-driven (rows: `gene_id`, `twas_weights_rds`, `gwas_sumstats_rds`, optional `fine_mapping_result_rds`), matching the `ctwas.ipynb` pattern. Per-row fan-out, one `twas.R` call per gene. Output: `{cwd}/{study}.{gene_id}.twas.rds`.
- `fine_mapping.R` + `fine_mapping.ipynb` — supports BOTH QTL (`fine_mapping` workflow, `--qtl-dataset` + `--genes`/`--regions` fan-out) AND GWAS (`fine_mapping_gwas` workflow, `--gwas-sumstats-list <RDS>...` per-block fan-out). One wrapper script dispatches on which S4 input was supplied; `pecotmr::fineMappingPipeline` dispatches on input class (QtlDataset / GwasSumStats). Smoked end-to-end on the MWE: produces `QtlFineMappingResult` for ENSG00000130538 and `GwasFineMappingResult` for all 20 chr22 LD blocks.
- `ctwas_assemble.R` + `ctwas_est.R` + `ctwas_finemap.R` + `ctwas.ipynb` — three-step cTWAS pipeline mirroring the legacy `[ctwas_1/2/3]`. Each step calls one of pecotmr's exported sub-pipelines (`assembleCtwasInputs` / `estCtwasParam` / `screenCtwasRegions` + `finemapCtwasRegions`). The SoS notebook chains them via intermediate RDS files so analysts can re-run only the screen/finemap step when tuning knobs.

**Bugs surfaced in installed pecotmr 0.5.3 during testing**:
1. `.twasNormalizeMethods("default")` echoes the literal `"default"` as the only token (skips the `methodList`-based expansion that the `is.null` branch uses). Workaround: wrappers pass `methods = NULL`.
2. `pecotmr:::getRegionalLdMeta` returns the LD-meta file path as the *value* with the data path as the *name* of a single-element character vector, so `GenotypeHandle(ldMeta=…, region=…)` ends up checking the meta file's `.tsv` extension instead of the genotype payload's. Workaround: `gwas_sumstats_construct.R` resolves LD-meta rows directly and calls `GenotypeHandle(plink2Prefix=…)` (or the matching constructor) itself.
3. `cisWindow` is required in `fineMappingPipeline(... region = ...)` despite the doc saying it's required only with `traitId`. Wrappers pass `cisWindow` in both modes.

## Outstanding questions for the user

1. **Covariate orientation**: MWE files are QTLtools format (covariates × samples). Our new `qtl_dataset_construct.R` expects samples × PCs (transposed). Options: (a) transpose inside the wrapper, (b) make orientation a flag, (c) keep current and require pre-transposition. Which?
2. **Genotype format**: MWE has PLINK1 + VCF, no PLINK2. Do you want to (a) convert MWE inputs to PLINK2 once, (b) extend `qtl_dataset_construct.R` to accept PLINK1 / VCF, or (c) leave the wrapper PLINK2-only?
3. **Meta TSV consumption**: the legacy workflow ingests per-region manifest TSVs (`fine_mapping_meta.tsv`) and expands them into per-region data dicts via `[get_analysis_regions]`. Under the new design, where does the meta TSV live? Options: (a) supply CLI args directly to `qtl_dataset_construct.R` (current), (b) add a CLI flag that reads a meta TSV and builds the args internally, (c) leave the TSV→args translation as a one-off pre-step.
4. **`--region` vs `--window`**: legacy CLI distinguished a per-gene fine-mapping region from a larger association window for variant selection. The new wrappers only expose `--cis-window` (per-trait). Confirm this consolidation is intentional and the wider association-window concept is gone.
5. **Quantile TWAS**: does pecotmr's new S4 API have a quantile-TWAS pipeline (`quantileTwasPipeline` or similar), or is quantile TWAS still bolted on top of the legacy `load_regional_association_data` flow?
6. **Pre-computed TWAS weights inputs**: `twas_ctwas` / `SuSiE_enloc` / `intact` all consume pre-computed `*.univariate_susie_twas_weights.rds` files. After the migration of `twas_weights.ipynb`, the RDS schema is the new `TwasWeights` S4 collection — which should be a drop-in replacement everywhere the old `.rds` was consumed via `loadTwasWeights` / `batchLoadTwasWeights`. Confirm: are we OK requiring downstream notebooks to take new-format `TwasWeights` RDS files, or do we need a back-compat shim?
7. **`ensemble_weights` (mnm_regression `[ensemble_twas_weight]`)**: still wants to be a callable. Has it been moved to a camelCase `ensembleWeights` (still exported), or removed in the S4 refactor? Affects whether the ensemble step is a 1-line wrapper or needs reimplementation.
8. **Scope confirmation**: per user directive, `mnm_regression.ipynb` is treated as just `[susie_twas]` for this round; `[ensemble_twas_weight]`, `[mnm]`, `[mnm_genes]`, `[fsusie]`, `[mvfsusie]` are deferred. Confirm.
9. **`intact.ipynb`**: does the `intact` step call into pecotmr (deprecated or otherwise) or is it purely the INTACT R package + RDS file munging? Determines whether this notebook needs migration at all.
