# Python replacement scope

Inventory of all Python code in xqtl-protocol, categorized by what we plan to
do with it. Companion to the pecotmr refactor (see
`StatFunGen/pecotmr/dev/refactor-design.md`) — most of the "internal Python"
absorption happens inside pecotmr constructors and pipelines after that
refactor lands.

## Out of scope

The following Python is **not** considered for replacement and is excluded
from every category below.

- **SoS framework scaffolding** (~3,065 LOC across all SoS notebooks): the
  `[step_name]` headers, `parameter:`, `input:`, `output:`, `task:`, and
  `depends:` directives, plus action-block headers (`R:`, `bash:`,
  `python:`, etc.). These use Python syntax but are SoS framework
  declarations, not Python you write. Stays implicitly because SoS itself
  stays.
- **Snakemake** (`code/snakemake/` directory in its entirety, including
  `compat/python/sitecustomize.py`, `Snakefile`, and `rules/*`). Out of
  scope for this work.

## Categories

There are four buckets of Python in xqtl-protocol. Categories 1 contains
code we keep; categories 2–4 contain code we plan to replace.

### Category 1 — Vendored Python scripts and external tools (KEEP)

Vendored scripts and external tool wrappers we do not plan to replace.
These are either copies of upstream tools or wrappers around large
external libraries (ML / GPU frameworks) with no clean R equivalent. They
are invoked from the workflow layer via shell-out (`bash:` blocks or
`subprocess`), so the rest of the codebase doesn't care about their
language.

| File | LOC | Origin / purpose |
|---|---|---|
| `code/SoS/molecular_phenotypes/calling/apa/DaPars_Extract_Anno.py` | 152 | Vendored from the [DaPars](https://github.com/ZhengXia/DaPars) APA-calling project. Extracts 3'UTR annotation from gene BED + symbol map. **Python 2.** |
| `code/SoS/molecular_phenotypes/calling/apa/Dapars2_Multi_Sample.py` | 448 | Vendored DaPars2 multi-sample driver. Same project family as `DaPars_Extract_Anno.py`. **Python 2.** |
| `code/SoS/molecular_phenotypes/calling/apa/gtf2bed12.py` | 209 | Vendored GTF→BED12 converter (Xudong Zou, 2021). Preprocessing step for APA workflows. **Python 2.** |
| `code/SoS/xqtl_modifier_score/gems_pipeline.py` | 678 | GEMS (Generalized Expression Modifier Scores) ML pipeline. Uses sklearn, torch, xgboost, optuna. Self-contained CLI with `train` and `predict` subcommands; invoked from `ems_training.ipynb` / `ems_prediction.ipynb` via `bash:` blocks. |
| `code/script/association_scan/TensorQTL/TensorQTL.py` | 693 | Wrapper around the `tensorqtl` Python library for QTL association testing (cis/trans, nominal + permutation). |
| **Total** | **2,180** | |

### Category 2 — SoS notebooks with step-body Python (REPLACE)

Notebooks running under the SoS kernel that contain Python written
**outside** the `R:` / `bash:` action blocks. This includes both:

- Step-body Python at the SoS step level (manifest assembly, region
  parsing, file-path resolution, conditional logic, output-filename
  templating)
- `python:` / `python3:` action block bodies (Python executed as the
  step's action)

After the pecotmr refactor, the data-construction slice of this Python
(region parsing, GWAS-meta TSV reading, manifest assembly, per-trait /
per-context path resolution) is absorbed into pecotmr constructors. The
remaining step-body Python — per-job argument shaping, output-filename
templating, control flow — is what we want to replace with bash + R.

**82 SoS notebooks have some step-body Python.** The 16 heavy hitters
(≥100 LOC) are where the bulk of the work lives:

| Notebook | Step-body LOC |
|---|---|
| `code/SoS/pecotmr_integration/twas_ctwas.ipynb` | 673 |
| `code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb` | 589 |
| `code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb` | 389 |
| `code/SoS/commands_generator/eQTL_analysis_commands.ipynb` | 267 |
| `code/SoS/mnm_analysis/mnm_postprocessing.ipynb` | 247 |
| `code/SoS/association_scan/quantile_models/qr_and_twas.ipynb` | 243 |
| `code/SoS/molecular_phenotypes/QC/pseudobulk_expression_aggregation_QC_norm.ipynb` | 243 *(mixed-kernel — see Category 3)* |
| `code/SoS/pecotmr_integration/SuSiE_enloc.ipynb` | 239 |
| `code/SoS/enrichment/sldsc_enrichment.ipynb` | 211 |
| `code/SoS/molecular_phenotypes/calling/RNA_calling.ipynb` | 182 |
| `code/SoS/data_preprocessing/genotype/genotype_formatting.ipynb` | 151 |
| `code/SoS/data_preprocessing/genotype/GWAS_QC.ipynb` | 137 |
| `code/SoS/molecular_phenotypes/QC/pseudobulk_mega_expression_QC_and_normalization.ipynb` | 130 *(mixed-kernel — see Category 3)* |
| `code/SoS/mnm_analysis/mnm_methods/rss_analysis.ipynb` | 129 |
| `code/SoS/association_scan/TensorQTL/TensorQTL.ipynb` | 112 |
| `code/SoS/data_preprocessing/genotype/PCA.ipynb` | 104 |

**32 more notebooks have 20–99 LOC of step-body Python** (medium size; mostly
smaller workflows and graveyard files). **34 more have <20 LOC** (incidental
and cleanable in passing).

Aggregate Python in SoS code cells across all SoS notebooks:

- **5,655 LOC** of step-body Python (outside action blocks)
- **1,011 LOC** of `python:` action block bodies
- **106 LOC** of helper `def`/`class` definitions

Total Category 2 LOC: **~6,772**.

### Category 3 — Notebooks with Python kernels (REPLACE)

Notebooks whose notebook-level `kernelspec` is Python (i.e., not SoS
workflows — standalone Python notebooks). These are mostly analysis,
plotting, and ML-driver notebooks.

| Notebook | Code LOC | Notes |
|---|---|---|
| `code/SoS/cv2f/notebooks/analyze_selection_criteria.ipynb` | 738 | Analysis of cross-validation selection in fine-mapping |
| `code/SoS/reference_data/rss_ld_sketch.ipynb` | 419 | **Mixed-kernel.** Notebook-level kernel is Python, but most cells are R-implicit (R code in cells with no per-cell kernel). True Python content is small. |
| `code/SoS/molecular_phenotypes/QC/pseudobulk_expression_aggregation_QC_norm.ipynb` | 282 | **Mixed-kernel.** Also appears in Category 2 with 243 LOC of SoS step-body Python. |
| `code/SoS/molecular_phenotypes/QC/pseudobulk_mega_expression_QC_and_normalization.ipynb` | 149 | **Mixed-kernel.** Also appears in Category 2 with 130 LOC of SoS step-body Python. |
| `code/SoS/reference_data/ld_reference_generation.ipynb` | 146 | LD reference data prep |
| `code/SoS/xqtl_modifier_score/ems_prediction.ipynb` | 117 | Driver for `gems_pipeline.py predict` (Category 1). Mostly shell-out invocations. |
| `code/SoS/xqtl_modifier_score/ems_training.ipynb` | 76 | Driver for `gems_pipeline.py train`. Mostly shell-out invocations. |
| `code/SoS/mnm_analysis/mnm_miniprotocol.ipynb` | 62 | Mini-protocol tutorial |
| `code/SoS/mnm_analysis/multivariate_multigene_fine_mapping_vignette.ipynb` | 21 | Vignette |
| `code/SoS/mnm_analysis/multivariate_fine_mapping_vignette.ipynb` | 21 | Vignette |
| `code/SoS/mnm_analysis/univariate_fine_mapping_fsusie_vignette.ipynb` | 11 | Vignette |
| `code/SoS/mnm_analysis/summary_stats_finemapping_vignette.ipynb` | 9 | Vignette |
| `code/SoS/pecotmr_integration/twas_vignette.ipynb` | 0 | Markdown-only |
| **Total** | **~2,051** | |

Two additional notes:

- `website/nature_protocol/conversion_notebook.ipynb` (516 LOC) is a
  Python-kernel notebook used for the protocol website's doc generation.
  Listed for completeness but is not analysis code; treat separately if /
  when website automation gets touched.
- The three **mixed-kernel notebooks** (`rss_ld_sketch.ipynb`,
  `pseudobulk_expression_aggregation_QC_norm.ipynb`,
  `pseudobulk_mega_expression_QC_and_normalization.ipynb`) appear in both
  Category 2 (their SoS-cell step-body Python) and Category 3 (their
  notebook-level Python kernel). When tackling these, the work spans
  both flavors.

### Category 4 — Standalone Python scripts to replace (REPLACE)

`.py` files outside of the vendored / external set, mostly under
`code/script/`. These are real production code that wraps existing R-side
libraries (edgeR, limma, tabix, plink) or does data manipulation that
ports cleanly to R or bash.

| File | LOC | What it does | Replacement target |
|---|---|---|---|
| `code/script/association_scan/TensorQTL/TensorQTL.py` | 693 | *(Category 1 — kept as-is; listed here only to flag that we are not replacing it)* | — |
| `code/script/data_preprocessing/phenotype/gene_annotation.py` | 589 | Maps leafcutter / psichomics splicing clusters and isoforms to genes. Pandas-heavy. | R |
| `code/script/data_preprocessing/phenotype/phenotype_formatting.py` | 325 | Phenotype file format conversions (BED, tabix prep). | Bash + R |
| `code/script/molecular_phenotypes/calling/RNA_calling.py` | 316 | Merges STAR / RSEM GCT outputs into a combined expression matrix. | R |
| `code/script/molecular_phenotypes/QC/bulk_expression_normalization.py` | 256 | TMM / CPM / voom / quantile normalization. Already wraps edgeR / limma via rpy2. | Native R port |
| `code/script/data_preprocessing/genotype/genotype_formatting.py` | 247 | VCF / PLINK format conversions and splitting. Mostly `subprocess` calls. | Bash |
| **Total to replace** | **~1,733** | | |

(Note: `TensorQTL.py` is shown in the table for completeness but is part of
Category 1 — we are explicitly not replacing it. Excluding it, the
replacement target in Category 4 is ~1,733 LOC.)

## Summary

| Category | Disposition | Approx. LOC | File / notebook count |
|---|---|---|---|
| 1. Vendored Python / external tools | Keep | 2,180 | 5 files |
| 2. SoS notebooks with step-body Python | Replace | 6,772 | 82 notebooks (16 heavy hitters) |
| 3. Notebooks with Python kernels | Replace | 2,051 | 13 notebooks (3 mixed-kernel) |
| 4. Standalone Python scripts (excl. TensorQTL) | Replace | 1,733 | 5 files |
| **Replacement target** | | **10,556** | |
| **Out of scope** (SoS scaffolding) | Implicit keep | ~3,065 | — |
| **Out of scope** (Snakemake) | Excluded | ~125+ | `code/snakemake/` tree |

After the pecotmr refactor absorbs the data-construction slice of Category 2
(rough estimate: 2,000–3,000 LOC of region parsing / manifest assembly /
GWAS-meta TSV reading), the remaining replacement target is approximately
**7,500–8,500 LOC** of step-body orchestration Python, Python-kernel analysis
notebooks, and standalone scripts. Roughly 80% of this is straightforwardly
portable to R or bash; the other 20% is per-step orchestration that needs to
be rewritten in the SoS notebooks (as inline bash + R, replacing the
imperative Python that's there today).
