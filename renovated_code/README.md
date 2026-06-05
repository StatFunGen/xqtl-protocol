# Renovated Code Runtime Handoff

## Human Note

This folder now contains the minimal Modular SoS runtime bundle for the refactored
xQTL workflow. The active runtime is no longer named `route3`; use
`renovated_code/snakemake/modular_sos/`.

The committed source bundle includes the Snakemake rules, generated SoS wrapper
notebooks, modular scripts, a small non-trivial TensorQTL/SuSiE test package, and
host-side MWE runner scripts. It does not include the full MWE data tree or the
local Pixi environment holder.

Primary commands:

```bash
# MWE through TensorQTL and SuSiE/TWAS, excluding plots
renovated_code/snakemake/modular_sos/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../mwe_data \
  --run-tag modular_sos_mwe_core \
  --cores 1

# Non-trivial legacy-vs-Modular-SoS notebook comparison
renovated_code/snakemake/modular_sos/tests/run_nontrivial_tensorqtl_susie.sh \
  --run-tag modular_sos_nontrivial_compare \
  --num-threads 4
```

Use target `all` only when fine-mapping plots are needed.

## AI Handoff

### Current Objective

Preserve the legacy mini-protocol workflow semantics while shipping the renovated
implementation as a minimal runtime bundle. The source-of-truth workflow intent
remains the legacy mini-protocol notebook layer under `pipeline/`; the Modular
SoS bundle is the refactored Snakemake/SoS/script translation.

Fidelity verdict: `faithful`. The committed runtime keeps the notebook-first
stage graph and the rule structure `00` through `06`; it does not collapse
notebook stages into one script.

### Active Runtime Surface

Canonical Snakemake entry point:

```bash
renovated_code/snakemake/modular_sos/Snakefile
```

Active rule modules:

```text
renovated_code/snakemake/modular_sos/rules/00_phenotype_preprocessing.smk
renovated_code/snakemake/modular_sos/rules/01_molecular_phenotypes.smk
renovated_code/snakemake/modular_sos/rules/02_genotype_preprocessing.smk
renovated_code/snakemake/modular_sos/rules/03_sample_qc_pca.smk
renovated_code/snakemake/modular_sos/rules/04_phenotype_covariate_prep.smk
renovated_code/snakemake/modular_sos/rules/05_association_testing.smk
renovated_code/snakemake/modular_sos/rules/06_univariate_finemapping.smk
```

Active generated wrapper notebooks:

```text
GWAS_QC.ipynb
PCA.ipynb
RNA_calling.ipynb
TensorQTL.ipynb
VCF_QC.ipynb
bulk_expression_QC.ipynb
bulk_expression_normalization.ipynb
covariate_formatting.ipynb
covariate_hidden_factor.ipynb
gene_annotation.ipynb
genotype_formatting.ipynb
mnm_regression.ipynb
phenotype_formatting.ipynb
phenotype_imputation.ipynb
```

The Modular SoS Snakefile executes:

```text
Snakemake -> sos run renovated_code/snakemake/modular_sos/notebooks/<notebook>.ipynb <step>
```

Those notebooks call modular scripts under `renovated_code/script/`.

### Targets

`xqtl_core` runs the complete runtime through TensorQTL and SuSiE/TWAS
fine-mapping, excluding fine-mapping plot generation.

```bash
snakemake \
  --snakefile renovated_code/snakemake/modular_sos/Snakefile \
  --configfile path/to/modular_sos.config.yaml \
  --cores 1 \
  xqtl_core
```

`all` includes the plot target:

```bash
snakemake \
  --snakefile renovated_code/snakemake/modular_sos/Snakefile \
  --configfile path/to/modular_sos.config.yaml \
  --cores 1 \
  all
```

### MWE Runner

MWE driver:

```bash
renovated_code/snakemake/modular_sos/tests/run_mwe_xqtl_core.sh
```

It wraps:

```bash
renovated_code/snakemake/dryrun/run_modular_sos_mwe_snakemake.sh
renovated_code/snakemake/dryrun/prepare_modular_sos_mwe_inputs.sh
```

The full MWE data is external. The default expected path is:

```bash
../mwe_data
```

The MWE data input can also be a `.tar.gz`, `.tgz`, or `.zip` containing a root
with:

```text
AC_sample_fastq.list
```

Observed local size:

```text
../mwe_data = 73G
```

Large local components include `.pixi` at 22G, a PLINK bed at 27G, FASTQ at
8.9G, VCF at 5.7G, and BAM at 3.9G. Do not commit these.

### Non-Trivial Test Package

Committed fixture:

```bash
renovated_code/snakemake/modular_sos/tests/data/modular_sos_nontrivial_tensorqtl_susie.tar.gz
```

Committed size:

```text
282,949 bytes on git object listing; 288K by du
```

Test driver:

```bash
renovated_code/snakemake/modular_sos/tests/run_nontrivial_tensorqtl_susie.sh
```

It runs four notebook invocations:

```text
legacy TensorQTL.ipynb cis
Modular SoS TensorQTL.ipynb cis
legacy mnm_regression.ipynb susie_twas
Modular SoS mnm_regression.ipynb susie_twas
```

It fails if:

- TensorQTL output MD5s differ.
- TensorQTL regional output has no finite non-trivial p-value row.
- SuSiE/TWAS head records differ.

### Local Pixi Scope

The Pixi environment holder is intentionally excluded from git.

Ignored paths include:

```text
.pixi/
**/.pixi/
renovated_code/snakemake/dryrun/bin/
renovated_code/snakemake/tmp/
renovated_code/snakemake/archive/
```

The local activation helpers are committed for record/reproducibility:

```text
renovated_code/snakemake/dryrun/LOCAL_PIXI_ENV.md
renovated_code/snakemake/dryrun/_local_pixi_common.sh
renovated_code/snakemake/dryrun/activate_local_pixi.sh
renovated_code/snakemake/dryrun/check_local_pixi_env.sh
```

These scripts assume the external local Pixi install when present:

```text
../mwe_data/.pixi
```

### Validation State

Committed runtime bundle commit:

```text
d0f822d3 Add Modular SoS runtime bundle
```

Post-namespace static checks run before that commit:

```bash
bash -n renovated_code/snakemake/dryrun/run_modular_sos_mwe_snakemake.sh \
  renovated_code/snakemake/dryrun/prepare_modular_sos_mwe_inputs.sh \
  renovated_code/snakemake/modular_sos/tests/run_mwe_xqtl_core.sh \
  renovated_code/snakemake/modular_sos/tests/run_nontrivial_tensorqtl_susie.sh

python -m py_compile \
  renovated_code/snakemake/compat/python/sitecustomize.py \
  renovated_code/script/data_preprocessing/genotype/genotype_formatting.py \
  renovated_code/script/data_preprocessing/phenotype/gene_annotation.py \
  renovated_code/script/molecular_phenotypes/QC/bulk_expression_normalization.py \
  renovated_code/script/molecular_phenotypes/calling/RNA_calling.py \
  renovated_code/script/association_scan/TensorQTL/TensorQTL.py
```

Namespace checks after commit:

```bash
git grep -n -E 'route3|Route3|Route 3|ROUTE3|route 3' HEAD -- renovated_code ':!renovated_code/snakemake/archive'
```

returned no matches.

Important limitation: after the final namespace rename to `modular_sos`, I did
not rerun the full Snakemake MWE. Earlier non-trivial TensorQTL/SuSiE comparison
completed before the final namespace rename; the namespace rename was then
validated by static checks and fixture listing.

### Current Git State At Handoff

Branch:

```text
notebook-only-old-notebook-fixes-clean
```

Before pushing, the branch was ahead of `origin/main` by 31 commits and behind
by 112 commits. This PR will therefore represent the current refactor branch
stack, not only the final README commit.

Known unstaged files at this point were outside the committed runtime bundle:

```text
README.md
CODEX_HANDOFF.md
code/association_scan/TensorQTL/TensorQTL.ipynb
code/commands_generator/eQTL_analysis_commands.ipynb
code/data_preprocessing/genotype/GWAS_QC.ipynb
code/data_preprocessing/genotype/VCF_QC.ipynb
code/misc/data_preprocessing/2_genotype_preprocessing.ipynb
```

Do not accidentally include those unless the user explicitly wants them.

### Recommended Next Steps

1. Push the current branch.
2. Open a PR against `origin/main`.
3. In the PR body, call out that MWE data is external and the committed
   non-trivial test package is 288K.
4. If runtime verification is requested, run `run_mwe_xqtl_core.sh` against the
   external MWE data and run `run_nontrivial_tensorqtl_susie.sh`.
