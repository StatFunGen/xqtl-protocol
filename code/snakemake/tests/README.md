# script-backed Minimal Runtime Tests

This directory keeps two script-backed test surfaces:

1. `run_mwe_xqtl_core.sh` runs the script-backed Snakemake DAG through TensorQTL and SuSiE/TWAS fine-mapping, excluding plots via the `xqtl_core` target.
2. `run_nontrivial_tensorqtl_susie.sh` reruns legacy-vs-script-backed notebook compares for TensorQTL and SuSiE/TWAS using `data/nontrivial_tensorqtl_susie.tar.gz`.

The script-backed runtime remains notebook-first: Snakemake rules call the canonical notebooks under `pipeline/`, and those notebooks call the modular scripts. The rule set kept active is `00` through `06`; `xqtl_core` includes rule `06` SuSiE/TWAS and excludes only the plot target.

## Run The MWE

From the repository root:

```bash
code/snakemake/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../xqtl-renovated/mwe_data \
  --run-tag xqtl_mwe_core \
  --cores 1
```

`--mwe-data` can be a directory, `.tar.gz`/`.tgz`, or `.zip` containing the MWE data root with `AC_sample_fastq.list`. The raw MWE data is intentionally external to this cleanup bundle; the local source tree is 73G and includes 8.9G FASTQ, 3.9G BAM, and a 5.7G VCF.

To include fine-mapping plots, change the target explicitly:

```bash
code/snakemake/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../xqtl-renovated/mwe_data \
  --run-tag xqtl_mwe_all \
  --cores 1 \
  --target all
```

## Run Non-Trivial Notebook Compares

```bash
code/snakemake/tests/run_nontrivial_tensorqtl_susie.sh \
  --run-tag xqtl_nontrivial_compare \
  --num-threads 4
```

This unpacks `data/nontrivial_tensorqtl_susie.tar.gz`, runs:

- legacy `code/SoS/association_scan/TensorQTL/TensorQTL.ipynb cis`
- script-backed `pipeline/TensorQTL.ipynb cis`
- legacy `code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb susie_twas`
- script-backed `pipeline/mnm_regression.ipynb susie_twas`

The test fails if TensorQTL output MD5s differ or if SuSiE/TWAS head records differ. Outputs are written under `code/snakemake/tmp/xqtl_tests/`.

## Run RSS Analysis MWE

```bash
code/snakemake/tests/run_rss_analysis_mwe.sh
```

This builds or reuses the small `pecotmr` toy-data fixture under `code/snakemake/tests/data/rss_analysis_mwe/`, then runs the RSS notebook stages directly through SoS: `get_analysis_regions`, `univariate_rss`, and `univariate_plot`. The test asserts the legacy notebook contract that the saved RDS top-level key is `chr22_49355984_50799822`; the canonical `chr22:49355984-50799822` region is used to load the LD block, not as the saved-key string.

## Run Other Downstream Data

Use the script-backed Snakefile directly with the data-specific config:

```bash
snakemake \
  --snakefile code/snakemake/Snakefile \
  --configfile path/to/xqtl.config.yaml \
  --cores 1 \
  xqtl_core
```

Use target `all` only when plot generation is required.

Fidelity verdict: `faithful`. These commands preserve the legacy notebook stage graph and script-backed notebook-first orchestration; the only excluded target in `xqtl_core` is fine-mapping plot generation.
