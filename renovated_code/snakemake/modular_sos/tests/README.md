# Modular SoS Minimal Runtime Tests

This directory keeps two Modular SoS test surfaces:

1. `run_mwe_xqtl_core.sh` runs the Modular SoS Snakemake DAG through TensorQTL and SuSiE/TWAS fine-mapping, excluding plots via the `xqtl_core` target.
2. `run_nontrivial_tensorqtl_susie.sh` reruns legacy-vs-Modular SoS notebook compares for TensorQTL and SuSiE/TWAS using `data/modular_sos_nontrivial_tensorqtl_susie.tar.gz`.

The Modular SoS runtime remains notebook-first: Snakemake rules call the canonical modular notebooks under `renovated_code/notebook/modular_sos`, and those notebooks call the modular scripts. The rule set kept active is `00` through `06`; `xqtl_core` includes rule `06` SuSiE/TWAS and excludes only the plot target.

## Run The MWE

From the repository root:

```bash
renovated_code/snakemake/modular_sos/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../mwe_data \
  --run-tag modular_sos_mwe_core \
  --cores 1
```

`--mwe-data` can be a directory, `.tar.gz`/`.tgz`, or `.zip` containing the MWE data root with `AC_sample_fastq.list`. The raw MWE data is intentionally external to this cleanup bundle; the local source tree is 73G and includes 8.9G FASTQ, 3.9G BAM, and a 5.7G VCF.

To include fine-mapping plots, change the target explicitly:

```bash
renovated_code/snakemake/modular_sos/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../mwe_data \
  --run-tag modular_sos_mwe_all \
  --cores 1 \
  --target all
```

## Run Non-Trivial Notebook Compares

```bash
renovated_code/snakemake/modular_sos/tests/run_nontrivial_tensorqtl_susie.sh \
  --run-tag modular_sos_nontrivial_compare \
  --num-threads 4
```

This unpacks `data/modular_sos_nontrivial_tensorqtl_susie.tar.gz`, runs:

- legacy `code/association_scan/TensorQTL/TensorQTL.ipynb cis`
- Modular SoS `renovated_code/notebook/modular_sos/TensorQTL.ipynb cis`
- legacy `code/mnm_analysis/mnm_methods/mnm_regression.ipynb susie_twas`
- Modular SoS `renovated_code/notebook/modular_sos/mnm_regression.ipynb susie_twas`

The test fails if TensorQTL output MD5s differ or if SuSiE/TWAS head records differ. Outputs are written under `renovated_code/snakemake/tmp/modular_sos_tests/`.

## Run MESA Or Other Downstream Data

Use the Modular SoS Snakefile directly with the data-specific config:

```bash
snakemake \
  --snakefile renovated_code/snakemake/modular_sos/Snakefile \
  --configfile path/to/modular_sos.config.yaml \
  --cores 1 \
  xqtl_core
```

Use target `all` only when plot generation is required.

Fidelity verdict: `faithful`. These commands preserve the legacy notebook stage graph and Modular SoS notebook-first orchestration; the only excluded target in `xqtl_core` is fine-mapping plot generation.
