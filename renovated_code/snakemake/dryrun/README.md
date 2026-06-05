# Local MWE Runtime Helpers

This directory keeps the small host-side helpers used by the Modular SoS MWE
runner. It does not contain generated run outputs.

## Scripts

- `run_modular_sos_mwe_snakemake.sh`: prepares a temporary MWE run directory,
  writes a Modular SoS config, and runs Snakemake.
- `prepare_modular_sos_mwe_inputs.sh`: normalizes the external MWE data into the
  layout expected by the Snakemake config.
- `activate_local_pixi.sh` and `_local_pixi_common.sh`: optional local Pixi
  activation helpers for this checkout.
- `check_local_pixi_env.sh`: verifies the local Pixi tools when that environment
  is available.

## Pixi Scope

The Pixi environment itself is intentionally not part of the source bundle.
The ignored local holder is:

```bash
renovated_code/snakemake/dryrun/bin/
```

The default live Pixi home used on this machine is outside the repository:

```bash
../mwe_data/.pixi
```

## Example

From the repository root:

```bash
renovated_code/snakemake/modular_sos/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../mwe_data \
  --run-tag modular_sos_mwe_core \
  --cores 1
```

Use `--target all` only when fine-mapping plot generation is required.
