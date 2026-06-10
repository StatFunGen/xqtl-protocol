# Local MWE Runtime Helpers

This directory keeps the small host-side helpers used by the script-backed MWE
runner. It does not contain generated run outputs.

## Scripts

- `run_mwe_snakemake.sh`: prepares a temporary MWE run directory,
  writes a script-backed config, and runs Snakemake.
- `prepare_mwe_inputs.sh`: normalizes the external MWE data into the
  layout expected by the Snakemake config.
- `activate_local_pixi.sh` and `_local_pixi_common.sh`: optional local Pixi
  activation helpers for this checkout.
- `check_local_pixi_env.sh`: verifies the local Pixi tools when that environment
  is available.

## Pixi Scope

The Pixi environment itself is intentionally not part of the source bundle.
The ignored local holder is:

```bash
code/snakemake/dryrun/bin/
```

The default live Pixi home used on this machine is outside the repository:

```bash
../xqtl-renovated/mwe_data/.pixi
```

## MWE Data Provenance

The full external MWE data tree was downloaded from:

```bash
s3://statfungen/ftp_fgc_xqtl/xqtl_protocol_data/mwe_data/
```

It is intentionally not committed to this repository. Point `--mwe-data` at a
local copy of that tree, or at an archive containing that tree.

## Example

From the repository root:

```bash
code/snakemake/tests/run_mwe_xqtl_core.sh \
  --mwe-data ../xqtl-renovated/mwe_data \
  --run-tag xqtl_mwe_core \
  --cores 1
```

Use `--target all` only when fine-mapping plot generation is required.
