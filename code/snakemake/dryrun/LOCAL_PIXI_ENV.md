# Local Pixi Compatibility Note

This directory contains a repo-local compatibility layer for the current dry-run environment. It does not modify the installed Pixi tree under `../mwe_data/.pixi`; it only makes the relocated install usable from this checkout.

## Canonical local paths

- Repo root: `/sc/arion/projects/load/users/sunh14/xqtl/xqtl-renovated/xqtl-protocol`
- Live Pixi home: `/sc/arion/projects/load/users/sunh14/xqtl/xqtl-renovated/mwe_data/.pixi`
- Sibling setup repo: `/sc/arion/projects/load/users/sunh14/xqtl/xqtl-renovated/pixi-setup`

## Why this compatibility layer exists

The installed Pixi environments appear to have been built against the older prefix `/sc/arion/projects/load/users/sunh14/xqtl-renovated/...`, while this checkout lives under `/sc/arion/projects/load/users/sunh14/xqtl/xqtl-renovated/...`.

That prefix drift breaks two things:

- exposed shims under `.pixi/bin`
- Python and R console entrypoints that hard-code the old prefix in their shebang or wrapper

Examples observed on 2026-03-30:

- `envs/python/bin/sos` points at `/sc/arion/projects/load/users/sunh14/xqtl-renovated/.pixi/envs/python/bin/python3.12`
- `envs/r-base/bin/R` hard-codes `/sc/arion/projects/load/users/sunh14/xqtl-renovated/mwe_data/.pixi/envs/r-base/lib/R`

The wrappers in `dryrun/bin/` route around those stale entrypoints by:

- calling the real Python interpreter in `envs/python/bin/python`
- calling the real R executable in `envs/r-base/lib/R/bin/exec/R`
- prepending `envs/*/bin` and `envs/*/lib/jvm/bin` to `PATH`

## Pinned setup source

The local install is most consistent with the sibling `pixi-setup` repo running its `full` path:

- repo: `https://github.com/StatFunGen/pixi-setup.git`
- local commit: `a46a4b73861669ba3265b41e4f441b50edf496aa`
- package list file: `envs/full_packages.txt`

Important caveat: `pixi-setup.sh` currently downloads package lists from GitHub `main`, not from a local lockfile. This note pins the local sibling repo commit, but it does not magically lock the historical install.

## Activation and verification

From the repo root:

```bash
source renovated_code/snakemake/dryrun/activate_local_pixi.sh
renovated_code/snakemake/dryrun/check_local_pixi_env.sh
```

The activation script:

- exports `PIXI_HOME` to the live local install if it is not already set
- prepends repo-local wrappers from `dryrun/bin/`
- prepends every `envs/*/bin`
- prepends every `envs/*/lib/jvm/bin`

## Current state snapshot

Observed on 2026-03-30:

- Python: `3.12.13`
- R core executable: `4.4.3`
- bcftools: `1.23`
- PLINK: `1.9.0-b.8`
- PLINK2: `2.0.0-a.6.9LM`
- KING: `2.3.0`
- samtools: `1.23`
- STAR: `2.7.11b`
- FastQC works once the embedded JVM bin is on `PATH`
- `flashpcaR` is present in the R library
- `peer` is currently missing from the local `r-base` environment

## Scope

This is intentionally the lowest-risk fix:

- no rebuild of the installed Pixi envs
- no mutation of `../mwe_data/.pixi`
- no switch to project-style `pixi.toml`

If you later want a cleaner rebuild, do it in a separate test `PIXI_HOME` and run the 21 dry-run scripts against that test install before switching anything over.
