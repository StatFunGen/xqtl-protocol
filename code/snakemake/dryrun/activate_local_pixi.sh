#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf 'Source this script instead of executing it:\n  source %s\n' "$0" >&2
    exit 1
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_local_pixi_common.sh"

xqtl_activate_local_pixi || return 1
xqtl_activate_repo_runtime_home || return 1

printf 'Activated local Pixi compatibility layer.\n'
printf 'PIXI_HOME=%s\n' "${PIXI_HOME}"
printf 'HOME=%s\n' "${HOME}"
printf 'Runtime root=%s\n' "${XQTL_LOCAL_RUNTIME_ROOT}"
printf 'Helper bin=%s\n' "${XQTL_LOCAL_PIXI_HELPER_BIN}"
printf 'PATH prefix now includes repo-local wrappers plus %s/envs/*/bin\n' "${PIXI_HOME}"
printf 'Run renovated_code/snakemake/dryrun/check_local_pixi_env.sh to verify the current state.\n'
