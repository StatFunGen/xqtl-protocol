#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_mwe_xqtl_core.sh [options]

Options:
  --mwe-data PATH          MWE data directory, .tar.gz/.tgz, or zip file. Defaults to xqtl-renovated/mwe_data.
  --run-tag TAG           Run label under renovated_code/snakemake/tmp/modular_sos_mwe.
  --cores N               Snakemake cores. Default: 1.
  --target TARGET         Snakemake target. Default: xqtl_core.
  --snakemake-dry-run     Build the DAG without executing jobs.
  --no-pixi               Do not source the local pixi compatibility layer.
  -h, --help              Show this help.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/../../dryrun/run_modular_sos_mwe_snakemake.sh"

CORES=1
TARGET="xqtl_core"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cores)
            [[ $# -ge 2 ]] || { printf 'error: --cores requires a value\n' >&2; exit 1; }
            CORES="$2"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || { printf 'error: --target requires a value\n' >&2; exit 1; }
            TARGET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

exec "${RUNNER}" --cores "${CORES}" --target "${TARGET}" "${ARGS[@]}"
