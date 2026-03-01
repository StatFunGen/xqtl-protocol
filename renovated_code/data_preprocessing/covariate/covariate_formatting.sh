#!/usr/bin/env bash
# ============================================================
# covariate_formatting.sh
# Mirrors: code/data_preprocessing/covariate/covariate_formatting.ipynb
#
# Steps:
#   merge_genotype_pc — merge projected genotype PCs with fixed covariates
#
# Usage:
#   covariate_formatting.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run covariate_formatting.ipynb <step> [--flag value ...]
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [--container PATH] [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=""
DRY_RUN=false
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            CONTAINER="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        *)
            PASS_ARGS+=("$1"); shift ;;
    esac
done

_cmd=(Rscript "$SCRIPT_DIR/covariate_formatting.R" --step "$STEP" "${PASS_ARGS[@]}")

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$CONTAINER" ]] && printf '[DRY-RUN] Container: %s
' "$CONTAINER" >&2
    echo "[DRY-RUN] Full command (copy-paste to debug):" >&2
    if [[ -n "$CONTAINER" ]]; then
        printf '  singularity exec %s \
' "$CONTAINER" >&2
        printf '    %s \
' "${_cmd[@]}" >&2
        echo "    --dry-run" >&2
    else
        printf '  %s \
' "${_cmd[@]}" >&2
        echo "  --dry-run" >&2
    fi
    exit 0
fi

case "$STEP" in
    merge_genotype_pc)
        if [[ -n "$CONTAINER" ]]; then
            singularity exec "$CONTAINER" "${_cmd[@]}"
        else
            "${_cmd[@]}"
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Available: merge_genotype_pc" >&2
        exit 1 ;;
esac
