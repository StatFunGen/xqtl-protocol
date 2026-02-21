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
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            CONTAINER="$2"; shift 2 ;;
        *)
            PASS_ARGS+=("$1"); shift ;;
    esac
done

case "$STEP" in
    merge_genotype_pc)
        if [[ -n "$CONTAINER" ]]; then
            singularity exec "$CONTAINER" Rscript \
                "$SCRIPT_DIR/covariate_formatting.R" --step "$STEP" "${PASS_ARGS[@]}"
        else
            Rscript "$SCRIPT_DIR/covariate_formatting.R" --step "$STEP" "${PASS_ARGS[@]}"
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Available: merge_genotype_pc" >&2
        exit 1 ;;
esac
