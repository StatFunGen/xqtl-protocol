#!/usr/bin/env bash
# ============================================================
# PCA.sh
# Mirrors: code/data_preprocessing/genotype/PCA.ipynb
#
# Steps:
#   flashpca        — compute PCs from LD-pruned unrelated samples
#   project_samples — project related samples onto unrelated PC space
#
# Usage:
#   PCA.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run PCA.ipynb <step> [--flag value ...]
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
    flashpca|project_samples)
        if [[ -n "$CONTAINER" ]]; then
            singularity exec "$CONTAINER" Rscript \
                "$SCRIPT_DIR/PCA.R" --step "$STEP" "${PASS_ARGS[@]}"
        else
            Rscript "$SCRIPT_DIR/PCA.R" --step "$STEP" "${PASS_ARGS[@]}"
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Available: flashpca, project_samples" >&2
        exit 1 ;;
esac
