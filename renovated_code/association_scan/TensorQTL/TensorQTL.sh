#!/usr/bin/env bash
# ============================================================
# TensorQTL.sh
# Mirrors: code/association_scan/TensorQTL/TensorQTL.ipynb
#
# Steps:
#   cis  — cis-QTL nominal scan + permutation across all chromosomes
#   trans — trans-QTL scan
#
# Usage:
#   TensorQTL.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run TensorQTL.ipynb <step> [--flag value ...]
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
    cis|trans)
        if [[ -n "$CONTAINER" ]]; then
            singularity exec "$CONTAINER" python \
                "$SCRIPT_DIR/TensorQTL.py" --step "$STEP" "${PASS_ARGS[@]}"
        else
            python "$SCRIPT_DIR/TensorQTL.py" --step "$STEP" "${PASS_ARGS[@]}"
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Available: cis, trans" >&2
        exit 1 ;;
esac
