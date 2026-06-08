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

_cmd=(python "$SCRIPT_DIR/TensorQTL.py" --step "$STEP" "${PASS_ARGS[@]}")

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
    cis|trans)
        if [[ -n "$CONTAINER" ]]; then
            singularity exec "$CONTAINER" "${_cmd[@]}"
        else
            "${_cmd[@]}"
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Available: cis, trans" >&2
        exit 1 ;;
esac
