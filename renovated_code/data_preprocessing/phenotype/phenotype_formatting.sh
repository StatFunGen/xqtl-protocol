#!/usr/bin/env bash
# ============================================================
# phenotype_formatting.sh
# Mirrors: code/data_preprocessing/phenotype/phenotype_formatting.ipynb
#
# Steps:
#   phenotype_by_chrom   — split BED.gz into per-chromosome files
#   phenotype_by_region  — extract phenotypes by region list
#   gct_extract_samples  — filter samples from a GCT file
#
# Usage:
#   phenotype_formatting.sh <step> [--container PATH] [--flag value ...]
#
# The first positional argument is the step name.
# --container PATH  Run inside singularity exec PATH (optional).
# --dry-run         Print the full command that would run; do not execute.
# All other flags are forwarded to the underlying R/Python script.
#
# Interface is kept identical to:
#   sos run phenotype_formatting.ipynb <step> [--flag value ...]
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

_run() {
    local lang="$1"; shift
    local script="$1"; shift
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $(basename "$0") $STEP" >&2
        [[ -n "$CONTAINER" ]] && printf '[DRY-RUN] Container: %s\n' "$CONTAINER" >&2
        echo "[DRY-RUN] Full command (copy-paste to debug):" >&2
        if [[ -n "$CONTAINER" ]]; then
            printf '  singularity exec %s \\\n' "$CONTAINER" >&2
            printf '    %s %s \\\n' "$lang" "$script" >&2
            printf '    %s \\\n' "${PASS_ARGS[@]}" >&2
            echo "    --dry-run" >&2
        else
            printf '  %s %s \\\n' "$lang" "$script" >&2
            printf '    %s \\\n' "${PASS_ARGS[@]}" >&2
            echo "  --dry-run" >&2
        fi
        return 0
    fi
    if [[ -n "$CONTAINER" ]]; then
        singularity exec "$CONTAINER" "$lang" "$script" "${PASS_ARGS[@]}"
    else
        "$lang" "$script" "${PASS_ARGS[@]}"
    fi
}

case "$STEP" in
    phenotype_by_chrom)
        _run python "$SCRIPT_DIR/phenotype_formatting.py" ;;
    phenotype_by_region)
        _run python "$SCRIPT_DIR/phenotype_formatting.py" ;;
    gct_extract_samples)
        _run Rscript "$SCRIPT_DIR/phenotype_formatting.R" ;;
    *)
        echo "ERROR: Unknown step '$STEP'" >&2
        echo "Available steps: phenotype_by_chrom, phenotype_by_region, gct_extract_samples" >&2
        exit 1 ;;
esac
