#!/usr/bin/env bash
# ============================================================
# phenotype_imputation.sh
# Mirrors: code/data_preprocessing/phenotype/phenotype_imputation.ipynb
#
# Steps:
#   EBMF         — Empirical Bayes Matrix Factorization (flashier)
#   gEBMF        — Grouped EBMF (flashier with grouped data)
#   missforest   — Random forest imputation
#   missxgboost  — XGBoost-based imputation
#   knn          — K-nearest neighbors imputation
#   soft         — Soft Impute (nuclear norm minimization)
#   mean         — Mean value imputation
#   lod          — Limit of Detection imputation
#   bed_filter_na — Filter and soft-impute NA values
#
# Usage:
#   phenotype_imputation.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run phenotype_imputation.ipynb <step> [--flag value ...]
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [--container PATH] [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=""
OUTPUT=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            CONTAINER="$2"; shift 2 ;;
        --output)
            OUTPUT="${2:?--output requires a value}"
            PASS_ARGS+=("$1" "$2")
            shift 2 ;;
        --output=*)
            OUTPUT="${1#--output=}"
            PASS_ARGS+=("$1")
            shift ;;
        *)
            PASS_ARGS+=("$1"); shift ;;
    esac
done

VALID_STEPS="EBMF gEBMF missforest missxgboost knn soft mean lod bed_filter_na"

run_rscript() {
    if [[ -n "$CONTAINER" ]]; then
        singularity exec "$CONTAINER" Rscript \
            "$SCRIPT_DIR/phenotype_imputation.R" --step "$STEP" "${PASS_ARGS[@]}"
    else
        Rscript "$SCRIPT_DIR/phenotype_imputation.R" --step "$STEP" "${PASS_ARGS[@]}"
    fi
}

check_missxgboost_output() {
    [[ -n "$OUTPUT" ]] || return 0
    [[ -s "$OUTPUT" ]] || {
        echo "ERROR: missing or empty missxgboost output: $OUTPUT" >&2
        return 1
    }
    [[ -s "${OUTPUT}.tbi" ]] || {
        echo "ERROR: missing or empty missxgboost tabix index: ${OUTPUT}.tbi" >&2
        return 1
    }
    echo "output_info: $OUTPUT "
    echo "output_size: $(ls -lh "$OUTPUT" | awk '{print $5}')"
}

case "$STEP" in
    EBMF|gEBMF|missforest|missxgboost|knn|soft|mean|lod|bed_filter_na)
        run_rscript
        if [[ "$STEP" == "missxgboost" ]]; then
            check_missxgboost_output
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'" >&2
        echo "Available steps: $VALID_STEPS" >&2
        exit 1 ;;
esac
