#!/usr/bin/env bash
# ============================================================
# regtools_junctions.sh
# Extract splice junctions from a coordinate-sorted BAM using the regtools
# executable (`samtools index` + `regtools junctions extract`), mirroring the
# splicing_calling.ipynb [leafcutter_1, leafcutter_preprocessing_1] step.
#
# Per collaborator decision the pipeline uses the regtools binary; the pure-R
# port (regtools_junctions.R) is retained in the repo as an alternative
# implementation but is no longer wired into the notebook.
#
# NOTE: regtools 1.0.0's option parser stops at the first positional argument,
# so ALL options (including -o) MUST precede the BAM. It also takes the strand
# mode as the XS/RF/FR string (NOT an integer).
# ============================================================
set -euo pipefail

BAM=""
OUTPUT=""
MIN_ANCHOR=8
MIN_INTRON=50
MAX_INTRON=500000
STRANDNESS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bam)          BAM="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --min-anchor)   MIN_ANCHOR="$2"; shift 2 ;;
        --min-intron)   MIN_INTRON="$2"; shift 2 ;;
        --max-intron)   MAX_INTRON="$2"; shift 2 ;;
        --strandness)   STRANDNESS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${BAM:?--bam is required}"
: "${OUTPUT:?--output is required}"
: "${STRANDNESS:?--strandness is required (XS | RF | FR)}"

samtools index "$BAM"
regtools junctions extract \
    -a "$MIN_ANCHOR" -m "$MIN_INTRON" -M "$MAX_INTRON" -s "$STRANDNESS" \
    -o "$OUTPUT" "$BAM"
