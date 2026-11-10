#!/usr/bin/env bash
# ============================================================
# torino_coverage.sh
# Per-base read coverage of one BAM as bgzipped, tabix-indexed bedGraph, for the
# torino_1 step of isoform_calling_torino.ipynb.
#
# Coverage is `bedtools genomecov -bga -split`: spliced reads add depth to the
# exons they cover and none to the intron they skip, and every base of every
# reference sequence is reported, zero-depth runs included. -bga rather than -bg
# lets torino_count_matrix.R tell a zero-coverage stretch from a region absent
# from the file.
#
# --strandness is the manifest value:
#   strand_missing  one file, <prefix>.coverage.bedgraph.gz, all reads.
#   rf | fr         two files, <prefix>.plus.coverage.bedgraph.gz and
#                   <prefix>.minus.coverage.bedgraph.gz, holding the coverage of
#                   transcripts on that genomic strand. rf is a dUTP library (first
#                   read antisense to the transcript), fr the opposite. -du flips
#                   the second mate so both mates of a pair count on one strand.
#
# --min-mapq N (default 0, off) keeps only reads with mapping quality >= N; STAR
# gives uniquely mapped reads 255. Secondary and supplementary alignments are
# always dropped so each read is counted once.
# ============================================================
set -euo pipefail

BAM=""
PREFIX=""
STRANDNESS=""
MIN_MAPQ=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bam)          BAM="$2"; shift 2 ;;
        --output-prefix) PREFIX="$2"; shift 2 ;;
        --strandness)   STRANDNESS="$2"; shift 2 ;;
        --min-mapq)     MIN_MAPQ="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${BAM:?--bam is required}"
: "${PREFIX:?--output-prefix is required}"
: "${STRANDNESS:?--strandness is required (strand_missing | rf | fr)}"
case "$MIN_MAPQ" in ""|*[!0-9]*) echo "--min-mapq must be a whole number, got '$MIN_MAPQ'" >&2; exit 1 ;; esac
case "$STRANDNESS" in
    strand_missing|rf|fr) ;;
    *) echo "--strandness must be strand_missing, rf or fr, got '$STRANDNESS'" >&2; exit 1 ;;
esac
[ -s "$BAM" ] || { echo "BAM not found or empty: $BAM" >&2; exit 1; }

mkdir -p "$(dirname "$PREFIX")"
FILTERED="${PREFIX}.filtered.bam"
trap 'rm -f "$FILTERED"' EXIT
# 0x100 secondary, 0x800 supplementary, 0x4 unmapped
samtools view -b -F 0x904 -q "$MIN_MAPQ" -o "$FILTERED" "$BAM"

coverage() {   # coverage <output.bedgraph.gz> [bedtools strand options]
    local out="$1"; shift
    bedtools genomecov -bga -split "$@" -ibam "$FILTERED" | bgzip -c > "$out"
    tabix -f -p bed "$out"
}

if [ "$STRANDNESS" = strand_missing ]; then
    coverage "${PREFIX}.coverage.bedgraph.gz"
else
    # bedtools -strand selects reads by their own strand; with -du that is the first
    # mate's strand for a pair. For rf the first read is antisense, so plus-strand
    # transcripts are read from minus-strand alignments.
    if [ "$STRANDNESS" = rf ]; then plus_reads="-"; minus_reads="+"; else plus_reads="+"; minus_reads="-"; fi
    coverage "${PREFIX}.plus.coverage.bedgraph.gz" -strand "$plus_reads" -du
    coverage "${PREFIX}.minus.coverage.bedgraph.gz" -strand "$minus_reads" -du
fi
