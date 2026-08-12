#!/bin/bash
# MACS2 broad peak-calling worker. Per-sample; the sample loop lives in the notebook.
#
# Usage: macs2.sh --chip <chip.bed> --control <control.bed> --sample <name> --outdir <dir>
set -euo pipefail

chip=""; control=""; sample=""; outdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chip)    chip="$2";    shift 2 ;;
    --control) control="$2"; shift 2 ;;
    --sample)  sample="$2";  shift 2 ;;
    --outdir)  outdir="$2";  shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
for v in chip control sample outdir; do
  [[ -n "${!v}" ]] || { echo "ERROR: --$v is required" >&2; exit 1; }
done

mkdir -p "$outdir"

macs2 callpeak -t "$chip" \
  -c "$control" \
  --format BED \
  --gsize hs \
  --keep-dup auto \
  --qvalue 0.05 \
  --broad \
  --broad-cutoff 0.1 \
  --name "$sample" \
  --outdir "$outdir"

# Peak table -> simple BED (chrom, start, end, score); drop comment/blank/header lines.
# (Header line contains "chr"; Ensembl-style peak rows -- "1".."22" -- do not.)
awk -F'\t' '$0 !~ /#/ && $0 != "" && $0 !~ /chr/ {print $1"\t"$2"\t"$3"\t"$6}' \
  "$outdir/${sample}_peaks.xls" > "$outdir/${sample}_peaks.bed"
