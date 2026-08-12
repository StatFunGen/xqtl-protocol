#!/bin/bash
# ChIP QC worker: ENCODE metrics + cross-correlation + chromatin-state overlap for one sample.
# Per-sample; the sample loop lives in the notebook. Analysis lives in lib/encodeMetrics.R.
#
# Usage: chipQc.sh --sample <name> --bed <frags.bed> --peaks <peaks.bed> \
#          --blacklist <blacklist.bed> --epigenome <chromState.bed[,chromState2.bed...]> \
#          --lib <encodeMetrics.R> --outdir <dir>
set -euo pipefail

sample=""; bed=""; peaks=""; blacklist=""; epigenome=""; lib=""; outdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample)    sample="$2";    shift 2 ;;
    --bed)       bed="$2";       shift 2 ;;
    --peaks)     peaks="$2";     shift 2 ;;
    --blacklist) blacklist="$2"; shift 2 ;;
    --epigenome) epigenome="$2"; shift 2 ;;
    --lib)       lib="$2";       shift 2 ;;
    --outdir)    outdir="$2";    shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
for v in sample bed peaks blacklist epigenome lib outdir; do
  [[ -n "${!v}" ]] || { echo "ERROR: --$v is required" >&2; exit 1; }
done
mkdir -p "$outdir"

Rscript --vanilla -e "source('$lib'); \
  metrics <- encodeMetrics(sample='$sample', bed='$bed', peaks='$peaks', blacklist='$blacklist'); \
  write.csv(metrics, quote=FALSE, row.names=FALSE, file='$outdir/${sample}_metrics.csv'); \
  crossCor <- crossCorrelation(bed='$bed', blacklist='$blacklist', rmDup=TRUE); \
  write.csv(crossCor, quote=FALSE, row.names=FALSE, file='$outdir/${sample}_crossCor.csv'); \
  epigenome <- strsplit('$epigenome', ',')[[1]]; \
  for (i in seq_along(epigenome)) { \
    chromStates <- chromatinStateOverlap(sample='$sample', bed='$bed', chromState=epigenome[i], blacklist='$blacklist', rmDup=TRUE); \
    outFile <- paste('$outdir/${sample}_', chromStates\$Epigenome[1], '_chromStates.csv', sep=''); \
    write.csv(chromStates, quote=FALSE, row.names=FALSE, file=outFile); \
  }"
