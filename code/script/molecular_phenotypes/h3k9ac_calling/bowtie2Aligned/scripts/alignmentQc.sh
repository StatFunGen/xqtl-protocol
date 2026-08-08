#!/bin/bash
# Alignment QC worker: Picard CollectMultipleMetrics + CollectRnaSeqMetrics on one BAM.
# Per-sample; the sample loop lives in the notebook.
#
# Usage: alignmentQc.sh --bam <in.bam> --reference <genome.fa> \
#          --ref-flat <refFlat> --output-prefix <prefix>
set -euo pipefail

bam=""; reference=""; ref_flat=""; prefix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bam)           bam="$2";       shift 2 ;;
    --reference)     reference="$2"; shift 2 ;;
    --ref-flat)      ref_flat="$2";  shift 2 ;;
    --output-prefix) prefix="$2";    shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
for v in bam reference ref_flat prefix; do
  [[ -n "${!v}" ]] || { echo "ERROR: --${v//_/-} is required" >&2; exit 1; }
done

mkdir -p "$(dirname "$prefix")"

picard -Xmx24G CollectMultipleMetrics \
  REFERENCE_SEQUENCE="$reference" \
  PROGRAM=CollectAlignmentSummaryMetrics \
  PROGRAM=QualityScoreDistribution \
  PROGRAM=MeanQualityByCycle \
  PROGRAM=CollectBaseDistributionByCycle \
  PROGRAM=CollectGcBiasMetrics \
  VALIDATION_STRINGENCY=SILENT \
  INPUT="$bam" \
  OUTPUT="$prefix"

picard -Xmx24G CollectRnaSeqMetrics \
  REF_FLAT="$ref_flat" \
  STRAND_SPECIFICITY=NONE \
  CHART_OUTPUT="${prefix}.rna_metrics.pdf" \
  VALIDATION_STRINGENCY=SILENT \
  INPUT="$bam" \
  OUTPUT="${prefix}.rna_metrics"
