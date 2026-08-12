#!/bin/bash
# bowtie2 alignment worker: align one FASTQ, then SAM->BAM / sort / mark-duplicates.
# Per-sample; the sample loop lives in the notebook.
#
# Usage: bowtie2Align.sh --fastq <in.fastq.gz> --sample <name> \
#          --index <bowtie2_index_prefix> --outdir <dir> [--threads N]
set -euo pipefail

fastq=""; sample=""; index=""; outdir=""; threads=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fastq)   fastq="$2";   shift 2 ;;
    --sample)  sample="$2";  shift 2 ;;
    --index)   index="$2";   shift 2 ;;
    --outdir)  outdir="$2";  shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
for v in fastq sample index outdir; do
  [[ -n "${!v}" ]] || { echo "ERROR: --$v is required" >&2; exit 1; }
done

mkdir -p "$outdir"
sam="$outdir/$sample.sam"
raw="$outdir/${sample}_raw.bam"
sorted="$outdir/${sample}_sorted.bam"
final="$outdir/$sample.bam"
date=$(date +%Y-%m-%dT%H:%M:%S:%z)

bowtie2 \
  -q --phred33 \
  --local --very-sensitive-local \
  --threads "$threads" \
  --rg "ID:${sample}" \
  --rg "SM:${sample}" \
  --rg "CN:CTCN" \
  --rg "DT:${date}" \
  -x "$index" \
  -U "$fastq" \
  -S "$sam" \
  2> "$outdir/$sample.log"

picard -Xmx48g SamFormatConverter \
  INPUT="$sam" \
  OUTPUT="$raw" \
  VALIDATION_STRINGENCY=SILENT

picard -Xmx48g SortSam \
  INPUT="$raw" \
  OUTPUT="$sorted" \
  SORT_ORDER=coordinate \
  VALIDATION_STRINGENCY=SILENT

picard -Xmx48g MarkDuplicates \
  INPUT="$sorted" \
  OUTPUT="$final" \
  VALIDATION_STRINGENCY=SILENT \
  METRICS_FILE="$outdir/$sample.duplicate_metrics" \
  REMOVE_SEQUENCING_DUPLICATES=false \
  REMOVE_DUPLICATES=false \
  ASSUME_SORTED=false \
  OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 \
  CREATE_INDEX=true \
  CREATE_MD5_FILE=true

rm -f "$sam" "$raw" "$sorted"
