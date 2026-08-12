#!/bin/bash
# BAM -> BED + scale-normalized bigWig coverage track worker.
# Per-sample; the sample loop lives in the notebook.
#
# Usage: bamToBed.sh --bam <in.bam> --sample <name> --outdir <dir>
set -euo pipefail

bam=""; sample=""; outdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bam)    bam="$2";    shift 2 ;;
    --sample) sample="$2"; shift 2 ;;
    --outdir) outdir="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
for v in bam sample outdir; do
  [[ -n "${!v}" ]] || { echo "ERROR: --$v is required" >&2; exit 1; }
done

mkdir -p "$outdir"
bed="$outdir/$sample.bed"
noDups="$outdir/${sample}_noDups.bed"
stats="$outdir/${sample}_stats.txt"
bw="$outdir/$sample.bw"

# -q 2 (map quality > 2)
samtools view -b -q 2 "$bam" | bamToBed -i stdin > "$bed"
# -q 2 (map quality > 2) and -F 0x400 (remove duplicates)
samtools view -b -F 0x400 -q 2 "$bam" | bamToBed -i stdin > "$noDups"

# Fragment count / bases / scale factor (normalize to 50M fragments x 100)
Rscript -e "f <- read.table('$noDups', sep='\t', header=FALSE); \
  l <- f[, 3] - f[, 2]; \
  o <- data.frame(Fragments=length(l), Bases=sum(as.numeric(l)), ScaleFactor=50000000 * 100 / sum(as.numeric(l))); \
  write.table(o, file='$stats', sep='\t', quote=FALSE, row.names=FALSE)"

scaleFactor=$(awk 'NR == 2 {print $3}' "$stats")

# Coverage track via deeptools bamCoverage (dedup + MAPQ>2 + the same scale factor),
# replacing the bedtools-genomecov + UCSC-wigToBigWig chain. bamCoverage reads chrom
# sizes from the BAM header, so no .fai is needed; it does require a BAM index.
[[ -f "${bam}.bai" || -f "${bam%.bam}.bai" ]] || samtools index "$bam"
bamCoverage -b "$bam" -o "$bw" \
  --ignoreDuplicates \
  --minMappingQuality 2 \
  --binSize 1 \
  --scaleFactor "$scaleFactor"

rm -f "$noDups"
