#!/bin/bash
# BAM -> FASTQ worker (Picard SamToFastq). Per-sample; the sample loop lives in the notebook.
#
# Usage: convertBamToFastq.sh --bam <input.bam> --output <output.fastq.gz>
set -euo pipefail

bam=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bam)    bam="$2";    shift 2 ;;
    --output) output="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
[[ -n "$bam"    ]] || { echo "ERROR: --bam is required"    >&2; exit 1; }
[[ -n "$output" ]] || { echo "ERROR: --output is required" >&2; exit 1; }

mkdir -p "$(dirname "$output")"
fastq="${output%.gz}"

picard -Xmx8G SamToFastq VALIDATION_STRINGENCY=LENIENT \
  INCLUDE_NON_PF_READS=true \
  INPUT="$bam" \
  VERBOSITY=DEBUG \
  CREATE_MD5_FILE=true \
  FASTQ="$fastq"

gzip -f "$fastq"
