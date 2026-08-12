#!/bin/bash
# FastQC worker — run FastQC on a single FASTQ file.
# Per-sample tool; the sample loop is driven by the calling notebook.
#
# Usage: fastqc_n726.sh --fastq <file.fastq.gz> --outdir <output_dir>
set -euo pipefail

fastq=""
outdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fastq)  fastq="$2";  shift 2 ;;
    --outdir) outdir="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
[[ -n "$fastq"  ]] || { echo "ERROR: --fastq is required"  >&2; exit 1; }
[[ -n "$outdir" ]] || { echo "ERROR: --outdir is required" >&2; exit 1; }

mkdir -p "$outdir"
fastqc -o "$outdir" -d "$outdir" --extract "$fastq"
