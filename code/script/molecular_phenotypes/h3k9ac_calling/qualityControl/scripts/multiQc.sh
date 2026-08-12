#!/bin/bash
# MultiQC aggregation worker (whole-cohort; aggregates fastq/alignment/macs2 QC outputs).
#
# Usage: multiQc.sh --fastq-dir <dir> --aln-dir <dir> --macs2-dir <dir> \
#          --config <multiqc_config.yaml> --outdir <dir> [--title "<report title>"]
set -euo pipefail

fastq_dir=""; aln_dir=""; macs2_dir=""; config=""; outdir=""; title="H3K9ac DLPFC ChIP-seq"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fastq-dir) fastq_dir="$2"; shift 2 ;;
    --aln-dir)   aln_dir="$2";   shift 2 ;;
    --macs2-dir) macs2_dir="$2"; shift 2 ;;
    --config)    config="$2";    shift 2 ;;
    --outdir)    outdir="$2";    shift 2 ;;
    --title)     title="$2";     shift 2 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
  esac
done
for v in fastq_dir aln_dir macs2_dir config outdir; do
  [[ -n "${!v}" ]] || { echo "ERROR: --${v//_/-} is required" >&2; exit 1; }
done

multiqc -f -i "$title" -o "$outdir" -c "$config" -v "$fastq_dir" "$aln_dir" "$macs2_dir"
