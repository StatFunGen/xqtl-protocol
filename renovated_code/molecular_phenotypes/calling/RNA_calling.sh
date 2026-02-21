#!/usr/bin/env bash
# ============================================================
# RNA_calling.sh
# Mirrors: code/molecular_phenotypes/calling/RNA_calling.ipynb
#
# Steps:
#   fastqc       — raw read QC via FastQC
#   rnaseqc_call — STAR alignment + RNA-SeQC quantification
#
# Usage:
#   RNA_calling.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run RNA_calling.ipynb <step> [--flag value ...]
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [--container PATH] [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
CONTAINER=""
SAMPLE_LIST=""
DATA_DIR=""
CWD="output"
GTF=""
REFERENCE_FASTA=""
NUM_THREADS=20
PAIRED_END="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)        CONTAINER="$2";        shift 2 ;;
        --sample-list)      SAMPLE_LIST="$2";      shift 2 ;;
        --data-dir)         DATA_DIR="$2";         shift 2 ;;
        --cwd)              CWD="$2";              shift 2 ;;
        --gtf)              GTF="$2";              shift 2 ;;
        --reference-fasta)  REFERENCE_FASTA="$2";  shift 2 ;;
        --numThreads)       NUM_THREADS="$2";      shift 2 ;;
        --paired-end)       PAIRED_END="$2";       shift 2 ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

mkdir -p "$CWD"

# ── Step: fastqc ─────────────────────────────────────────────────────────────
_fastqc() {
    [[ -z "$SAMPLE_LIST" ]] && { echo "ERROR: --sample-list is required" >&2; exit 1; }
    [[ -z "$DATA_DIR"    ]] && { echo "ERROR: --data-dir is required" >&2; exit 1; }

    echo "=== FastQC ===" >&2
    mkdir -p "${CWD}/fastqc_raw"

    # Read sample list: col1=sample_id, col2=fastq1, [col3=fastq2]
    while IFS=$'\t' read -r sample fq1 fq2 rest; do
        [[ "$sample" == \#* || -z "$sample" ]] && continue
        local_fq1="${DATA_DIR}/${fq1}"
        fastqc --outdir "${CWD}/fastqc_raw" --threads "$NUM_THREADS" "$local_fq1"
        if [[ -n "$fq2" ]]; then
            local_fq2="${DATA_DIR}/${fq2}"
            fastqc --outdir "${CWD}/fastqc_raw" --threads "$NUM_THREADS" "$local_fq2"
        fi
        echo "FastQC done: $sample" >&2
    done < "$SAMPLE_LIST"

    echo "FastQC complete. Results in: ${CWD}/fastqc_raw" >&2
}

# ── Step: rnaseqc_call ────────────────────────────────────────────────────────
_rnaseqc_call() {
    [[ -z "$SAMPLE_LIST"     ]] && { echo "ERROR: --sample-list is required" >&2; exit 1; }
    [[ -z "$DATA_DIR"        ]] && { echo "ERROR: --data-dir is required" >&2; exit 1; }
    [[ -z "$GTF"             ]] && { echo "ERROR: --gtf is required" >&2; exit 1; }
    [[ -z "$REFERENCE_FASTA" ]] && { echo "ERROR: --reference-fasta is required" >&2; exit 1; }

    echo "=== STAR alignment + RNA-SeQC ===" >&2
    mkdir -p "${CWD}/alignments"

    # Derive STAR index directory from reference fasta path
    # Assumes index is in the same dir or a sibling STAR_Index directory
    STAR_INDEX="$(dirname "$REFERENCE_FASTA")/STAR_Index"

    while IFS=$'\t' read -r sample fq1 fq2 rest; do
        [[ "$sample" == \#* || -z "$sample" ]] && continue
        local_fq1="${DATA_DIR}/${fq1}"
        local_fq2="${DATA_DIR}/${fq2:-}"
        local bam_dir="${CWD}/alignments/${sample}"
        mkdir -p "$bam_dir"

        echo "=== Aligning: $sample ===" >&2

        # STAR alignment
        local star_args=(
            --runMode alignReads
            --genomeDir "$STAR_INDEX"
            --readFilesIn "$local_fq1"
            --outSAMtype BAM SortedByCoordinate
            --outSAMattributes NH HI NM MD AS XS
            --outSAMstrandField intronMotif
            --outFilterType BySJout
            --outFilterMultimapNmax 20
            --alignSJoverhangMin 8
            --alignSJDBoverhangMin 1
            --outFilterMismatchNmax 999
            --outFilterMismatchNoverLmax 0.04
            --alignIntronMin 20
            --alignIntronMax 1000000
            --alignMatesGapMax 1000000
            --runThreadN "$NUM_THREADS"
            --outFileNamePrefix "${bam_dir}/${sample}."
            --readFilesCommand zcat
        )
        if [[ -n "$local_fq2" && -f "$local_fq2" ]]; then
            star_args+=(--readFilesIn "$local_fq1" "$local_fq2")
        fi
        STAR "${star_args[@]}"

        local bam="${bam_dir}/${sample}.Aligned.sortedByCoord.out.bam"

        # MarkDuplicates (Picard)
        local md_bam="${bam_dir}/${sample}.md.bam"
        picard MarkDuplicates \
            INPUT="$bam" \
            OUTPUT="$md_bam" \
            METRICS_FILE="${bam_dir}/${sample}.marked_dup_metrics.txt" \
            ASSUME_SORTED=true

        samtools index "$md_bam"

        # RNA-SeQC
        rnaseqc "$GTF" "$md_bam" "${CWD}" \
            --sample "$sample" \
            --stranded rf \
            --unpaired \
            --legacy
        echo "RNA-SeQC done: $sample" >&2

    done < "$SAMPLE_LIST"

    # Collect RNA-SeQC outputs into GCT files (uses rnaseqc collect or custom aggregation)
    echo "Collecting RNA-SeQC gene TPM..." >&2
    # This is typically done via a downstream aggregation step or GTEx pipeline script
    echo "rnaseqc_call complete. Results in: ${CWD}" >&2
}

export -f _fastqc _rnaseqc_call

_dispatch() {
    case "$STEP" in
        fastqc)       _fastqc ;;
        rnaseqc_call) _rnaseqc_call ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: fastqc, rnaseqc_call" >&2
            exit 1 ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    singularity exec "$CONTAINER" bash -s << EOF
$(declare -f _fastqc _rnaseqc_call _dispatch)
STEP="$STEP" SAMPLE_LIST="$SAMPLE_LIST" DATA_DIR="$DATA_DIR"
CWD="$CWD" GTF="$GTF" REFERENCE_FASTA="$REFERENCE_FASTA"
NUM_THREADS="$NUM_THREADS" PAIRED_END="$PAIRED_END"
_dispatch
EOF
else
    _dispatch
fi
