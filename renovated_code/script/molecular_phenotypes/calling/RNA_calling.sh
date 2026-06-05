#!/usr/bin/env bash
# ============================================================
# RNA_calling.sh
# Mirrors selected bash-heavy steps from RNA_calling.ipynb.
#
# Steps:
#   fastqc
#   rnaseqc_call
#   picard_qc_star_align_2
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [args...]}"
shift

CONTAINER=""
SAMPLE_LIST=""
DATA_DIR=""
CWD="output"
GTF=""
REF_FLAT=""
REFERENCE_FASTA=""
JAVA_MEM=""
NUM_THREADS=20
PAIRED_END="true"
INPUT_BAM=""
INPUT_FASTQ=""
SAMPLE_ID=""
STRAND="unstranded"
DETECTION_THRESHOLD=5
MAPPING_QUALITY=255
OPTICAL_DISTANCE=100
ZAP_RAW_BAM="false"
VAR_VCF_FILE=""
PICARD_METRICS=""
PICARD_RNA_METRICS=""
MD_BAM=""
MD_METRICS=""
BIGWIG=""
OUTPUT_SUMMARY=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)            CONTAINER="$2"; shift 2 ;;
        --sample-list)          SAMPLE_LIST="$2"; shift 2 ;;
        --data-dir)             DATA_DIR="$2"; shift 2 ;;
        --cwd)                  CWD="$2"; shift 2 ;;
        --gtf)                  GTF="$2"; shift 2 ;;
        --ref-flat)             REF_FLAT="$2"; shift 2 ;;
        --reference-fasta)      REFERENCE_FASTA="$2"; shift 2 ;;
        --java-mem)             JAVA_MEM="$2"; shift 2 ;;
        --numThreads)           NUM_THREADS="$2"; shift 2 ;;
        --paired-end)           PAIRED_END="$2"; shift 2 ;;
        --input-bam)            INPUT_BAM="$2"; shift 2 ;;
        --input-fastq)          INPUT_FASTQ="$2"; shift 2 ;;
        --sample-id)            SAMPLE_ID="$2"; shift 2 ;;
        --strand)               STRAND="$2"; shift 2 ;;
        --detection-threshold)  DETECTION_THRESHOLD="$2"; shift 2 ;;
        --mapping-quality)      MAPPING_QUALITY="$2"; shift 2 ;;
        --optical-distance)     OPTICAL_DISTANCE="$2"; shift 2 ;;
        --zap-raw-bam)          ZAP_RAW_BAM="$2"; shift 2 ;;
        --var-vcf-file)         VAR_VCF_FILE="$2"; shift 2 ;;
        --picard-metrics)       PICARD_METRICS="$2"; shift 2 ;;
        --picard-rna-metrics)   PICARD_RNA_METRICS="$2"; shift 2 ;;
        --md-bam)               MD_BAM="$2"; shift 2 ;;
        --md-metrics)           MD_METRICS="$2"; shift 2 ;;
        --bigwig)               BIGWIG="$2"; shift 2 ;;
        --output-summary)       OUTPUT_SUMMARY="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=true; shift ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

mkdir -p "$CWD"

is_true() {
    local value="${1:-}"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "True" || "$value" == "TRUE" ]]
}

has_value() {
    local value="${1:-}"
    [[ -n "$value" && "$value" != "0" && "$value" != "false" && "$value" != "False" && "$value" != "FALSE" ]]
}

strip_dot_ext() {
    local value="$1"
    local count="$2"
    local i
    for ((i = 0; i < count; i++)); do
        [[ "$value" == *.* ]] || break
        value="${value%.*}"
    done
    printf '%s' "$value"
}

picard_strand_spec() {
    case "$1" in
        rf) echo "SECOND_READ_TRANSCRIPTION_STRAND" ;;
        fr) echo "FIRST_READ_TRANSCRIPTION_STRAND" ;;
        *)  echo "NONE" ;;
    esac
}

picard_java_args() {
    if [[ -n "$JAVA_MEM" ]]; then
        printf '%s\n' "-Xmx${JAVA_MEM}"
    fi
}

_link_fastqc_outputs() {
    local fq_path="$1"
    local sample_alias="${2:-}"
    local base base_no_gz base_no_fastq actual_html actual_zip alias_base
    base="$(basename "$fq_path")"
    base_no_gz="${base%.gz}"
    base_no_fastq="${base_no_gz%.fastq}"
    base_no_fastq="${base_no_fastq%.fq}"

    actual_html="${CWD}/fastqc_raw/${base_no_fastq}_fastqc.html"
    actual_zip="${CWD}/fastqc_raw/${base_no_fastq}_fastqc.zip"

    for alias_base in "${base_no_fastq}_fastqc" "${base_no_gz}_fastqc"; do
        [[ -f "$actual_html" ]] && ln -sf "$actual_html" "${CWD}/${alias_base}.html"
        [[ -f "$actual_zip" ]] && ln -sf "$actual_zip" "${CWD}/${alias_base}.zip"
    done

    if [[ -n "$sample_alias" ]]; then
        [[ -f "$actual_html" ]] && ln -sf "$actual_html" "${CWD}/${sample_alias}_fastqc.html"
        [[ -f "$actual_zip" ]] && ln -sf "$actual_zip" "${CWD}/${sample_alias}_fastqc.zip"
    fi
}

_sample_read_rows() {
    awk -F '\t' '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            return value
        }
        function missing(value) {
            value = trim(value)
            return value == "" || value == "." || value == "NA" || value == "NaN" || value == "nan"
        }
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                header[$i] = i
            }
            id_col = ("ID" in header) ? header["ID"] : 1
            fq1_col = ("fq1" in header) ? header["fq1"] : 0
            fq2_col = ("fq2" in header) ? header["fq2"] : 0
            if (fq1_col == 0) {
                n_file_cols = 0
                for (i = 1; i <= NF; i++) {
                    if (i == id_col || $i == "strand" || $i == "read_length") {
                        continue
                    }
                    file_cols[++n_file_cols] = i
                }
                fq1_col = file_cols[1]
                fq2_col = file_cols[2]
            }
            next
        }
        $0 == "" || $1 ~ /^#/ {
            next
        }
        {
            sample = trim($(id_col))
            fq1 = trim($(fq1_col))
            fq2 = fq2_col ? trim($(fq2_col)) : ""
            if (sample == "" || missing(fq1)) {
                next
            }
            if (missing(fq2)) {
                fq2 = ""
            }
            print sample "\t" fq1 "\t" fq2
        }
    ' "$SAMPLE_LIST"
}

_fastqc() {
    [[ -z "$SAMPLE_LIST" ]] && { echo "ERROR: --sample-list is required" >&2; exit 1; }
    [[ -z "$DATA_DIR" ]] && { echo "ERROR: --data-dir is required" >&2; exit 1; }

    echo "=== FastQC ===" >&2
    mkdir -p "${CWD}/fastqc_raw"

    if [[ -n "$INPUT_FASTQ" ]]; then
        local_fq="$INPUT_FASTQ"
        if [[ "$local_fq" != /* && ! -f "$local_fq" ]]; then
            local_fq="${DATA_DIR}/${local_fq}"
        fi
        [[ -f "$local_fq" ]] || { echo "ERROR: input fastq not found: $INPUT_FASTQ" >&2; exit 1; }
        fastqc --outdir "${CWD}/fastqc_raw" --threads "$NUM_THREADS" "$local_fq"
        _link_fastqc_outputs "$local_fq"
        echo "FastQC done: $(basename "$local_fq")" >&2
        return
    fi

    while IFS=$'\t' read -r sample fq1 fq2; do
        local_fq1="${DATA_DIR}/${fq1}"
        [[ -f "$local_fq1" ]] || { echo "Skipping '$local_fq1' which didn't exist, or couldn't be read" >&2; continue; }
        fastqc --outdir "${CWD}/fastqc_raw" --threads "$NUM_THREADS" "$local_fq1"
        _link_fastqc_outputs "$local_fq1" "$sample"

        if [[ -n "$fq2" ]]; then
            local_fq2="${DATA_DIR}/${fq2}"
            if [[ -f "$local_fq2" ]]; then
                fastqc --outdir "${CWD}/fastqc_raw" --threads "$NUM_THREADS" "$local_fq2"
                _link_fastqc_outputs "$local_fq2"
            fi
        fi
        echo "FastQC done: $sample" >&2
    done < <(_sample_read_rows)
}

_move_rnaseqc_output() {
    local produced="$1"
    local target="$2"

    if [[ "$target" == *.gz ]]; then
        if [[ -f "${produced}.gz" ]]; then
            mv -f "${produced}.gz" "$target"
            return
        fi
        if [[ -f "$produced" ]]; then
            mv -f "$produced" "${target%.gz}"
            gzip -f "${target%.gz}"
            return
        fi
    else
        if [[ -f "$produced" ]]; then
            mv -f "$produced" "$target"
            return
        fi
        if [[ -f "${produced}.gz" ]]; then
            gzip -dc "${produced}.gz" > "$target"
            rm -f "${produced}.gz"
            return
        fi
    fi

    echo "ERROR: Expected RNA-SeQC output '${produced}' was not produced" >&2
    exit 1
}

_rnaseqc_from_bam() {
    [[ -z "$INPUT_BAM" ]] && { echo "ERROR: --input-bam is required" >&2; exit 1; }
    [[ -z "$SAMPLE_ID" ]] && { echo "ERROR: --sample-id is required" >&2; exit 1; }
    [[ -z "$GTF" ]] && { echo "ERROR: --gtf is required" >&2; exit 1; }

    local bam_base bam_stem out_prefix
    bam_base="$(basename "$INPUT_BAM")"
    bam_stem="${bam_base%.bam}"
    out_prefix="${CWD}/${SAMPLE_ID}.rnaseqc"

    local args=("$GTF" "$INPUT_BAM" "$CWD")
    if [[ "$STRAND" != "unstranded" ]]; then
        args+=(--stranded "$STRAND")
    fi
    args+=(--detection-threshold "$DETECTION_THRESHOLD" --mapping-quality "$MAPPING_QUALITY")
    if [[ "$PAIRED_END" == "false" || "$PAIRED_END" == "False" ]]; then
        args+=(--unpaired)
    fi

    rnaseqc "${args[@]}"

    for stem in "${bam_base}" "${bam_stem}"; do
        if [[ -f "${CWD}/${stem}.gene_tpm.gct" || -f "${CWD}/${stem}.gene_tpm.gct.gz" ]]; then
            _move_rnaseqc_output "${CWD}/${stem}.gene_tpm.gct" "${out_prefix}.gene_tpm.gct.gz"
            _move_rnaseqc_output "${CWD}/${stem}.gene_reads.gct" "${out_prefix}.gene_reads.gct.gz"
            _move_rnaseqc_output "${CWD}/${stem}.exon_reads.gct" "${out_prefix}.exon_reads.gct.gz"
            _move_rnaseqc_output "${CWD}/${stem}.metrics.tsv" "${out_prefix}.metrics.tsv"
            return
        fi
    done

    echo "ERROR: RNA-SeQC completed but expected per-BAM outputs were not found for ${bam_base}" >&2
    exit 1
}

_rnaseqc_from_fastq() {
    [[ -z "$SAMPLE_LIST" ]] && { echo "ERROR: --sample-list is required" >&2; exit 1; }
    [[ -z "$DATA_DIR" ]] && { echo "ERROR: --data-dir is required" >&2; exit 1; }
    [[ -z "$GTF" ]] && { echo "ERROR: --gtf is required" >&2; exit 1; }
    [[ -z "$REFERENCE_FASTA" ]] && { echo "ERROR: --reference-fasta is required" >&2; exit 1; }

    echo "=== STAR alignment + RNA-SeQC ===" >&2
    mkdir -p "${CWD}/alignments"
    STAR_INDEX="$(dirname "$REFERENCE_FASTA")/STAR_Index"

    while IFS=$'\t' read -r sample fq1 fq2; do
        local_fq1="${DATA_DIR}/${fq1}"
        local_fq2="${DATA_DIR}/${fq2:-}"
        local bam_dir="${CWD}/alignments/${sample}"
        mkdir -p "$bam_dir"

        local star_args=(
            --runMode alignReads
            --genomeDir "$STAR_INDEX"
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
        else
            star_args+=(--readFilesIn "$local_fq1")
        fi
        STAR "${star_args[@]}"

        local bam="${bam_dir}/${sample}.Aligned.sortedByCoord.out.bam"
        local md_bam="${bam_dir}/${sample}.md.bam"
        picard MarkDuplicates \
            INPUT="$bam" \
            OUTPUT="$md_bam" \
            METRICS_FILE="${bam_dir}/${sample}.marked_dup_metrics.txt" \
            ASSUME_SORTED=true
        samtools index "$md_bam"

        INPUT_BAM="$md_bam"
        SAMPLE_ID="$sample"
        STRAND="rf"
        _rnaseqc_from_bam
        echo "RNA-SeQC done: $sample" >&2
    done < <(_sample_read_rows)
}

_rnaseqc_call() {
    if [[ -n "$INPUT_BAM" ]]; then
        _rnaseqc_from_bam
    else
        _rnaseqc_from_fastq
    fi
}

_write_ribosomal_intervals() {
    local bam_file="$1"
    local gtf_file="$2"
    local output_file="$3"

    samtools view -H "$bam_file" > "$output_file"
    python3 - "$gtf_file" "$output_file" <<'PY'
import csv
import gzip
import sys
from pathlib import Path

gtf_path = sys.argv[1]
output_path = sys.argv[2]

def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def parse_transcript_id(attributes):
    for item in attributes.split(";"):
        item = item.strip()
        if not item:
            continue
        if item.startswith("transcript_id "):
            return item.split(" ", 1)[1].strip().strip('"')
    return None

rows = []
with open_text(gtf_path) as handle:
    for raw in handle:
        if raw.startswith("#"):
            continue
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 9 or parts[2] != "transcript":
            continue
        if "rRNA" not in raw or "transcript_id" not in parts[8]:
            continue
        transcript_id = parse_transcript_id(parts[8])
        if transcript_id is None:
            continue
        rows.append([parts[0], parts[3], parts[4], parts[6], transcript_id])

with open(output_path, "a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
PY
}

_write_bam_summary() {
    local md_base md_root4 md_root5 trans_bam sj_file
    md_base="$(basename "$MD_BAM")"
    md_root4="$(strip_dot_ext "$md_base" 4)"
    md_root5="$(strip_dot_ext "$md_base" 5)"
    trans_bam="${md_root4}.toTranscriptome.out"
    if has_value "$VAR_VCF_FILE"; then
        trans_bam="${trans_bam}_wasp_qc"
    fi
    trans_bam="${trans_bam}.bam"
    sj_file="${md_root5}.SJ.out.tab"

    {
        printf 'sample_id\tstrand\tcoord_bam_list\tBW_list\tSJ_list\ttrans_bam_list\n'
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$SAMPLE_ID" \
            "$STRAND" \
            "$(basename "$MD_BAM")" \
            "$(basename "$BIGWIG")" \
            "$sj_file" \
            "$trans_bam"
    } > "$OUTPUT_SUMMARY"
}

_picard_qc_star_align_2() {
    [[ -z "$INPUT_BAM" ]] && { echo "ERROR: --input-bam is required" >&2; exit 1; }
    [[ -z "$SAMPLE_ID" ]] && { echo "ERROR: --sample-id is required" >&2; exit 1; }
    [[ -z "$GTF" ]] && { echo "ERROR: --gtf is required" >&2; exit 1; }
    [[ -z "$REF_FLAT" ]] && { echo "ERROR: --ref-flat is required" >&2; exit 1; }
    [[ -z "$REFERENCE_FASTA" ]] && { echo "ERROR: --reference-fasta is required" >&2; exit 1; }
    [[ -z "$PICARD_METRICS" ]] && { echo "ERROR: --picard-metrics is required" >&2; exit 1; }
    [[ -z "$PICARD_RNA_METRICS" ]] && { echo "ERROR: --picard-rna-metrics is required" >&2; exit 1; }
    [[ -z "$MD_BAM" ]] && { echo "ERROR: --md-bam is required" >&2; exit 1; }
    [[ -z "$MD_METRICS" ]] && { echo "ERROR: --md-metrics is required" >&2; exit 1; }
    [[ -z "$BIGWIG" ]] && { echo "ERROR: --bigwig is required" >&2; exit 1; }
    [[ -z "$OUTPUT_SUMMARY" ]] && { echo "ERROR: --output-summary is required" >&2; exit 1; }

    mkdir -p "$(dirname "$PICARD_METRICS")" "$(dirname "$MD_BAM")" "$(dirname "$BIGWIG")"

    local java_args metrics_prefix ri_file
    metrics_prefix="${PICARD_METRICS%.*}"
    ri_file="${INPUT_BAM}.RI"
    mapfile -t java_args < <(picard_java_args)

    echo "CollectMultipleMetrics started at: $(date)" >&2
    picard "${java_args[@]}" CollectMultipleMetrics \
        -REFERENCE_SEQUENCE "$REFERENCE_FASTA" \
        -PROGRAM CollectAlignmentSummaryMetrics \
        -PROGRAM CollectInsertSizeMetrics \
        -PROGRAM QualityScoreDistribution \
        -PROGRAM MeanQualityByCycle \
        -PROGRAM CollectBaseDistributionByCycle \
        -PROGRAM CollectGcBiasMetrics \
        -VALIDATION_STRINGENCY STRICT \
        -INPUT "$INPUT_BAM" \
        -OUTPUT "$metrics_prefix"
    echo "CollectMultipleMetrics ended at: $(date)" >&2

    echo "MarkDuplicates started at: $(date)" >&2
    picard "${java_args[@]}" MarkDuplicates \
        -I "$INPUT_BAM" \
        -O "$MD_BAM" \
        -PROGRAM_RECORD_ID null \
        -M "$MD_METRICS" \
        -TMP_DIR "$CWD" \
        -MAX_RECORDS_IN_RAM 500000 \
        -SORTING_COLLECTION_SIZE_RATIO 0.25 \
        -ASSUME_SORT_ORDER coordinate \
        -TAGGING_POLICY DontTag \
        -OPTICAL_DUPLICATE_PIXEL_DISTANCE "$OPTICAL_DISTANCE" \
        -CREATE_INDEX true \
        -CREATE_MD5_FILE true \
        -VALIDATION_STRINGENCY STRICT \
        -REMOVE_SEQUENCING_DUPLICATES false \
        -REMOVE_DUPLICATES false
    samtools index "$MD_BAM"
    echo "MarkDuplicates ended at: $(date)" >&2

    _write_ribosomal_intervals "$INPUT_BAM" "$GTF" "$ri_file"

    echo "CollectRnaSeqMetrics started at: $(date)" >&2
    picard "${java_args[@]}" CollectRnaSeqMetrics \
        -REF_FLAT "$REF_FLAT" \
        -RIBOSOMAL_INTERVALS "$ri_file" \
        -STRAND_SPECIFICITY "$(picard_strand_spec "$STRAND")" \
        -CHART_OUTPUT "${PICARD_RNA_METRICS}.pdf" \
        -VALIDATION_STRINGENCY STRICT \
        -INPUT "$INPUT_BAM" \
        -OUTPUT "$PICARD_RNA_METRICS"
    echo "CollectRnaSeqMetrics ended at: $(date)" >&2

    bamCoverage -b "$MD_BAM" -o "$BIGWIG"
    _write_bam_summary

    if is_true "$ZAP_RAW_BAM"; then
        echo "WARN: zap_raw_bam requested for '$INPUT_BAM', but the modular_sos helper keeps inputs intact." >&2
    fi
}

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$SAMPLE_LIST" ]] && printf '    --sample-list %s\n' "$SAMPLE_LIST" >&2
    [[ -n "$INPUT_FASTQ" ]] && printf '    --input-fastq %s\n' "$INPUT_FASTQ" >&2
    [[ -n "$DATA_DIR" ]] && printf '    --data-dir %s\n' "$DATA_DIR" >&2
    printf '    --cwd %s\n' "$CWD" >&2
    [[ -n "$GTF" ]] && printf '    --gtf %s\n' "$GTF" >&2
    [[ -n "$REF_FLAT" ]] && printf '    --ref-flat %s\n' "$REF_FLAT" >&2
    [[ -n "$REFERENCE_FASTA" ]] && printf '    --reference-fasta %s\n' "$REFERENCE_FASTA" >&2
    [[ -n "$JAVA_MEM" ]] && printf '    --java-mem %s\n' "$JAVA_MEM" >&2
    [[ -n "$INPUT_BAM" ]] && printf '    --input-bam %s\n' "$INPUT_BAM" >&2
    [[ -n "$SAMPLE_ID" ]] && printf '    --sample-id %s\n' "$SAMPLE_ID" >&2
    printf '    --strand %s\n' "$STRAND" >&2
    printf '    --detection-threshold %s\n' "$DETECTION_THRESHOLD" >&2
    printf '    --mapping-quality %s\n' "$MAPPING_QUALITY" >&2
    printf '    --paired-end %s\n' "$PAIRED_END" >&2
    exit 0
fi

_dispatch() {
    case "$STEP" in
        fastqc) _fastqc ;;
        rnaseqc_call) _rnaseqc_call ;;
        picard_qc_star_align_2) _picard_qc_star_align_2 ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: fastqc, rnaseqc_call, picard_qc_star_align_2" >&2
            exit 1
            ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    singularity exec "$CONTAINER" bash -s <<EOF
$(declare -f is_true has_value strip_dot_ext picard_strand_spec picard_java_args _link_fastqc_outputs _sample_read_rows _fastqc _move_rnaseqc_output _rnaseqc_from_bam _rnaseqc_from_fastq _rnaseqc_call _write_ribosomal_intervals _write_bam_summary _picard_qc_star_align_2 _dispatch)
STEP="$STEP"
SAMPLE_LIST="$SAMPLE_LIST"
DATA_DIR="$DATA_DIR"
CWD="$CWD"
GTF="$GTF"
REF_FLAT="$REF_FLAT"
REFERENCE_FASTA="$REFERENCE_FASTA"
JAVA_MEM="$JAVA_MEM"
NUM_THREADS="$NUM_THREADS"
PAIRED_END="$PAIRED_END"
INPUT_BAM="$INPUT_BAM"
INPUT_FASTQ="$INPUT_FASTQ"
SAMPLE_ID="$SAMPLE_ID"
STRAND="$STRAND"
DETECTION_THRESHOLD="$DETECTION_THRESHOLD"
MAPPING_QUALITY="$MAPPING_QUALITY"
OPTICAL_DISTANCE="$OPTICAL_DISTANCE"
ZAP_RAW_BAM="$ZAP_RAW_BAM"
VAR_VCF_FILE="$VAR_VCF_FILE"
PICARD_METRICS="$PICARD_METRICS"
PICARD_RNA_METRICS="$PICARD_RNA_METRICS"
MD_BAM="$MD_BAM"
MD_METRICS="$MD_METRICS"
BIGWIG="$BIGWIG"
OUTPUT_SUMMARY="$OUTPUT_SUMMARY"
_dispatch
EOF
else
    _dispatch
fi
