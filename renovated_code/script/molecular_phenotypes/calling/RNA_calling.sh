#!/usr/bin/env bash
# ============================================================
# RNA_calling.sh
# Mirrors selected bash-heavy steps from RNA_calling.ipynb.
#
# Steps:
#   bam_to_fastq
#   fastqc
#   star_align_1
#   filter_reads
#   rnaseqc_call
#   picard_qc_star_align_2
#   multiqc_report
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [args...]}"
shift

CONTAINER=""
SAMPLE_LIST=""
DATA_DIR=""
INPUT_DIR=""
CWD="output"
GTF=""
REF_FLAT=""
REFERENCE_FASTA=""
STAR_INDEX=""
JAVA_MEM=""
NUM_THREADS=20
PAIRED_END="true"
INPUT_BAM=""
INPUT_READS=""
INPUT_CORD_BAM=""
INPUT_TRANS_BAM=""
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
SORTED_BAM=""
OUTPUT_FASTQ1=""
OUTPUT_FASTQ2=""
OUTPUT_PREFIX=""
OUTPUT_CORD_BAM=""
OUTPUT_TRANS_BAM=""
MD_BAM=""
MD_METRICS=""
BIGWIG=""
OUTPUT_SUMMARY=""
OUTPUT_REPORT=""
MULTIQC_CONFIG=""
SJDB_OVERHANG=100
READ_FILES_COMMAND="zcat"
OUT_FILTER_MULTIMAP_NMAX=20
ALIGN_SJ_OVERHANG_MIN=8
ALIGN_SJDB_OVERHANG_MIN=1
OUT_FILTER_MISMATCH_NMAX=999
OUT_FILTER_MISMATCH_NOVER_LMAX=0.1
ALIGN_INTRON_MIN=20
ALIGN_INTRON_MAX=1000000
ALIGN_MATES_GAP_MAX=1000000
OUT_FILTER_TYPE="BySJout"
OUT_FILTER_SCORE_MIN_OVER_LREAD=0.33
OUT_FILTER_MATCH_NMIN_OVER_LREAD=0.33
LIMIT_SJDB_INSERT_NSJ=1200000
OUT_SAM_STRAND_FIELD="intronMotif"
OUT_FILTER_INTRON_MOTIFS="None"
ALIGN_SOFT_CLIP_AT_REFERENCE_ENDS="Yes"
QUANT_MODE="TranscriptomeSAM GeneCounts"
OUT_SAM_ATTR_RG_LINE="ID:rg1 SM:sm1"
OUT_SAM_ATTRIBUTES="NH HI AS nM NM ch"
CHIM_SEGMENT_MIN=15
CHIM_JUNCTION_OVERHANG_MIN=15
CHIM_OUT_TYPE="Junctions WithinBAM SoftClip"
CHIM_MAIN_SEGMENT_MULT_NMAX=1
UNIQUE=""
WASP=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)            CONTAINER="$2"; shift 2 ;;
        --sample-list)          SAMPLE_LIST="$2"; shift 2 ;;
        --data-dir)             DATA_DIR="$2"; shift 2 ;;
        --input-dir)            INPUT_DIR="$2"; shift 2 ;;
        --cwd)                  CWD="$2"; shift 2 ;;
        --gtf)                  GTF="$2"; shift 2 ;;
        --ref-flat)             REF_FLAT="$2"; shift 2 ;;
        --reference-fasta)      REFERENCE_FASTA="$2"; shift 2 ;;
        --STAR-index)           STAR_INDEX="$2"; shift 2 ;;
        --java-mem)             JAVA_MEM="$2"; shift 2 ;;
        --numThreads)           NUM_THREADS="$2"; shift 2 ;;
        --paired-end)           PAIRED_END="$2"; shift 2 ;;
        --input-bam)            INPUT_BAM="$2"; shift 2 ;;
        --input-reads)          INPUT_READS="$2"; shift 2 ;;
        --input-cord-bam)       INPUT_CORD_BAM="$2"; shift 2 ;;
        --input-trans-bam)      INPUT_TRANS_BAM="$2"; shift 2 ;;
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
        --sorted-bam)           SORTED_BAM="$2"; shift 2 ;;
        --output-fastq1)        OUTPUT_FASTQ1="$2"; shift 2 ;;
        --output-fastq2)        OUTPUT_FASTQ2="$2"; shift 2 ;;
        --output-prefix)        OUTPUT_PREFIX="$2"; shift 2 ;;
        --output-cord-bam)      OUTPUT_CORD_BAM="$2"; shift 2 ;;
        --output-trans-bam)     OUTPUT_TRANS_BAM="$2"; shift 2 ;;
        --md-bam)               MD_BAM="$2"; shift 2 ;;
        --md-metrics)           MD_METRICS="$2"; shift 2 ;;
        --bigwig)               BIGWIG="$2"; shift 2 ;;
        --output-summary)       OUTPUT_SUMMARY="$2"; shift 2 ;;
        --output-report)        OUTPUT_REPORT="$2"; shift 2 ;;
        --multiqc-config)       MULTIQC_CONFIG="$2"; shift 2 ;;
        --sjdbOverhang)         SJDB_OVERHANG="$2"; shift 2 ;;
        --read-files-command)   READ_FILES_COMMAND="$2"; shift 2 ;;
        --outFilterMultimapNmax) OUT_FILTER_MULTIMAP_NMAX="$2"; shift 2 ;;
        --alignSJoverhangMin)   ALIGN_SJ_OVERHANG_MIN="$2"; shift 2 ;;
        --alignSJDBoverhangMin) ALIGN_SJDB_OVERHANG_MIN="$2"; shift 2 ;;
        --outFilterMismatchNmax) OUT_FILTER_MISMATCH_NMAX="$2"; shift 2 ;;
        --outFilterMismatchNoverLmax) OUT_FILTER_MISMATCH_NOVER_LMAX="$2"; shift 2 ;;
        --alignIntronMin)       ALIGN_INTRON_MIN="$2"; shift 2 ;;
        --alignIntronMax)       ALIGN_INTRON_MAX="$2"; shift 2 ;;
        --alignMatesGapMax)     ALIGN_MATES_GAP_MAX="$2"; shift 2 ;;
        --outFilterType)        OUT_FILTER_TYPE="$2"; shift 2 ;;
        --outFilterScoreMinOverLread) OUT_FILTER_SCORE_MIN_OVER_LREAD="$2"; shift 2 ;;
        --outFilterMatchNminOverLread) OUT_FILTER_MATCH_NMIN_OVER_LREAD="$2"; shift 2 ;;
        --limitSjdbInsertNsj)   LIMIT_SJDB_INSERT_NSJ="$2"; shift 2 ;;
        --outSAMstrandField)    OUT_SAM_STRAND_FIELD="$2"; shift 2 ;;
        --outFilterIntronMotifs) OUT_FILTER_INTRON_MOTIFS="$2"; shift 2 ;;
        --alignSoftClipAtReferenceEnds) ALIGN_SOFT_CLIP_AT_REFERENCE_ENDS="$2"; shift 2 ;;
        --quantMode)            QUANT_MODE="$2"; shift 2 ;;
        --outSAMattrRGline)     OUT_SAM_ATTR_RG_LINE="$2"; shift 2 ;;
        --outSAMattributes)     OUT_SAM_ATTRIBUTES="$2"; shift 2 ;;
        --chimSegmentMin)       CHIM_SEGMENT_MIN="$2"; shift 2 ;;
        --chimJunctionOverhangMin) CHIM_JUNCTION_OVERHANG_MIN="$2"; shift 2 ;;
        --chimOutType)          CHIM_OUT_TYPE="$2"; shift 2 ;;
        --chimMainSegmentMultNmax) CHIM_MAIN_SEGMENT_MULT_NMAX="$2"; shift 2 ;;
        --unique)               UNIQUE="$2"; shift 2 ;;
        --wasp)                 WASP="$2"; shift 2 ;;
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

_bam_to_fastq() {
    [[ -z "$INPUT_BAM" ]] && { echo "ERROR: --input-bam is required" >&2; exit 1; }
    [[ -z "$SORTED_BAM" ]] && { echo "ERROR: --sorted-bam is required" >&2; exit 1; }
    [[ -z "$OUTPUT_FASTQ1" ]] && { echo "ERROR: --output-fastq1 is required" >&2; exit 1; }
    [[ -z "$OUTPUT_FASTQ2" ]] && { echo "ERROR: --output-fastq2 is required" >&2; exit 1; }

    mkdir -p "$(dirname "$SORTED_BAM")" "$(dirname "$OUTPUT_FASTQ1")" "$(dirname "$OUTPUT_FASTQ2")"
    samtools sort -n "$INPUT_BAM" -o "$SORTED_BAM"
    bedtools bamtofastq -i "$SORTED_BAM" -fq "$OUTPUT_FASTQ1" -fq2 "$OUTPUT_FASTQ2"
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

_star_align_1() {
    [[ -z "$GTF" ]] && { echo "ERROR: --gtf is required" >&2; exit 1; }
    [[ -z "$STAR_INDEX" ]] && { echo "ERROR: --STAR-index is required" >&2; exit 1; }
    [[ -z "$INPUT_READS" ]] && { echo "ERROR: --input-reads is required" >&2; exit 1; }
    [[ -z "$OUTPUT_PREFIX" ]] && { echo "ERROR: --output-prefix is required" >&2; exit 1; }
    [[ -z "$OUTPUT_CORD_BAM" ]] && { echo "ERROR: --output-cord-bam is required" >&2; exit 1; }
    [[ -z "$OUTPUT_TRANS_BAM" ]] && { echo "ERROR: --output-trans-bam is required" >&2; exit 1; }

    mkdir -p "$(dirname "$OUTPUT_CORD_BAM")" "$(dirname "$OUTPUT_TRANS_BAM")"

    local -a input_reads quant_mode_args out_sam_attr_rgline_args out_sam_attributes_args chim_out_type_args var_args
    read -r -a input_reads <<< "$INPUT_READS"
    read -r -a quant_mode_args <<< "$QUANT_MODE"
    read -r -a out_sam_attr_rgline_args <<< "$OUT_SAM_ATTR_RG_LINE"
    read -r -a out_sam_attributes_args <<< "$OUT_SAM_ATTRIBUTES"
    read -r -a chim_out_type_args <<< "$CHIM_OUT_TYPE"
    var_args=()
    if has_value "$VAR_VCF_FILE"; then
        out_sam_attributes_args+=(vW)
        var_args=(--varVCFfile "$VAR_VCF_FILE" --waspOutputMode SAMtag)
    fi

    rm -rf "${CWD}/${SAMPLE_ID}".*.out.*.gz
    rm -rf "${CWD}/${SAMPLE_ID}._STARpass1"

    local align_start_time align_end_time align_runtime sort_start_time sort_end_time sort_runtime produced_trans
    align_start_time=$(date +%s)

    STAR --runMode alignReads \
        --runThreadN "$NUM_THREADS" \
        --genomeDir "$STAR_INDEX" \
        --readFilesIn "${input_reads[@]}" \
        --readFilesCommand "$READ_FILES_COMMAND" \
        --outFileNamePrefix "${OUTPUT_PREFIX}." \
        --outSAMstrandField "$OUT_SAM_STRAND_FIELD" \
        --twopassMode Basic \
        --outFilterMultimapNmax "$OUT_FILTER_MULTIMAP_NMAX" \
        --alignSJoverhangMin "$ALIGN_SJ_OVERHANG_MIN" \
        --alignSJDBoverhangMin "$ALIGN_SJDB_OVERHANG_MIN" \
        --outFilterMismatchNmax "$OUT_FILTER_MISMATCH_NMAX" \
        --outFilterMismatchNoverLmax "$OUT_FILTER_MISMATCH_NOVER_LMAX" \
        --alignIntronMin "$ALIGN_INTRON_MIN" \
        --alignIntronMax "$ALIGN_INTRON_MAX" \
        --alignMatesGapMax "$ALIGN_MATES_GAP_MAX" \
        --outFilterType "$OUT_FILTER_TYPE" \
        --outFilterScoreMinOverLread "$OUT_FILTER_SCORE_MIN_OVER_LREAD" \
        --outFilterMatchNminOverLread "$OUT_FILTER_MATCH_NMIN_OVER_LREAD" \
        --limitSjdbInsertNsj "$LIMIT_SJDB_INSERT_NSJ" \
        --outFilterIntronMotifs "$OUT_FILTER_INTRON_MOTIFS" \
        --alignSoftClipAtReferenceEnds "$ALIGN_SOFT_CLIP_AT_REFERENCE_ENDS" \
        --quantMode "${quant_mode_args[@]}" \
        --outSAMtype BAM Unsorted \
        --outSAMunmapped Within \
        --genomeLoad NoSharedMemory \
        --chimSegmentMin "$CHIM_SEGMENT_MIN" \
        --chimJunctionOverhangMin "$CHIM_JUNCTION_OVERHANG_MIN" \
        --chimOutType "${chim_out_type_args[@]}" \
        --chimMainSegmentMultNmax "$CHIM_MAIN_SEGMENT_MULT_NMAX" \
        --chimOutJunctionFormat 0 \
        --outSAMattributes "${out_sam_attributes_args[@]}" \
        --outSAMattrRGline "${out_sam_attr_rgline_args[@]}" \
        --sjdbOverhang "$SJDB_OVERHANG" \
        --sjdbGTFfile "$GTF" "${var_args[@]}"

    align_end_time=$(date +%s)
    align_runtime=$((align_end_time - align_start_time))

    rm -r "${OUTPUT_PREFIX}._STARgenome"
    if is_true "$PAIRED_END"; then
        rm -rf "${OUTPUT_PREFIX}._STARtmp"
    fi

    sort_start_time=$(date +%s)
    samtools sort --threads "$NUM_THREADS" -o "$OUTPUT_CORD_BAM" "${OUTPUT_PREFIX}.Aligned.out.bam"
    rm "${OUTPUT_PREFIX}.Aligned.out.bam"
    sort_end_time=$(date +%s)
    sort_runtime=$((sort_end_time - sort_start_time))

    samtools index "$OUTPUT_CORD_BAM"

    produced_trans="${OUTPUT_PREFIX}.Aligned.toTranscriptome.out.bam"
    if [[ "$produced_trans" != "$OUTPUT_TRANS_BAM" && -f "$produced_trans" ]]; then
        mv -f "$produced_trans" "$OUTPUT_TRANS_BAM"
    fi

    echo "STAR alignment started at: $(date -d "@$align_start_time")"
    echo "STAR alignment ended at: $(date -d "@$align_end_time")"
    echo "STAR alignment runtime: $align_runtime seconds"
    echo ""
    echo "Sorting started at: $(date -d "@$sort_start_time")"
    echo "Sorting ended at: $(date -d "@$sort_end_time")"
    echo "Sorting runtime: $sort_runtime seconds"
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
    if ! is_true "$PAIRED_END"; then
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

_filter_one_bam() {
    local input_file="$1"
    local output_file="$2"
    local base_name derived_file
    base_name="${input_file%.*}"
    derived_file="${base_name%.*}.int.bam"

    if has_value "$UNIQUE"; then
        if has_value "$WASP"; then
            echo "Applying unique_WASP filtering"
            samtools view -h -q "$MAPPING_QUALITY" "$input_file" | awk '!/vW:i:[2-7]/' | samtools view -b > "$derived_file"
        else
            echo "Applying unique filtering"
            samtools view -h -q "$MAPPING_QUALITY" "$input_file" | samtools view -b > "$derived_file"
        fi
    else
        if has_value "$WASP"; then
            echo "Applying WASP filtering"
            samtools view -h "$input_file" | awk '!/vW:i:[2-7]/' | samtools view -b > "$derived_file"
        else
            echo "Applying no filtering"
            cp "$input_file" "$derived_file"
        fi
    fi
    mv -f "$derived_file" "$output_file"
}

_filter_reads() {
    [[ -z "$INPUT_CORD_BAM" ]] && { echo "ERROR: --input-cord-bam is required" >&2; exit 1; }
    [[ -z "$INPUT_TRANS_BAM" ]] && { echo "ERROR: --input-trans-bam is required" >&2; exit 1; }
    [[ -z "$OUTPUT_CORD_BAM" ]] && { echo "ERROR: --output-cord-bam is required" >&2; exit 1; }
    [[ -z "$OUTPUT_TRANS_BAM" ]] && { echo "ERROR: --output-trans-bam is required" >&2; exit 1; }

    mkdir -p "$(dirname "$OUTPUT_CORD_BAM")" "$(dirname "$OUTPUT_TRANS_BAM")"
    _filter_one_bam "$INPUT_CORD_BAM" "$OUTPUT_CORD_BAM"
    _filter_one_bam "$INPUT_TRANS_BAM" "$OUTPUT_TRANS_BAM"
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
        rm -f "$INPUT_BAM" "${INPUT_BAM}.bai" "${INPUT_BAM%.bam}.bai"
    fi
}

_multiqc_report() {
    [[ -z "$INPUT_DIR" ]] && { echo "ERROR: --input-dir is required" >&2; exit 1; }
    [[ -z "$OUTPUT_REPORT" ]] && { echo "ERROR: --output-report is required" >&2; exit 1; }
    if [[ -z "$MULTIQC_CONFIG" ]]; then
        MULTIQC_CONFIG="${OUTPUT_REPORT%.*}.multiqc_config.yml"
    fi

    mkdir -p "$(dirname "$OUTPUT_REPORT")"
    cat > "$MULTIQC_CONFIG" <<'MULTIQC_EOF'
extra_fn_clean_exts:
    - '_rsem'
fn_ignore_dirs:
    - '*_STARpass1'
MULTIQC_EOF
    multiqc "$INPUT_DIR" -v -n "$(basename "$OUTPUT_REPORT")" -o "$(dirname "$OUTPUT_REPORT")" -c "$MULTIQC_CONFIG"
}

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$SAMPLE_LIST" ]] && printf '    --sample-list %s\n' "$SAMPLE_LIST" >&2
    [[ -n "$INPUT_FASTQ" ]] && printf '    --input-fastq %s\n' "$INPUT_FASTQ" >&2
    [[ -n "$INPUT_BAM" ]] && printf '    --input-bam %s\n' "$INPUT_BAM" >&2
    [[ -n "$INPUT_READS" ]] && printf '    --input-reads %s\n' "$INPUT_READS" >&2
    [[ -n "$INPUT_CORD_BAM" ]] && printf '    --input-cord-bam %s\n' "$INPUT_CORD_BAM" >&2
    [[ -n "$INPUT_TRANS_BAM" ]] && printf '    --input-trans-bam %s\n' "$INPUT_TRANS_BAM" >&2
    [[ -n "$DATA_DIR" ]] && printf '    --data-dir %s\n' "$DATA_DIR" >&2
    [[ -n "$INPUT_DIR" ]] && printf '    --input-dir %s\n' "$INPUT_DIR" >&2
    printf '    --cwd %s\n' "$CWD" >&2
    [[ -n "$GTF" ]] && printf '    --gtf %s\n' "$GTF" >&2
    [[ -n "$STAR_INDEX" ]] && printf '    --STAR-index %s\n' "$STAR_INDEX" >&2
    [[ -n "$REF_FLAT" ]] && printf '    --ref-flat %s\n' "$REF_FLAT" >&2
    [[ -n "$REFERENCE_FASTA" ]] && printf '    --reference-fasta %s\n' "$REFERENCE_FASTA" >&2
    [[ -n "$JAVA_MEM" ]] && printf '    --java-mem %s\n' "$JAVA_MEM" >&2
    [[ -n "$SAMPLE_ID" ]] && printf '    --sample-id %s\n' "$SAMPLE_ID" >&2
    printf '    --strand %s\n' "$STRAND" >&2
    printf '    --detection-threshold %s\n' "$DETECTION_THRESHOLD" >&2
    printf '    --mapping-quality %s\n' "$MAPPING_QUALITY" >&2
    printf '    --paired-end %s\n' "$PAIRED_END" >&2
    exit 0
fi

_dispatch() {
    case "$STEP" in
        bam_to_fastq) _bam_to_fastq ;;
        fastqc) _fastqc ;;
        star_align_1) _star_align_1 ;;
        filter_reads) _filter_reads ;;
        rnaseqc_call) _rnaseqc_call ;;
        picard_qc_star_align_2) _picard_qc_star_align_2 ;;
        multiqc_report) _multiqc_report ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: bam_to_fastq, fastqc, star_align_1, filter_reads, rnaseqc_call, picard_qc_star_align_2, multiqc_report" >&2
            exit 1
            ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    singularity exec "$CONTAINER" bash -s <<EOF
$(declare -f is_true has_value strip_dot_ext picard_strand_spec picard_java_args _bam_to_fastq _link_fastqc_outputs _sample_read_rows _fastqc _star_align_1 _move_rnaseqc_output _rnaseqc_from_bam _rnaseqc_from_fastq _rnaseqc_call _filter_one_bam _filter_reads _write_ribosomal_intervals _write_bam_summary _picard_qc_star_align_2 _multiqc_report _dispatch)
STEP="$STEP"
SAMPLE_LIST="$SAMPLE_LIST"
DATA_DIR="$DATA_DIR"
INPUT_DIR="$INPUT_DIR"
CWD="$CWD"
GTF="$GTF"
REF_FLAT="$REF_FLAT"
REFERENCE_FASTA="$REFERENCE_FASTA"
STAR_INDEX="$STAR_INDEX"
JAVA_MEM="$JAVA_MEM"
NUM_THREADS="$NUM_THREADS"
PAIRED_END="$PAIRED_END"
INPUT_BAM="$INPUT_BAM"
INPUT_READS="$INPUT_READS"
INPUT_CORD_BAM="$INPUT_CORD_BAM"
INPUT_TRANS_BAM="$INPUT_TRANS_BAM"
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
SORTED_BAM="$SORTED_BAM"
OUTPUT_FASTQ1="$OUTPUT_FASTQ1"
OUTPUT_FASTQ2="$OUTPUT_FASTQ2"
OUTPUT_PREFIX="$OUTPUT_PREFIX"
OUTPUT_CORD_BAM="$OUTPUT_CORD_BAM"
OUTPUT_TRANS_BAM="$OUTPUT_TRANS_BAM"
MD_BAM="$MD_BAM"
MD_METRICS="$MD_METRICS"
BIGWIG="$BIGWIG"
OUTPUT_SUMMARY="$OUTPUT_SUMMARY"
OUTPUT_REPORT="$OUTPUT_REPORT"
MULTIQC_CONFIG="$MULTIQC_CONFIG"
SJDB_OVERHANG="$SJDB_OVERHANG"
READ_FILES_COMMAND="$READ_FILES_COMMAND"
OUT_FILTER_MULTIMAP_NMAX="$OUT_FILTER_MULTIMAP_NMAX"
ALIGN_SJ_OVERHANG_MIN="$ALIGN_SJ_OVERHANG_MIN"
ALIGN_SJDB_OVERHANG_MIN="$ALIGN_SJDB_OVERHANG_MIN"
OUT_FILTER_MISMATCH_NMAX="$OUT_FILTER_MISMATCH_NMAX"
OUT_FILTER_MISMATCH_NOVER_LMAX="$OUT_FILTER_MISMATCH_NOVER_LMAX"
ALIGN_INTRON_MIN="$ALIGN_INTRON_MIN"
ALIGN_INTRON_MAX="$ALIGN_INTRON_MAX"
ALIGN_MATES_GAP_MAX="$ALIGN_MATES_GAP_MAX"
OUT_FILTER_TYPE="$OUT_FILTER_TYPE"
OUT_FILTER_SCORE_MIN_OVER_LREAD="$OUT_FILTER_SCORE_MIN_OVER_LREAD"
OUT_FILTER_MATCH_NMIN_OVER_LREAD="$OUT_FILTER_MATCH_NMIN_OVER_LREAD"
LIMIT_SJDB_INSERT_NSJ="$LIMIT_SJDB_INSERT_NSJ"
OUT_SAM_STRAND_FIELD="$OUT_SAM_STRAND_FIELD"
OUT_FILTER_INTRON_MOTIFS="$OUT_FILTER_INTRON_MOTIFS"
ALIGN_SOFT_CLIP_AT_REFERENCE_ENDS="$ALIGN_SOFT_CLIP_AT_REFERENCE_ENDS"
QUANT_MODE="$QUANT_MODE"
OUT_SAM_ATTR_RG_LINE="$OUT_SAM_ATTR_RG_LINE"
OUT_SAM_ATTRIBUTES="$OUT_SAM_ATTRIBUTES"
CHIM_SEGMENT_MIN="$CHIM_SEGMENT_MIN"
CHIM_JUNCTION_OVERHANG_MIN="$CHIM_JUNCTION_OVERHANG_MIN"
CHIM_OUT_TYPE="$CHIM_OUT_TYPE"
CHIM_MAIN_SEGMENT_MULT_NMAX="$CHIM_MAIN_SEGMENT_MULT_NMAX"
UNIQUE="$UNIQUE"
WASP="$WASP"
_dispatch
EOF
else
    _dispatch
fi
