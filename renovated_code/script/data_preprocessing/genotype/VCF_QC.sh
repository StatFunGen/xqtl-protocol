#!/usr/bin/env bash
# ============================================================
# VCF_QC.sh
# Mirrors selected shell-heavy steps from VCF_QC.ipynb.
#
# Steps:
#   qc
#   rename_chrs
#   dbsnp_annotate
#   qc_3
# ============================================================

STEP="${1:?Usage: $(basename "$0") <step> [args...]}"
shift

CONTAINER=""
GENO_FILE=""
CWD="output"
DBSNP=""
REFERENCE_GENOME=""
NUM_THREADS=8
BI_ALLELIC=false
SNP_ONLY=false
GT_ONLY_VCF_QC=false
SKIP_VCF_HEADER_FILTERING=false
OUTPUT=""
NOVEL_SUMSTATS=""
KNOWN_SUMSTATS=""
NOVEL_TSTV=""
KNOWN_TSTV=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)                  CONTAINER="$2"; shift 2 ;;
        --genoFile)                   GENO_FILE="$2"; shift 2 ;;
        --cwd)                        CWD="$2"; shift 2 ;;
        --dbsnp-variants)             DBSNP="$2"; shift 2 ;;
        --reference-genome)           REFERENCE_GENOME="$2"; shift 2 ;;
        --numThreads)                 NUM_THREADS="$2"; shift 2 ;;
        --bi-allelic)                 BI_ALLELIC="$2"; shift 2 ;;
        --snp-only)                   SNP_ONLY="$2"; shift 2 ;;
        --gt-only-vcf-qc)             GT_ONLY_VCF_QC="$2"; shift 2 ;;
        --skip-vcf-header-filtering)  SKIP_VCF_HEADER_FILTERING="$2"; shift 2 ;;
        --output)                     OUTPUT="$2"; shift 2 ;;
        --novel-sumstats)             NOVEL_SUMSTATS="$2"; shift 2 ;;
        --known-sumstats)             KNOWN_SUMSTATS="$2"; shift 2 ;;
        --novel-tstv)                 NOVEL_TSTV="$2"; shift 2 ;;
        --known-tstv)                 KNOWN_TSTV="$2"; shift 2 ;;
        --dry-run)                    DRY_RUN=true; shift ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

mkdir -p "$CWD"

bool_true() {
    local value="${1:-}"
    [[ "$value" == "true" || "$value" == "True" || "$value" == "TRUE" || "$value" == "1" ]]
}

stream_file() {
    local path="$1"
    if [[ "$path" == *.gz ]]; then
        gzip -dc "$path"
    else
        cat "$path"
    fi
}

infer_qc_output() {
    local input_stem suffix
    input_stem="$(basename "$GENO_FILE")"
    [[ "$input_stem" == *.vcf.gz ]] && input_stem="${input_stem%.vcf.gz}"
    [[ "$input_stem" == *.vcf ]] && input_stem="${input_stem%.vcf}"

    suffix="leftnorm"
    if bool_true "$BI_ALLELIC"; then
        suffix="biallelic"
    fi
    if bool_true "$SNP_ONLY"; then
        suffix="${suffix}.snp"
    fi
    if gt_only_vcf_qc_enabled; then
        suffix="${suffix}.gt_only"
    fi
    printf '%s/%s.%s.vcf.gz' "$CWD" "$input_stem" "$suffix"
}

gt_only_vcf_qc_enabled() {
    bool_true "$GT_ONLY_VCF_QC" || bool_true "$SKIP_VCF_HEADER_FILTERING"
}

_qc() {
    [[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }
    [[ -z "$DBSNP" ]] && { echo "ERROR: --dbsnp-variants is required" >&2; exit 1; }
    [[ -z "$REFERENCE_GENOME" ]] && { echo "ERROR: --reference-genome is required" >&2; exit 1; }

    local out split_cmd
    out="$(infer_qc_output)"

    if bool_true "$BI_ALLELIC"; then
        split_cmd=(bcftools view -m2 -M2)
    else
        split_cmd=(bcftools norm -m-any)
    fi
    if bool_true "$SNP_ONLY"; then
        split_cmd+=(-v snps)
    fi

    if [[ -z "${BCFTOOLS_PLUGINS:-}" && -d "$(dirname "$(dirname "$(command -v bcftools)")")/libexec/bcftools" ]]; then
        export BCFTOOLS_PLUGINS="$(dirname "$(dirname "$(command -v bcftools)")")/libexec/bcftools"
    fi

    "${split_cmd[@]}" "$GENO_FILE" \
        | bcftools norm -d exact --check-ref ws -f "$REFERENCE_GENOME" --threads "$NUM_THREADS" \
        | {
            if gt_only_vcf_qc_enabled; then
                bcftools +fill-tags -- -t F_MISSING,HWE
            else
                bcftools +fill-tags -- -t all,F_MISSING,'VD=sum(FMT/DP)'
            fi
        } \
        | bcftools annotate -x ID -I +'%CHROM:%POS:%REF:%ALT' \
        | bcftools annotate \
            -a "$DBSNP" \
            -h <(echo '##INFO=<ID=RSID,Number=1,Type=String,Description="dbSNP rsID">') \
            -c CHROM,FROM,TO,INFO/RSID \
            -Oz --threads "$NUM_THREADS" -o "$out"
}

_rename_chrs() {
    [[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }
    [[ -z "$OUTPUT" ]] && { echo "ERROR: --output is required" >&2; exit 1; }

    local chr_map
    chr_map="${OUTPUT%.vcf.gz}.chr_name_conv.txt"
    for chrom in {1..22} X Y MT; do
        printf '%s\tchr%s\n' "$chrom" "$chrom"
    done > "$chr_map"
    bcftools annotate --rename-chrs "$chr_map" "$GENO_FILE" -Oz -o "$OUTPUT"
    tabix -p vcf "$OUTPUT"
}

_dbsnp_annotate() {
    [[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }
    [[ -z "$OUTPUT" ]] && { echo "ERROR: --output is required" >&2; exit 1; }

    bcftools query -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\n' "$GENO_FILE" \
        | awk 'BEGIN { OFS="\t" } {
            if (length($4) > length($5)) {
                end_pos = $2 + (length($4) - 1)
            } else {
                end_pos = $2 + (length($5) - 1)
            }
            print $1, $2, end_pos, $3
        }' \
        | bgzip -c > "$OUTPUT"
}

_append_file_info() {
    local stdout_file="$1"
    local file_path="$2"
    local output_size output_rows output_columns output_header_rows output_preview
    output_size="$(ls -lh "$file_path" | awk '{print $5}')"
    output_rows="$(stream_file "$file_path" | wc -l | awk '{print $1}')"
    output_columns="$(stream_file "$file_path" | grep -v '##' | head -1 | wc -w | awk '{print $1}')"
    output_header_rows="$(stream_file "$file_path" | grep '##' | wc -l | awk '{print $1}')"
    output_preview="$(stream_file "$file_path" | grep -v '##' | head | cut -f 1-11)"
    printf 'output_info: %s\noutput_size: %s\noutput_rows: %s\noutput_column: %s\noutput_header_row: %s\noutput_preview:\n%s\n' \
        "$file_path" "$output_size" "$output_rows" "$output_columns" "$output_header_rows" "$output_preview" >> "$stdout_file"
}

_qc_3() {
    [[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }
    [[ -z "$NOVEL_SUMSTATS" || -z "$KNOWN_SUMSTATS" || -z "$NOVEL_TSTV" || -z "$KNOWN_TSTV" ]] && {
        echo "ERROR: --novel-sumstats, --known-sumstats, --novel-tstv, and --known-tstv are required" >&2
        exit 1
    }

    bcftools stats -i 'RSID="."' -v "$GENO_FILE" > "$NOVEL_SUMSTATS"
    bcftools stats -i 'RSID!="."' -v "$GENO_FILE" > "$KNOWN_SUMSTATS"
    bcftools filter -i 'RSID="."' "$GENO_FILE" | SnpSift tstv - > "$NOVEL_TSTV"
    bcftools filter -i 'RSID!="."' "$GENO_FILE" | SnpSift tstv - > "$KNOWN_TSTV"

    local stdout_file="${NOVEL_SUMSTATS%.*}.stdout"
    : > "$stdout_file"
    for file_path in "$NOVEL_SUMSTATS" "$KNOWN_SUMSTATS" "$NOVEL_TSTV" "$KNOWN_TSTV"; do
        _append_file_info "$stdout_file" "$file_path"
    done
}

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$GENO_FILE" ]] && printf '    --genoFile %s\n' "$GENO_FILE" >&2
    [[ -n "$DBSNP" ]] && printf '    --dbsnp-variants %s\n' "$DBSNP" >&2
    [[ -n "$REFERENCE_GENOME" ]] && printf '    --reference-genome %s\n' "$REFERENCE_GENOME" >&2
    printf '    --cwd %s\n' "$CWD" >&2
    printf '    --numThreads %s\n' "$NUM_THREADS" >&2
    printf '    --bi-allelic %s\n' "$BI_ALLELIC" >&2
    printf '    --snp-only %s\n' "$SNP_ONLY" >&2
    printf '    --gt-only-vcf-qc %s\n' "$GT_ONLY_VCF_QC" >&2
    printf '    --skip-vcf-header-filtering %s\n' "$SKIP_VCF_HEADER_FILTERING" >&2
    [[ -n "$OUTPUT" ]] && printf '    --output %s\n' "$OUTPUT" >&2
    [[ -n "$NOVEL_SUMSTATS" ]] && printf '    --novel-sumstats %s\n' "$NOVEL_SUMSTATS" >&2
    [[ -n "$KNOWN_SUMSTATS" ]] && printf '    --known-sumstats %s\n' "$KNOWN_SUMSTATS" >&2
    [[ -n "$NOVEL_TSTV" ]] && printf '    --novel-tstv %s\n' "$NOVEL_TSTV" >&2
    [[ -n "$KNOWN_TSTV" ]] && printf '    --known-tstv %s\n' "$KNOWN_TSTV" >&2
    exit 0
fi

run_dispatch() {
    case "$STEP" in
        qc) _qc ;;
        rename_chrs) _rename_chrs ;;
        dbsnp_annotate) _dbsnp_annotate ;;
        qc_3) _qc_3 ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: qc, rename_chrs, dbsnp_annotate, qc_3" >&2
            exit 1
            ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    singularity exec "$CONTAINER" bash -s <<EOF
$(declare -f bool_true stream_file infer_qc_output _qc _rename_chrs _dbsnp_annotate _append_file_info _qc_3 run_dispatch)
STEP="$STEP"
GENO_FILE="$GENO_FILE"
CWD="$CWD"
DBSNP="$DBSNP"
REFERENCE_GENOME="$REFERENCE_GENOME"
NUM_THREADS="$NUM_THREADS"
BI_ALLELIC="$BI_ALLELIC"
SNP_ONLY="$SNP_ONLY"
SKIP_VCF_HEADER_FILTERING="$SKIP_VCF_HEADER_FILTERING"
OUTPUT="$OUTPUT"
NOVEL_SUMSTATS="$NOVEL_SUMSTATS"
KNOWN_SUMSTATS="$KNOWN_SUMSTATS"
NOVEL_TSTV="$NOVEL_TSTV"
KNOWN_TSTV="$KNOWN_TSTV"
run_dispatch
EOF
else
    run_dispatch
fi
