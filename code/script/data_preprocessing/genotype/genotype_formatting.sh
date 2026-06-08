#!/usr/bin/env bash
# ============================================================
# genotype_formatting.sh
# Mirrors: code/data_preprocessing/genotype/genotype_formatting.ipynb
#
# Steps:
#   plink_to_vcf_1       — convert a PLINK BED prefix to bgzipped VCF
#   vcf_to_plink         — convert a single VCF to PLINK binary format
#   genotype_by_region_1 — subset a PLINK BED prefix to one region
#   merge_plink          — merge multiple PLINK files into one PLINK set
#   genotype_by_chrom    — split PLINK file into per-chromosome files
#   merge_vcf            — merge multiple VCFs into one bgzipped VCF
#
# Usage:
#   genotype_formatting.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run genotype_formatting.ipynb <step> [--flag value ...]
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [--container PATH] [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
CONTAINER=""
GENO_FILE=""
CWD="output"
NAME=""
CHROM=()
NUM_THREADS=8
MEM_LIMIT=""
OUTPUT=""
PLINK_COMMAND=""
MAKE_COMMAND=""
OUTPUT_FORMAT=""
REGION_CHROM=""
REGION_START=""
REGION_END=""
WINDOW=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)    CONTAINER="$2";    shift 2 ;;
        --genoFile)     GENO_FILE="$2";    shift 2 ;;
        --cwd)          CWD="$2";          shift 2 ;;
        --name)         NAME="$2";         shift 2 ;;
        --mem)          MEM_LIMIT="$2";    shift 2 ;;
        --output)       OUTPUT="$2";       shift 2 ;;
        --plink-command) PLINK_COMMAND="$2"; shift 2 ;;
        --make-command) MAKE_COMMAND="$2"; shift 2 ;;
        --output-format) OUTPUT_FORMAT="$2"; shift 2 ;;
        --region-chrom) REGION_CHROM="$2"; shift 2 ;;
        --region-start) REGION_START="$2"; shift 2 ;;
        --region-end)   REGION_END="$2";   shift 2 ;;
        --window)       WINDOW="$2";       shift 2 ;;
        --chrom)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CHROM+=("$1"); shift
            done ;;
        --numThreads)   NUM_THREADS="$2";  shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

[[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }

mkdir -p "$CWD"

# Derive name from genoFile if not provided
if [[ -z "$NAME" ]]; then
    NAME="$(basename "$GENO_FILE")"
    NAME="${NAME%.vcf.gz}"; NAME="${NAME%.vcf}"; NAME="${NAME%.bed}"
fi

_mem_spec_to_mb() {
    local spec="${1^^}"
    spec="${spec// /}"
    [[ -z "$spec" ]] && return 1
    if [[ "$spec" =~ ^([0-9]+)([KMGTP]?)B?$ ]]; then
        local value="${BASH_REMATCH[1]}"
        local unit="${BASH_REMATCH[2]}"
        case "$unit" in
            ""|M) echo "$value" ;;
            K) echo $(( (value + 999) / 1000 )) ;;
            G) echo "$((value * 1000))" ;;
            T) echo "$((value * 1000 * 1000))" ;;
            P) echo "$((value * 1000 * 1000 * 1000))" ;;
            *) return 1 ;;
        esac
        return 0
    fi
    return 1
}

_plink_workspace_mb() {
    local total_mb
    total_mb="$(_mem_spec_to_mb "$1")" || return 1
    # Match the source notebook behavior and leave headroom under the task cap.
    echo $(( total_mb * 9 / 10 ))
}

_detect_plink_format() {
    local path="$1" base
    case "$path" in
        *.bed|*.bim|*.fam) echo "plink1"; return 0 ;;
        *.pgen|*.pvar|*.psam) echo "plink2"; return 0 ;;
    esac
    base="${path%.*}"
    if [[ -f "${base}.bim" && -f "${base}.fam" ]]; then
        echo "plink1"
    elif [[ -f "${base}.pvar" && -f "${base}.psam" ]]; then
        echo "plink2"
    else
        echo "plink1"
    fi
}

_plink_prefix() {
    local path="$1" format="$2"
    if [[ "$format" == "plink2" ]]; then
        path="${path%.pgen}"
        path="${path%.pvar}"
        path="${path%.psam}"
    else
        path="${path%.bed}"
        path="${path%.bim}"
        path="${path%.fam}"
    fi
    echo "$path"
}

_plink_command_for_format() {
    [[ "$1" == "plink2" ]] && echo "--pfile" || echo "--bfile"
}

_make_command_for_format() {
    [[ "$1" == "plink2" ]] && echo "--make-pgen" || echo "--make-bed"
}

_output_ext_for_format() {
    [[ "$1" == "plink2" ]] && echo ".pgen" || echo ".bed"
}

_output_prefix() {
    local output="$1" format="$2"
    if [[ "$format" == "plink2" ]]; then
        output="${output%.pgen}"
    else
        output="${output%.bed}"
    fi
    echo "$output"
}

# ── Step implementations ─────────────────────────────────────────────────────

_plink_to_vcf_1() {
    local bed="$1" output="$2" threads="$3" mem_spec="${4:-}"
    [[ -z "$output" ]] && { echo "ERROR: --output is required for plink_to_vcf_1" >&2; exit 1; }

    local bed_prefix out_prefix vcf_path
    bed_prefix="${bed%.bed}"
    out_prefix="${output%.vcf.gz}"
    out_prefix="${out_prefix%.vcf}"
    vcf_path="${out_prefix}.vcf"

    mkdir -p "$(dirname "$output")"

    local -a memory_arg=()
    if [[ -n "$mem_spec" ]]; then
        local plink_mem_mb
        if plink_mem_mb="$(_plink_workspace_mb "$mem_spec")"; then
            memory_arg=(--memory "$plink_mem_mb")
        fi
    fi

    plink2 --bfile "$bed_prefix" \
        --recode vcf-iid \
        --out "$out_prefix" \
        --threads "$threads" \
        "${memory_arg[@]}" \
        --output-chr chrMT
    bgzip -l 9 "$vcf_path"
    tabix -f -p vcf "$output"
}

_vcf_to_plink() {
    local vcf="$1" outdir="$2" name="$3" threads="$4"
    echo "=== vcf_to_plink: $vcf → $outdir/$name ===" >&2
    plink2 \
        --vcf "$vcf" \
        --make-bed \
        --out "${outdir}/${name}" \
        --threads "$threads" \
        --no-psam-pheno \
        --set-missing-var-ids '@:#:\$r:\$a' \
        --new-id-max-allele-len 1000
    echo "Output: ${outdir}/${name}.{bed,bim,fam}" >&2
}

_genotype_by_region_1() {
    local bed="$1" output="$2" chrom="$3" start="$4" end="$5" window="$6"
    [[ -z "$output" ]] && { echo "ERROR: --output is required for genotype_by_region_1" >&2; exit 1; }
    [[ -z "$chrom" || -z "$start" || -z "$end" ]] && {
        echo "ERROR: --region-chrom, --region-start, and --region-end are required for genotype_by_region_1" >&2
        exit 1
    }

    local bed_prefix out_prefix from_bp to_bp
    bed_prefix="${bed%.bed}"
    out_prefix="${output%.bed}"
    from_bp=$(( start - window ))
    if [[ "$from_bp" -lt 0 ]]; then
        from_bp=0
    fi
    to_bp=$(( end + window ))

    mkdir -p "$(dirname "$output")"

    plink2 --bfile "$bed_prefix" \
        --make-bed \
        --out "$out_prefix" \
        --chr "$chrom" \
        --from-bp "$from_bp" \
        --to-bp "$to_bp" \
        --allow-no-sex --output-chr chrMT || touch "$output"
}

_merge_plink() {
    local file_list="$1" outdir="$2" name="$3" threads="$4" mem_spec="${5:-}"
    echo "=== merge_plink: merging files in $file_list ===" >&2

    local -a inputs=()
    if [[ -f "$file_list" ]]; then
        mapfile -t inputs < "$file_list"
    else
        read -r -a inputs <<< "$file_list"
    fi

    [[ "${#inputs[@]}" -eq 0 ]] && { echo "ERROR: merge_plink received no input files" >&2; exit 1; }

    # Determine input type (VCF list, PLINK1 bed, or PLINK2 pgen)
    local first_file
    first_file="${inputs[0]}"
    local input_kind="bed"

    local tmp_merge="${outdir}/${name}_merge_list.txt"
    > "$tmp_merge"

    if [[ "$first_file" == *.vcf.gz || "$first_file" == *.vcf ]]; then
        echo "ERROR: merge_plink expects PLINK inputs, not VCF inputs." >&2
        echo "ERROR: Run the notebook-defined vcf_to_plink stage first, then merge_plink." >&2
        exit 2
    elif [[ "$first_file" == *.pgen ]]; then
        input_kind="pgen"
    fi

    # Strip the file-type extension for the merge list.
    for f in "${inputs[@]}"; do
        [[ -z "$f" ]] && continue
        if [[ "$input_kind" == "pgen" ]]; then
            echo "${f%.pgen}" >> "$tmp_merge"
        else
            echo "${f%.bed}" >> "$tmp_merge"
        fi
    done

    local -a memory_arg=()
    if [[ -n "$mem_spec" ]]; then
        local plink_mem_mb
        if plink_mem_mb="$(_plink_workspace_mb "$mem_spec")"; then
            echo "Using PLINK memory cap: ${plink_mem_mb} MB (from task mem ${mem_spec})" >&2
            memory_arg=(--memory "$plink_mem_mb")
        else
            echo "WARN: Could not parse --mem ${mem_spec}; falling back to PLINK default memory behavior" >&2
        fi
    fi

    local n_files
    n_files="$(wc -l < "$tmp_merge")"
    if [[ "$n_files" -eq 1 ]]; then
        # Only one file — just rename
        local src
        src="$(cat "$tmp_merge")"
        if [[ "$input_kind" == "pgen" ]]; then
            for ext in pgen pvar psam; do
                cp "${src}.${ext}" "${outdir}/${name}.${ext}"
            done
        else
            for ext in bed bim fam; do
                cp "${src}.${ext}" "${outdir}/${name}.${ext}"
            done
        fi
    else
        local first_file_prefix
        first_file_prefix="$(head -1 "$tmp_merge")"
        tail -n +2 "$tmp_merge" > "${tmp_merge}.rest"
        if [[ "$input_kind" == "pgen" ]]; then
            plink2 --pfile "$first_file_prefix" \
                   --pmerge-list "${tmp_merge}.rest" \
                   --make-pgen \
                   --out "${outdir}/${name}" \
                   --threads "$threads" \
                   "${memory_arg[@]}"
        else
            plink --keep-allele-order --bfile "$first_file_prefix" \
                  --merge-list "${tmp_merge}.rest" \
                  --make-bed \
                  --out "${outdir}/${name}" \
                  --threads "$threads" \
                  "${memory_arg[@]}"
        fi
        rm -f "${tmp_merge}.rest"
    fi
    rm -f "$tmp_merge"
    if [[ "$input_kind" == "pgen" ]]; then
        echo "Output: ${outdir}/${name}.{pgen,pvar,psam}" >&2
    else
        echo "Output: ${outdir}/${name}.{bed,bim,fam}" >&2
    fi
}

_merge_vcf() {
    local file_list="$1" output="$2"
    [[ -z "$output" ]] && { echo "ERROR: --output is required for merge_vcf" >&2; exit 1; }

    local -a inputs=()
    if [[ -f "$file_list" ]]; then
        mapfile -t inputs < "$file_list"
    else
        read -r -a inputs <<< "$file_list"
    fi

    [[ "${#inputs[@]}" -eq 0 ]] && { echo "ERROR: merge_vcf received no input files" >&2; exit 1; }

    mkdir -p "$(dirname "$output")"
    bcftools concat -Oz "${inputs[@]}" > "$output"
    tabix -p vcf "$output"
}

_genotype_by_chrom() {
    local geno="$1" outdir="$2" name="$3" output="$4" plink_command="$5" make_command="$6" output_format="$7" threads="$8" mem_spec="${9:-}"
    shift 9
    local chroms=("$@")
    echo "=== genotype_by_chrom: splitting ${#chroms[@]} chromosomes ===" >&2

    [[ -n "$output" && "${#chroms[@]}" -ne 1 ]] && {
        echo "ERROR: --output can only be used with one --chrom value" >&2
        exit 1
    }

    if [[ -z "$output_format" ]]; then
        output_format="$(_detect_plink_format "$geno")"
    fi
    if [[ -z "$plink_command" ]]; then
        plink_command="$(_plink_command_for_format "$output_format")"
    fi
    if [[ -z "$make_command" ]]; then
        make_command="$(_make_command_for_format "$output_format")"
    fi

    local input_prefix output_ext
    input_prefix="$(_plink_prefix "$geno" "$output_format")"
    output_ext="$(_output_ext_for_format "$output_format")"

    local manifest="${outdir}/${name}.genotype_by_chrom_files.txt"
    > "$manifest"

    local -a memory_arg=()
    if [[ -n "$mem_spec" ]]; then
        local plink_mem_mb
        if plink_mem_mb="$(_plink_workspace_mb "$mem_spec")"; then
            memory_arg=(--memory "$plink_mem_mb")
        fi
    fi

    for chrom in "${chroms[@]}"; do
        local chrom_label="$chrom"
        chrom_label="${chrom_label#[}"
        chrom_label="${chrom_label%]}"
        chrom_label="${chrom_label#,}"
        chrom_label="${chrom_label%,}"
        chrom_label="${chrom_label#\'}"
        chrom_label="${chrom_label%\'}"
        chrom_label="${chrom_label#\"}"
        chrom_label="${chrom_label%\"}"
        local chrom_query="${chrom_label#chr}"
        local out
        if [[ -n "$output" ]]; then
            out="$(_output_prefix "$output" "$output_format")"
        else
            out="${outdir}/${name}.${chrom_label}"
        fi
        plink2 \
            "$plink_command" "$input_prefix" \
            --chr "$chrom_query" \
            "$make_command" \
            --out "$out" \
            --threads "$threads" \
            --allow-no-sex \
            "${memory_arg[@]}"
        echo "${out}${output_ext}" >> "$manifest"
        if [[ "$output_format" == "plink2" ]]; then
            echo "  Written: ${out}.{pgen,pvar,psam}" >&2
        else
            echo "  Written: ${out}.{bed,bim,fam}" >&2
        fi
    done
    echo "Manifest: $manifest" >&2
}

export -f _mem_spec_to_mb _plink_workspace_mb _detect_plink_format _plink_prefix _plink_command_for_format _make_command_for_format _output_ext_for_format _output_prefix _plink_to_vcf_1 _vcf_to_plink _genotype_by_region_1 _merge_plink _merge_vcf _genotype_by_chrom

# ── Dry-run: print parameters and exit ───────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$CONTAINER" ]] && printf '[DRY-RUN] Container: %s\n' "$CONTAINER" >&2
    echo "[DRY-RUN] Full command (copy-paste to debug):" >&2
    printf '  bash %s %s --dry-run \\\n' "$(realpath "$0" 2>/dev/null || echo "$0")" "$STEP" >&2
    [[ -n "$CONTAINER" ]] && printf '    --container %s \\\n' "$CONTAINER" >&2
    printf '    --genoFile %s \\\n'   "$GENO_FILE"  >&2
    printf '    --cwd %s \\\n'        "$CWD"         >&2
    [[ -n "$NAME" ]] && printf '    --name %s \\\n' "$NAME" >&2
    [[ -n "$MEM_LIMIT" ]] && printf '    --mem %s \\\n' "$MEM_LIMIT" >&2
    [[ -n "$OUTPUT" ]] && printf '    --output %s \\\n' "$OUTPUT" >&2
    [[ -n "$PLINK_COMMAND" ]] && printf '    --plink-command %s \\\n' "$PLINK_COMMAND" >&2
    [[ -n "$MAKE_COMMAND" ]] && printf '    --make-command %s \\\n' "$MAKE_COMMAND" >&2
    [[ -n "$OUTPUT_FORMAT" ]] && printf '    --output-format %s \\\n' "$OUTPUT_FORMAT" >&2
    [[ -n "$REGION_CHROM" ]] && printf '    --region-chrom %s \\\n' "$REGION_CHROM" >&2
    [[ -n "$REGION_START" ]] && printf '    --region-start %s \\\n' "$REGION_START" >&2
    [[ -n "$REGION_END" ]] && printf '    --region-end %s \\\n' "$REGION_END" >&2
    [[ "$WINDOW" != "0" ]] && printf '    --window %s \\\n' "$WINDOW" >&2
    [[ ${#CHROM[@]} -gt 0 ]] && printf '    --chrom %s \\\n' "${CHROM[*]}" >&2
    printf '    --numThreads %s\n'     "$NUM_THREADS" >&2
    echo "[DRY-RUN] Input file check:" >&2
    if [[ -e "$GENO_FILE" ]]; then printf '  ✓ %s\n' "$GENO_FILE" >&2
    else                            printf '  ✗ NOT FOUND: %s\n' "$GENO_FILE" >&2; fi
    exit 0
fi


# ── Dispatch ─────────────────────────────────────────────────────────────────
_dispatch() {
    case "$STEP" in
        plink_to_vcf_1)
            _plink_to_vcf_1 "$GENO_FILE" "$OUTPUT" "$NUM_THREADS" "$MEM_LIMIT"
            ;;
        vcf_to_plink)
            _vcf_to_plink "$GENO_FILE" "$CWD" "$NAME" "$NUM_THREADS"
            ;;
        genotype_by_region_1)
            _genotype_by_region_1 "$GENO_FILE" "$OUTPUT" "$REGION_CHROM" "$REGION_START" "$REGION_END" "$WINDOW"
            ;;
        merge_plink)
            _merge_plink "$GENO_FILE" "$CWD" "$NAME" "$NUM_THREADS" "$MEM_LIMIT"
            ;;
        merge_vcf)
            _merge_vcf "$GENO_FILE" "$OUTPUT"
            ;;
        genotype_by_chrom)
            [[ ${#CHROM[@]} -eq 0 ]] && { echo "ERROR: --chrom is required for genotype_by_chrom" >&2; exit 1; }
            _genotype_by_chrom "$GENO_FILE" "$CWD" "$NAME" "$OUTPUT" "$PLINK_COMMAND" "$MAKE_COMMAND" "$OUTPUT_FORMAT" "$NUM_THREADS" "$MEM_LIMIT" "${CHROM[@]}"
            ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: plink_to_vcf_1, vcf_to_plink, genotype_by_region_1, merge_plink, merge_vcf, genotype_by_chrom" >&2
            exit 1 ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    # Re-export all parsed variables into the container
singularity exec "$CONTAINER" bash -s << EOF
$(declare -f _mem_spec_to_mb _plink_workspace_mb _detect_plink_format _plink_prefix _plink_command_for_format _make_command_for_format _output_ext_for_format _output_prefix _plink_to_vcf_1 _vcf_to_plink _genotype_by_region_1 _merge_plink _merge_vcf _genotype_by_chrom)
STEP="$STEP" GENO_FILE="$GENO_FILE" CWD="$CWD" NAME="$NAME"
OUTPUT="$OUTPUT" REGION_CHROM="$REGION_CHROM" REGION_START="$REGION_START" REGION_END="$REGION_END" WINDOW="$WINDOW"
PLINK_COMMAND="$PLINK_COMMAND" MAKE_COMMAND="$MAKE_COMMAND" OUTPUT_FORMAT="$OUTPUT_FORMAT"
NUM_THREADS="$NUM_THREADS" MEM_LIMIT="$MEM_LIMIT" CHROM=(${CHROM[*]+"${CHROM[*]}"})
$(declare -f _dispatch)
_dispatch
EOF
else
    _dispatch
fi
