#!/usr/bin/env bash
# ============================================================
# genotype_formatting.sh
# Mirrors: code/data_preprocessing/genotype/genotype_formatting.ipynb
#
# Steps:
#   vcf_to_plink    — convert a single VCF to PLINK binary format
#   merge_plink     — merge multiple VCFs/PLINK files into one PLINK set
#   genotype_by_chrom — split PLINK file into per-chromosome files
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
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)    CONTAINER="$2";    shift 2 ;;
        --genoFile)     GENO_FILE="$2";    shift 2 ;;
        --cwd)          CWD="$2";          shift 2 ;;
        --name)         NAME="$2";         shift 2 ;;
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

# ── Step implementations ─────────────────────────────────────────────────────

_vcf_to_plink() {
    local vcf="$1" outdir="$2" name="$3" threads="$4"
    echo "=== vcf_to_plink: $vcf → $outdir/$name ===" >&2
    plink2 \
        --vcf "$vcf" \
        --make-bed \
        --out "${outdir}/${name}" \
        --threads "$threads" \
        --memory 16000 \
        --no-psam-pheno \
        --set-missing-var-ids '@:#:\$r:\$a' \
        --new-id-max-allele-len 1000
    echo "Output: ${outdir}/${name}.{bed,bim,fam}" >&2
}

_merge_plink() {
    local file_list="$1" outdir="$2" name="$3" threads="$4"
    echo "=== merge_plink: merging files in $file_list ===" >&2

    # Determine input type (VCF list or PLINK list)
    local first_file
    first_file="$(head -1 "$file_list")"

    local tmp_merge="${outdir}/${name}_merge_list.txt"

    if [[ "$first_file" == *.vcf.gz || "$first_file" == *.vcf ]]; then
        # Convert each VCF to plink, then merge
        > "$tmp_merge"
        while IFS= read -r vcf; do
            [[ -z "$vcf" ]] && continue
            local bname
            bname="$(basename "$vcf" .vcf.gz)"; bname="${bname%.vcf}"
            local tmp_out="${outdir}/tmp_${bname}"
            plink2 --vcf "$vcf" --make-bed --out "$tmp_out" \
                --threads "$threads" --memory 16000 --no-psam-pheno \
                --set-missing-var-ids '@:#:\$r:\$a' --new-id-max-allele-len 1000
            echo "$tmp_out" >> "$tmp_merge"
        done < "$file_list"
    else
        # PLINK files — strip .bed extension for merge list
        while IFS= read -r f; do
            [[ -z "$f" ]] && continue
            echo "${f%.bed}" >> "$tmp_merge"
        done < "$file_list"
    fi

    # plink1 merge (plink2 pmerge for large sets)
    local n_files
    n_files="$(wc -l < "$tmp_merge")"
    if [[ "$n_files" -eq 1 ]]; then
        # Only one file — just rename
        local src
        src="$(cat "$tmp_merge")"
        for ext in bed bim fam; do
            cp "${src}.${ext}" "${outdir}/${name}.${ext}"
        done
    else
        local first_file_prefix
        first_file_prefix="$(head -1 "$tmp_merge")"
        tail -n +2 "$tmp_merge" > "${tmp_merge}.rest"
        plink --bfile "$first_file_prefix" \
              --merge-list "${tmp_merge}.rest" \
              --make-bed \
              --out "${outdir}/${name}" \
              --threads "$threads"
        rm -f "${tmp_merge}.rest"
    fi
    rm -f "$tmp_merge"
    echo "Output: ${outdir}/${name}.{bed,bim,fam}" >&2
}

_genotype_by_chrom() {
    local bed="$1" outdir="$2" name="$3" threads="$4"
    shift 4
    local chroms=("$@")
    echo "=== genotype_by_chrom: splitting ${#chroms[@]} chromosomes ===" >&2

    local manifest="${outdir}/${name}.genotype_by_chrom_files.txt"
    > "$manifest"

    for chrom in "${chroms[@]}"; do
        local out="${outdir}/${name}.${chrom}"
        plink2 \
            --bfile "${bed%.bed}" \
            --chr "$chrom" \
            --make-bed \
            --out "$out" \
            --threads "$threads" \
            --memory 16000
        echo "${out}.bed" >> "$manifest"
        echo "  Written: ${out}.{bed,bim,fam}" >&2
    done
    echo "Manifest: $manifest" >&2
}

export -f _vcf_to_plink _merge_plink _genotype_by_chrom

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
        vcf_to_plink)
            _vcf_to_plink "$GENO_FILE" "$CWD" "$NAME" "$NUM_THREADS"
            ;;
        merge_plink)
            _merge_plink "$GENO_FILE" "$CWD" "$NAME" "$NUM_THREADS"
            ;;
        genotype_by_chrom)
            [[ ${#CHROM[@]} -eq 0 ]] && { echo "ERROR: --chrom is required for genotype_by_chrom" >&2; exit 1; }
            _genotype_by_chrom "$GENO_FILE" "$CWD" "$NAME" "$NUM_THREADS" "${CHROM[@]}"
            ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: vcf_to_plink, merge_plink, genotype_by_chrom" >&2
            exit 1 ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    # Re-export all parsed variables into the container
    singularity exec "$CONTAINER" bash -s << EOF
$(declare -f _vcf_to_plink _merge_plink _genotype_by_chrom)
STEP="$STEP" GENO_FILE="$GENO_FILE" CWD="$CWD" NAME="$NAME"
NUM_THREADS="$NUM_THREADS" CHROM=(${CHROM[*]+"${CHROM[*]}"})
$(declare -f _dispatch)
_dispatch
EOF
else
    _dispatch
fi
