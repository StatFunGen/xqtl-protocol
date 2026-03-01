#!/usr/bin/env bash
# ============================================================
# VCF_QC.sh
# Mirrors: code/data_preprocessing/genotype/VCF_QC.ipynb
#
# Steps:
#   qc — VCF quality control: left-normalize, rename chromosomes,
#          annotate with dbSNP rsIDs, apply variant-level filters
#
# Usage:
#   VCF_QC.sh qc [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run VCF_QC.ipynb qc [--flag value ...]
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [--container PATH] [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
CONTAINER=""
GENO_FILE=""
CWD="output"
DBSNP=""
REFERENCE_GENOME=""
NUM_THREADS=8
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)          CONTAINER="$2";          shift 2 ;;
        --genoFile)           GENO_FILE="$2";          shift 2 ;;
        --cwd)                CWD="$2";                shift 2 ;;
        --dbsnp-variants)     DBSNP="$2";              shift 2 ;;
        --reference-genome)   REFERENCE_GENOME="$2";   shift 2 ;;
        --numThreads)         NUM_THREADS="$2";         shift 2 ;;
        --dry-run)            DRY_RUN=true;              shift ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

[[ -z "$GENO_FILE" ]]         && { echo "ERROR: --genoFile is required" >&2; exit 1; }
[[ -z "$DBSNP" ]]             && { echo "ERROR: --dbsnp-variants is required" >&2; exit 1; }
[[ -z "$REFERENCE_GENOME" ]]  && { echo "ERROR: --reference-genome is required" >&2; exit 1; }

mkdir -p "$CWD"

# Derive output basename
BASENAME="$(basename "$GENO_FILE" .vcf.gz)"
BASENAME="$(basename "$BASENAME" .vcf)"
OUT_PREFIX="${CWD}/${BASENAME}"

# ── Inner logic (runs inside container if requested) ─────────────────────────
_run_qc() {
    set -euo pipefail
    local geno="$1" out="$2" dbsnp="$3" ref="$4" threads="$5"

    echo "=== Step 1: Left-normalize and rename chromosomes ===" >&2
    # Add 'chr' prefix to chromosomes if missing, then left-normalize
    bcftools annotate \
        --rename-chrs <(for i in $(seq 1 22) X Y MT; do echo -e "${i}\tchr${i}"; done) \
        "$geno" \
        --output-type u \
    | bcftools norm \
        --fasta-ref "$ref" \
        --multiallelics -any \
        --output-type u \
    | bcftools view \
        --min-alleles 2 \
        --max-alleles 2 \
        --types snps,indels \
        --output-type u \
    > "${out}.add_chr.leftnorm.bcf"

    echo "=== Step 2: Annotate with dbSNP rsIDs ===" >&2
    bcftools annotate \
        --annotations "$dbsnp" \
        --columns ID \
        --output-type u \
        "${out}.add_chr.leftnorm.bcf" \
    > "${out}.add_chr.leftnorm.annotated.bcf"

    echo "=== Step 3: Filter variants ===" >&2
    # Remove variants with missing IDs or duplicate positions
    bcftools view \
        --exclude 'ID="." || ID="N/A"' \
        --output-type u \
        "${out}.add_chr.leftnorm.annotated.bcf" \
    | bcftools norm \
        --rm-dup all \
        --output-type z \
        --threads "$threads" \
        --output "${out}.add_chr.leftnorm.filtered.vcf.gz"

    bcftools index --tbi --threads "$threads" "${out}.add_chr.leftnorm.filtered.vcf.gz"

    # Clean up intermediates
    rm -f "${out}.add_chr.leftnorm.bcf" "${out}.add_chr.leftnorm.annotated.bcf"

    echo "Output: ${out}.add_chr.leftnorm.filtered.vcf.gz" >&2
}

export -f _run_qc

# ── Dry-run: print parameters and exit ───────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$CONTAINER" ]] && printf '[DRY-RUN] Container: %s\n' "$CONTAINER" >&2
    echo "[DRY-RUN] Full command (copy-paste to debug):" >&2
    printf '  bash %s %s --dry-run \\\n' "$(realpath "$0" 2>/dev/null || echo "$0")" "$STEP" >&2
    [[ -n "$CONTAINER"        ]] && printf '    --container %s \\\n'        "$CONTAINER"       >&2
    printf '    --genoFile %s \\\n'          "$GENO_FILE"       >&2
    printf '    --cwd %s \\\n'               "$CWD"             >&2
    printf '    --dbsnp-variants %s \\\n'    "$DBSNP"           >&2
    printf '    --reference-genome %s \\\n'  "$REFERENCE_GENOME" >&2
    printf '    --numThreads %s\n'             "$NUM_THREADS"     >&2
    echo "[DRY-RUN] Input file check:" >&2
    for _f in "$GENO_FILE" "$DBSNP" "$REFERENCE_GENOME"; do
        [[ -z "$_f" ]] && continue
        if [[ -e "$_f" ]]; then printf '  ✓ %s\n' "$_f" >&2
        else               printf '  ✗ NOT FOUND: %s\n' "$_f" >&2; fi
    done
    exit 0
fi


case "$STEP" in
    qc)
        if [[ -n "$CONTAINER" ]]; then
            singularity exec "$CONTAINER" bash -c \
                "_run_qc '$GENO_FILE' '$OUT_PREFIX' '$DBSNP' '$REFERENCE_GENOME' '$NUM_THREADS'"
        else
            _run_qc "$GENO_FILE" "$OUT_PREFIX" "$DBSNP" "$REFERENCE_GENOME" "$NUM_THREADS"
        fi
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Available: qc" >&2
        exit 1 ;;
esac
