#!/usr/bin/env bash
# ============================================================
# GWAS_QC.sh
# Mirrors: code/data_preprocessing/genotype/GWAS_QC.ipynb
#
# Steps:
#   qc_no_prune                     — variant/sample QC without LD pruning
#   qc                              — variant/sample QC + LD pruning
#   genotype_phenotype_sample_overlap — intersect genotype and phenotype samples
#   king                            — KING kinship + related/unrelated split
#
# Usage:
#   GWAS_QC.sh <step> [--container PATH] [--flag value ...]
#
# Interface kept identical to:
#   sos run GWAS_QC.ipynb <step> [--flag value ...]
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [--container PATH] [args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ──────────────────────────────────────────────────────────────
CONTAINER=""
GENO_FILE=""
PHENO_FILE=""
CWD="output"
NAME=""
KEEP_SAMPLES=""
KEEP_VARIANTS=""
MAC_FILTER=5
MAF_FILTER=0
GENO_FILTER=0.1
MIND_FILTER=0.1
HWE_FILTER=1e-6
WINDOW=200
SHIFT=25
R2=0.1
NUM_THREADS=8
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)        CONTAINER="$2";       shift 2 ;;
        --genoFile)         GENO_FILE="$2";       shift 2 ;;
        --phenoFile)        PHENO_FILE="$2";      shift 2 ;;
        --cwd)              CWD="$2";             shift 2 ;;
        --name)             NAME="$2";            shift 2 ;;
        --keep-samples)     KEEP_SAMPLES="$2";    shift 2 ;;
        --keep-variants)    KEEP_VARIANTS="$2";   shift 2 ;;
        --mac-filter)       MAC_FILTER="$2";      shift 2 ;;
        --maf-filter)       MAF_FILTER="$2";      shift 2 ;;
        --geno-filter)      GENO_FILTER="$2";     shift 2 ;;
        --mind-filter)      MIND_FILTER="$2";     shift 2 ;;
        --hwe-filter)       HWE_FILTER="$2";      shift 2 ;;
        --window)           WINDOW="$2";          shift 2 ;;
        --shift)            SHIFT="$2";           shift 2 ;;
        --r2)               R2="$2";              shift 2 ;;
        --numThreads)       NUM_THREADS="$2";     shift 2 ;;
        --dry-run)          DRY_RUN=true;          shift ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

[[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }
mkdir -p "$CWD"

if [[ -z "$NAME" ]]; then
    NAME="$(basename "$GENO_FILE" .bed)"
fi

BED_PREFIX="${GENO_FILE%.bed}"

# ── Step: qc_no_prune ────────────────────────────────────────────────────────
_qc_no_prune() {
    echo "=== qc_no_prune ===" >&2
    local args=""
    [[ -n "$KEEP_SAMPLES"  ]] && args="$args --keep $KEEP_SAMPLES"
    [[ -n "$KEEP_VARIANTS" ]] && args="$args --extract $KEEP_VARIANTS"
    [[ "$MAC_FILTER" != "0" ]] && args="$args --mac $MAC_FILTER"
    [[ "$MAF_FILTER" != "0" ]] && args="$args --maf $MAF_FILTER"

    plink2 \
        --bfile "$BED_PREFIX" \
        --geno "$GENO_FILTER" \
        --mind "$MIND_FILTER" \
        --hwe "$HWE_FILTER" \
        --make-bed \
        --out "${CWD}/${NAME}" \
        --threads "$NUM_THREADS" \
        --memory 16000 \
        $args
    echo "Output: ${CWD}/${NAME}.{bed,bim,fam}" >&2
}

# ── Step: qc (with LD pruning) ───────────────────────────────────────────────
_qc() {
    echo "=== qc: variant QC + LD pruning ===" >&2
    local args=""
    [[ "$MAC_FILTER" != "0" ]] && args="$args --mac $MAC_FILTER"
    [[ "$MAF_FILTER" != "0" ]] && args="$args --maf $MAF_FILTER"

    # 1. Basic QC
    plink2 \
        --bfile "$BED_PREFIX" \
        --geno "$GENO_FILTER" \
        --mind "$MIND_FILTER" \
        --hwe "$HWE_FILTER" \
        --make-bed \
        --out "${CWD}/${NAME}.plink_qc" \
        --threads "$NUM_THREADS" \
        --memory 16000 \
        $args

    # 2. LD pruning
    plink2 \
        --bfile "${CWD}/${NAME}.plink_qc" \
        --indep-pairwise "$WINDOW" "$SHIFT" "$R2" \
        --out "${CWD}/${NAME}.plink_qc.prune" \
        --threads "$NUM_THREADS" \
        --memory 16000

    # 3. Extract pruned variants
    plink2 \
        --bfile "${CWD}/${NAME}.plink_qc" \
        --extract "${CWD}/${NAME}.plink_qc.prune.in" \
        --make-bed \
        --out "${CWD}/${NAME}.plink_qc.prune" \
        --threads "$NUM_THREADS" \
        --memory 16000

    echo "Output: ${CWD}/${NAME}.plink_qc.prune.{bed,bim,fam}" >&2
    echo "Prune list: ${CWD}/${NAME}.plink_qc.prune.in" >&2
}

# ── Step: genotype_phenotype_sample_overlap ──────────────────────────────────
_sample_overlap() {
    [[ -z "$PHENO_FILE" ]] && { echo "ERROR: --phenoFile is required" >&2; exit 1; }
    echo "=== genotype_phenotype_sample_overlap ===" >&2

    # Extract genotype sample IDs from .fam
    local fam="${BED_PREFIX}.fam"
    awk '{print $1"\t"$2}' "$fam" > "${CWD}/${NAME}.genotype_samples.txt"

    # Extract phenotype sample IDs from BED.gz header (tab-delimited, cols 5+)
    local pheno_samples
    if [[ "$PHENO_FILE" == *.gz ]]; then
        pheno_samples=$(zcat "$PHENO_FILE" | head -1 | cut -f5- | tr '\t' '\n')
    else
        pheno_samples=$(head -1 "$PHENO_FILE" | cut -f5- | tr '\t' '\n')
    fi
    echo "$pheno_samples" > "${CWD}/${NAME}.phenotype_samples.txt"

    # Overlap: sample IDs present in both (FID=IID assumed for phenotype samples)
    awk 'NR==FNR{pheno[$1]=1; next} ($2 in pheno){print $1"\t"$2}' \
        "${CWD}/${NAME}.phenotype_samples.txt" \
        "${CWD}/${NAME}.genotype_samples.txt" \
        > "${CWD}/${NAME}.sample_genotypes.txt"

    # Record overlap list (just IDs)
    awk '{print $2}' "${CWD}/${NAME}.sample_genotypes.txt" \
        > "${CWD}/${NAME}.sample_overlap.txt"

    local n_overlap
    n_overlap=$(wc -l < "${CWD}/${NAME}.sample_genotypes.txt")
    echo "Overlap: $n_overlap samples" >&2
    echo "Output: ${CWD}/${NAME}.sample_genotypes.txt" >&2
    echo "Output: ${CWD}/${NAME}.sample_overlap.txt" >&2
}

# ── Step: king ───────────────────────────────────────────────────────────────
_king() {
    echo "=== king: kinship analysis ===" >&2
    local keep_arg=""
    [[ -n "$KEEP_SAMPLES" ]] && keep_arg="--keep $KEEP_SAMPLES"

    # Subset to phenotype-matched samples
    plink2 \
        --bfile "$BED_PREFIX" \
        --make-bed \
        --out "${CWD}/${NAME}.matched" \
        --threads "$NUM_THREADS" \
        --memory 16000 \
        $keep_arg

    # Run KING kinship
    king \
        --bfile "${CWD}/${NAME}.matched" \
        --kinship \
        --prefix "${CWD}/${NAME}.king" \
        --cpus "$NUM_THREADS"

    # Split into related and unrelated sets using KING's cutoff output
    # KING produces .kin0 file; use plink2 to split by kinship threshold
    plink2 \
        --bfile "${CWD}/${NAME}.matched" \
        --king-cutoff "${CWD}/${NAME}.king" 0.0884 \
        --make-bed \
        --out "${CWD}/${NAME}.unrelated" \
        --threads "$NUM_THREADS" \
        --memory 16000

    # Related = in matched but NOT in unrelated
    awk 'NR==FNR{unrep[$2]=1; next} !($2 in unrep){print $1"\t"$2}' \
        "${CWD}/${NAME}.unrelated.fam" \
        "${CWD}/${NAME}.matched.fam" \
        > "${CWD}/${NAME}.related_ids.txt"

    plink2 \
        --bfile "${CWD}/${NAME}.matched" \
        --keep "${CWD}/${NAME}.related_ids.txt" \
        --make-bed \
        --out "${CWD}/${NAME}.related" \
        --threads "$NUM_THREADS" \
        --memory 16000

    echo "Output: ${CWD}/${NAME}.unrelated.{bed,bim,fam}" >&2
    echo "Output: ${CWD}/${NAME}.related.{bed,bim,fam}" >&2
}

export -f _qc_no_prune _qc _sample_overlap _king

# ── Dry-run: print parameters and exit ───────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    [[ -n "$CONTAINER" ]] && printf '[DRY-RUN] Container: %s\n' "$CONTAINER" >&2
    echo "[DRY-RUN] Full command (copy-paste to debug):" >&2
    printf '  bash %s %s --dry-run \\\n' "$(realpath "$0" 2>/dev/null || echo "$0")" "$STEP" >&2
    [[ -n "$CONTAINER"     ]] && printf '    --container %s \\\n'     "$CONTAINER"    >&2
    printf '    --genoFile %s \\\n'       "$GENO_FILE"    >&2
    [[ -n "$PHENO_FILE"    ]] && printf '    --phenoFile %s \\\n'     "$PHENO_FILE"   >&2
    printf '    --cwd %s \\\n'            "$CWD"          >&2
    [[ -n "$NAME"          ]] && printf '    --name %s \\\n'          "$NAME"         >&2
    [[ -n "$KEEP_SAMPLES"  ]] && printf '    --keep-samples %s \\\n'  "$KEEP_SAMPLES" >&2
    [[ -n "$KEEP_VARIANTS" ]] && printf '    --keep-variants %s \\\n' "$KEEP_VARIANTS" >&2
    printf '    --mac-filter %s  --maf-filter %s \\\n' "$MAC_FILTER" "$MAF_FILTER" >&2
    printf '    --geno-filter %s --mind-filter %s --hwe-filter %s \\\n'            "$GENO_FILTER" "$MIND_FILTER" "$HWE_FILTER" >&2
    printf '    --numThreads %s\n' "$NUM_THREADS" >&2
    echo "[DRY-RUN] Input file check:" >&2
    for _f in "$GENO_FILE" "$PHENO_FILE" "$KEEP_SAMPLES" "$KEEP_VARIANTS"; do
        [[ -z "$_f" ]] && continue
        if [[ -e "$_f" ]]; then printf '  ✓ %s\n' "$_f" >&2
        else               printf '  ✗ NOT FOUND: %s\n' "$_f" >&2; fi
    done
    exit 0
fi


# ── Dispatch ─────────────────────────────────────────────────────────────────
_dispatch() {
    case "$STEP" in
        qc_no_prune)
            _qc_no_prune ;;
        qc)
            _qc ;;
        genotype_phenotype_sample_overlap)
            _sample_overlap ;;
        king)
            _king ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: qc_no_prune, qc, genotype_phenotype_sample_overlap, king" >&2
            exit 1 ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    singularity exec "$CONTAINER" bash -s << EOF
$(declare -f _qc_no_prune _qc _sample_overlap _king _dispatch)
STEP="$STEP" GENO_FILE="$GENO_FILE" PHENO_FILE="$PHENO_FILE"
CWD="$CWD" NAME="$NAME" BED_PREFIX="$BED_PREFIX"
KEEP_SAMPLES="$KEEP_SAMPLES" KEEP_VARIANTS="$KEEP_VARIANTS"
MAC_FILTER="$MAC_FILTER" MAF_FILTER="$MAF_FILTER"
GENO_FILTER="$GENO_FILTER" MIND_FILTER="$MIND_FILTER" HWE_FILTER="$HWE_FILTER"
WINDOW="$WINDOW" SHIFT="$SHIFT" R2="$R2" NUM_THREADS="$NUM_THREADS"
_dispatch
EOF
else
    _dispatch
fi
