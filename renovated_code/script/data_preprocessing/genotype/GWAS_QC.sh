#!/usr/bin/env bash
# ============================================================
# GWAS_QC.sh
# Mirrors: code/data_preprocessing/genotype/GWAS_QC.ipynb
#
# Steps:
#   qc_no_prune
#   qc
#   genotype_phenotype_sample_overlap
#   king
# ============================================================
set -euo pipefail

STEP="${1:?Usage: $(basename "$0") <step> [args...]}"
shift

CONTAINER=""
GENO_FILE=""
PHENO_FILE=""
SAMPLE_PARTICIPANT_LOOKUP=""
CWD="output"
NAME=""
PLINK_COMMAND=""
OUTPUT_FORMAT=""
MAKE_COMMAND=""
OUT_PREFIX=""
PRUNE_PREFIX=""
KEEP_SAMPLES=""
KEEP_VARIANTS=""
REMOVE_SAMPLES=""
EXCLUDE_VARIANTS=""
OTHER_ARGS=()
META_ONLY=false
RM_DUPS=false
TREAT_DOSAGE_MISSING=false
MAC_FILTER=0
MAF_FILTER=0
MAF_MAX_FILTER=0
MAC_MAX_FILTER=0
GENO_FILTER=0.1
MIND_FILTER=0.1
HWE_FILTER=1e-15
WINDOW=50
SHIFT=10
R2=0.1
KINSHIP=0.0625
KIN_MAF=0.01
NUM_THREADS=8
BAD_LD=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)                 CONTAINER="$2"; shift 2 ;;
        --genoFile)                  GENO_FILE="$2"; shift 2 ;;
        --phenoFile)                 PHENO_FILE="$2"; shift 2 ;;
        --sample-participant-lookup) SAMPLE_PARTICIPANT_LOOKUP="$2"; shift 2 ;;
        --cwd)                       CWD="$2"; shift 2 ;;
        --name)                      NAME="$2"; shift 2 ;;
        --plink-command)             PLINK_COMMAND="$2"; shift 2 ;;
        --output-format)             OUTPUT_FORMAT="$2"; shift 2 ;;
        --make-command)              MAKE_COMMAND="$2"; shift 2 ;;
        --out-prefix)                OUT_PREFIX="$2"; shift 2 ;;
        --prune-prefix)              PRUNE_PREFIX="$2"; shift 2 ;;
        --keep-samples)              KEEP_SAMPLES="$2"; shift 2 ;;
        --keep-variants)             KEEP_VARIANTS="$2"; shift 2 ;;
        --remove-samples)            REMOVE_SAMPLES="$2"; shift 2 ;;
        --exclude-variants)          EXCLUDE_VARIANTS="$2"; shift 2 ;;
        --other-arg)                 OTHER_ARGS+=("$2"); shift 2 ;;
        --meta-only)                 META_ONLY=true; shift ;;
        --rm-dups)                   RM_DUPS=true; shift ;;
        --treat-dosage-missing)      TREAT_DOSAGE_MISSING=true; shift ;;
        --mac-filter)                MAC_FILTER="$2"; shift 2 ;;
        --maf-filter)                MAF_FILTER="$2"; shift 2 ;;
        --maf-max-filter)            MAF_MAX_FILTER="$2"; shift 2 ;;
        --mac-max-filter)            MAC_MAX_FILTER="$2"; shift 2 ;;
        --geno-filter)               GENO_FILTER="$2"; shift 2 ;;
        --mind-filter)               MIND_FILTER="$2"; shift 2 ;;
        --hwe-filter)                HWE_FILTER="$2"; shift 2 ;;
        --window)                    WINDOW="$2"; shift 2 ;;
        --shift)                     SHIFT="$2"; shift 2 ;;
        --r2)                        R2="$2"; shift 2 ;;
        --kinship)                   KINSHIP="$2"; shift 2 ;;
        --kin-maf)                   KIN_MAF="$2"; shift 2 ;;
        --numThreads)                NUM_THREADS="$2"; shift 2 ;;
        --bad-ld)                    BAD_LD=true; shift ;;
        --dry-run)                   DRY_RUN=true; shift ;;
        *) echo "WARN: Unknown flag '$1' — ignored" >&2; shift ;;
    esac
done

[[ -z "$GENO_FILE" ]] && { echo "ERROR: --genoFile is required" >&2; exit 1; }
mkdir -p "$CWD"

INPUT_STEM="$(basename "$GENO_FILE")"
INPUT_STEM="${INPUT_STEM%.bed}"
INPUT_STEM="${INPUT_STEM%.pgen}"
INPUT_STEM="${INPUT_STEM%.fam}"
INPUT_STEM="${INPUT_STEM%.psam}"
BED_PREFIX="${GENO_FILE%.bed}"
BED_PREFIX="${BED_PREFIX%.pgen}"
BED_PREFIX="${BED_PREFIX%.fam}"
BED_PREFIX="${BED_PREFIX%.psam}"

resolve_plink_command() {
    if [[ -n "$PLINK_COMMAND" ]]; then
        case "$PLINK_COMMAND" in
            --bfile|--pfile) printf '%s\n' "$PLINK_COMMAND" ;;
            *) echo "ERROR: unsupported --plink-command '$PLINK_COMMAND'" >&2; exit 2 ;;
        esac
        return
    fi

    case "$OUTPUT_FORMAT" in
        plink1) printf '%s\n' "--bfile"; return ;;
        plink2) printf '%s\n' "--pfile"; return ;;
        "") ;;
        *) echo "ERROR: unsupported --output-format '$OUTPUT_FORMAT'" >&2; exit 2 ;;
    esac

    if [[ "$GENO_FILE" == *.pgen || -f "${BED_PREFIX}.pvar" || -f "${BED_PREFIX}.psam" ]]; then
        printf '%s\n' "--pfile"
    else
        printf '%s\n' "--bfile"
    fi
}

resolve_make_args() {
    local requested="$MAKE_COMMAND"
    if [[ -z "$requested" ]]; then
        if [[ "$META_ONLY" == "true" ]]; then
            requested="--write-snplist --write-samples"
        elif [[ "$OUTPUT_FORMAT" == "plink2" || "$(resolve_plink_command)" == "--pfile" ]]; then
            requested="--make-pgen"
        else
            requested="--make-bed"
        fi
    fi

    case "$requested" in
        --make-bed) printf '%s\n' "--make-bed" ;;
        --make-pgen) printf '%s\n' "--make-pgen" ;;
        "--write-snplist --write-samples")
            printf '%s\n' "--write-snplist"
            printf '%s\n' "--write-samples"
            ;;
        *) echo "ERROR: unsupported --make-command '$requested'" >&2; exit 2 ;;
    esac
}

quote_array_for_bash() {
    local value
    for value in "$@"; do
        printf ' %q' "$value"
    done
}

is_nonzero() {
    awk -v value="$1" 'BEGIN { exit !((value + 0) != 0) }'
}

ensure_plink2() {
    local activate_script
    activate_script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)/snakemake/dryrun/activate_local_pixi.sh"

    if command -v plink2 >/dev/null 2>&1; then
        return 0
    fi

    if [[ -r "$activate_script" ]]; then
        set +u
        # Modular SoS notebook tasks can lose the parent shell PATH even though the
        # faithful notebook flow expects plink2 to be available in-task.
        source "$activate_script" >/dev/null 2>&1 || true
        set -u
    fi

    if ! command -v plink2 >/dev/null 2>&1; then
        echo "ERROR: plink2 is not available on PATH" >&2
        exit 127
    fi
}

_qc_no_prune() {
    ensure_plink2

    local out_prefix="${CWD}/${INPUT_STEM}"
    [[ -n "$NAME" ]] && out_prefix="${out_prefix}.${NAME}"
    out_prefix="${out_prefix}.plink_qc"
    [[ -n "$KEEP_VARIANTS" && -f "$KEEP_VARIANTS" ]] && out_prefix="${out_prefix}.extracted"
    [[ -n "$OUT_PREFIX" ]] && out_prefix="$OUT_PREFIX"

    local plink_command
    plink_command="$(resolve_plink_command)"
    local args=("$plink_command" "$BED_PREFIX" --allow-extra-chr)
    local make_args=()
    mapfile -t make_args < <(resolve_make_args)
    is_nonzero "$MAF_FILTER" && args+=(--maf "$MAF_FILTER")
    is_nonzero "$MAF_MAX_FILTER" && args+=(--max-maf "$MAF_MAX_FILTER")
    is_nonzero "$MAC_FILTER" && args+=(--mac "$MAC_FILTER")
    is_nonzero "$MAC_MAX_FILTER" && args+=(--max-mac "$MAC_MAX_FILTER")
    if is_nonzero "$GENO_FILTER"; then
        args+=(--geno "$GENO_FILTER")
        [[ "$TREAT_DOSAGE_MISSING" == "true" ]] && args+=(dosage)
    fi
    is_nonzero "$HWE_FILTER" && args+=(--hwe "$HWE_FILTER")
    if is_nonzero "$MIND_FILTER"; then
        args+=(--mind "$MIND_FILTER")
        [[ "$TREAT_DOSAGE_MISSING" == "true" ]] && args+=(dosage)
    fi
    [[ -n "$KEEP_SAMPLES" && -f "$KEEP_SAMPLES" ]] && args+=(--keep "$KEEP_SAMPLES")
    [[ -n "$REMOVE_SAMPLES" && -f "$REMOVE_SAMPLES" ]] && args+=(--remove "$REMOVE_SAMPLES")
    [[ -n "$EXCLUDE_VARIANTS" && -f "$EXCLUDE_VARIANTS" ]] && args+=(--exclude "$EXCLUDE_VARIANTS")
    [[ -n "$KEEP_VARIANTS" && -f "$KEEP_VARIANTS" ]] && args+=(--extract "$KEEP_VARIANTS")
    [[ "$META_ONLY" == "true" ]] && make_args=(--write-snplist --write-samples)
    [[ "$RM_DUPS" == "true" ]] && args+=(--rm-dup force-first list)
    local other_arg
    for other_arg in "${OTHER_ARGS[@]}"; do
        if [[ ! "$other_arg" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]*$ ]]; then
            echo "ERROR: unsafe --other-arg '$other_arg'; pass PLINK flag names without leading dashes or values" >&2
            exit 2
        fi
        args+=("--${other_arg}")
    done

    plink2 \
        "${args[@]}" \
        "${make_args[@]}" \
        --out "$out_prefix" \
        --threads "$NUM_THREADS" \
        --memory 16000
}

_qc() {
    ensure_plink2

    local prune_base="${CWD}/${INPUT_STEM}"
    local pruned_out="${CWD}/${INPUT_STEM}.prune"
    [[ -n "$PRUNE_PREFIX" ]] && prune_base="$PRUNE_PREFIX"
    [[ -n "$OUT_PREFIX" ]] && pruned_out="$OUT_PREFIX"
    local plink_command
    plink_command="$(resolve_plink_command)"
    local make_args=()
    mapfile -t make_args < <(resolve_make_args)
    local prune_args=("$plink_command" "$BED_PREFIX" --allow-extra-chr --rm-dup force-first)
    [[ "$BAD_LD" == "true" ]] && prune_args+=(--bad-ld)

    plink2 \
        "${prune_args[@]}" \
        --indep-pairwise "$WINDOW" "$SHIFT" "$R2" \
        --out "$prune_base" \
        --threads "$NUM_THREADS" \
        --memory 16000

    plink2 \
        "$plink_command" "$BED_PREFIX" \
        --allow-extra-chr \
        --extract "${prune_base}.prune.in" \
        "${make_args[@]}" \
        --out "$pruned_out" \
        --threads "$NUM_THREADS" \
        --memory 16000
}

_sample_overlap() {
    [[ -z "$PHENO_FILE" ]] && { echo "ERROR: --phenoFile is required" >&2; exit 1; }
    local fam
    if [[ "$GENO_FILE" == *.fam || "$GENO_FILE" == *.psam ]]; then
        fam="$GENO_FILE"
    else
        fam="${BED_PREFIX}.fam"
    fi
    local pheno_stem
    pheno_stem="$(basename "$PHENO_FILE")"
    [[ "$pheno_stem" == *.gz ]] && pheno_stem="${pheno_stem%.gz}"
    [[ "$pheno_stem" == *.tsv ]] && pheno_stem="${pheno_stem%.tsv}"

    python3 - "$fam" "$PHENO_FILE" "$SAMPLE_PARTICIPANT_LOOKUP" \
        "${CWD}/${pheno_stem}.sample_overlap.txt" \
        "${CWD}/${pheno_stem}.sample_genotypes.txt" <<'PY'
import csv
import gzip
import sys
from pathlib import Path

fam_path = Path(sys.argv[1])
pheno_path = Path(sys.argv[2])
lookup_path = Path(sys.argv[3]) if sys.argv[3] else None
sample_overlap_path = Path(sys.argv[4])
sample_genotypes_path = Path(sys.argv[5])

geno_rows = []
with fam_path.open() as fh:
    for line in fh:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split()
        geno_rows.append((parts[0], parts[1]))

opener = gzip.open if pheno_path.suffix == ".gz" else open
with opener(pheno_path, "rt") as fh:
    header = fh.readline().rstrip("\n").split("\t")
pheno_samples = header[4:]

lookup_pairs = []
if lookup_path and lookup_path.is_file():
    with lookup_path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 2:
            raise SystemExit("sample lookup needs at least two columns")
        first = fieldnames[0]
        second = fieldnames[1]
        for row in reader:
            sample_id = row.get("participant_id") or row[first]
            genotype_id = row.get("genotype_id") or row.get("participant_id") or row[second]
            lookup_pairs.append((genotype_id, sample_id))
else:
    lookup_pairs = [(iid, iid) for _, iid in geno_rows]

geno_ids = {iid for _, iid in geno_rows}
pheno_ids = set(pheno_samples)
lookup_pairs = [(gid, sid) for gid, sid in lookup_pairs if gid in geno_ids and sid in pheno_ids]

with sample_overlap_path.open("w") as out:
    out.write("genotype_id\tsample_id\n")
    for gid, sid in lookup_pairs:
        out.write(f"{gid}\t{sid}\n")

lookup_geno_ids = {gid for gid, _ in lookup_pairs}
with sample_genotypes_path.open("w") as out:
    for fid, iid in geno_rows:
        if iid in lookup_geno_ids:
            out.write(f"{fid}\t{iid}\n")
PY
}

_king() {
    ensure_plink2

    local out_prefix="${CWD}/${INPUT_STEM}"
    [[ -n "$NAME" ]] && out_prefix="${out_prefix}.${NAME}"
    [[ -n "$OUT_PREFIX" ]] && out_prefix="$OUT_PREFIX"

    local plink_command
    plink_command="$(resolve_plink_command)"
    local args=("$plink_command" "$BED_PREFIX" --make-king-table --king-table-filter "$KINSHIP")
    [[ -n "$KEEP_SAMPLES" && -f "$KEEP_SAMPLES" ]] && args+=(--keep "$KEEP_SAMPLES")
    [[ -n "$REMOVE_SAMPLES" && -f "$REMOVE_SAMPLES" ]] && args+=(--remove "$REMOVE_SAMPLES")
    args+=(--min-af "$KIN_MAF" --max-af "$(python3 - <<PY
kin_maf = float(${KIN_MAF})
print(1 - kin_maf)
PY
)")

    plink2 \
        "${args[@]}" \
        --out "$out_prefix" \
        --threads "$NUM_THREADS" \
        --memory 16000
}

export -f resolve_plink_command resolve_make_args is_nonzero ensure_plink2 _qc_no_prune _qc _sample_overlap _king

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $(basename "$0") $STEP" >&2
    printf '    --genoFile %s\n' "$GENO_FILE" >&2
    [[ -n "$PHENO_FILE" ]] && printf '    --phenoFile %s\n' "$PHENO_FILE" >&2
    [[ -n "$SAMPLE_PARTICIPANT_LOOKUP" ]] && printf '    --sample-participant-lookup %s\n' "$SAMPLE_PARTICIPANT_LOOKUP" >&2
    printf '    --cwd %s\n' "$CWD" >&2
    [[ -n "$NAME" ]] && printf '    --name %s\n' "$NAME" >&2
    exit 0
fi

_dispatch() {
    case "$STEP" in
        qc_no_prune) _qc_no_prune ;;
        qc) _qc ;;
        genotype_phenotype_sample_overlap) _sample_overlap ;;
        king) _king ;;
        *)
            echo "ERROR: Unknown step '$STEP'. Available: qc_no_prune, qc, genotype_phenotype_sample_overlap, king" >&2
            exit 1
            ;;
    esac
}

if [[ -n "$CONTAINER" ]]; then
    singularity exec "$CONTAINER" bash -s <<EOF
$(declare -f resolve_plink_command resolve_make_args is_nonzero ensure_plink2 _qc_no_prune _qc _sample_overlap _king _dispatch)
STEP="$STEP"
GENO_FILE="$GENO_FILE"
PHENO_FILE="$PHENO_FILE"
SAMPLE_PARTICIPANT_LOOKUP="$SAMPLE_PARTICIPANT_LOOKUP"
CWD="$CWD"
NAME="$NAME"
PLINK_COMMAND="$PLINK_COMMAND"
OUTPUT_FORMAT="$OUTPUT_FORMAT"
MAKE_COMMAND="$MAKE_COMMAND"
OUT_PREFIX="$OUT_PREFIX"
PRUNE_PREFIX="$PRUNE_PREFIX"
BED_PREFIX="$BED_PREFIX"
INPUT_STEM="$INPUT_STEM"
KEEP_SAMPLES="$KEEP_SAMPLES"
KEEP_VARIANTS="$KEEP_VARIANTS"
REMOVE_SAMPLES="$REMOVE_SAMPLES"
EXCLUDE_VARIANTS="$EXCLUDE_VARIANTS"
OTHER_ARGS=($(quote_array_for_bash "${OTHER_ARGS[@]}"))
META_ONLY="$META_ONLY"
RM_DUPS="$RM_DUPS"
TREAT_DOSAGE_MISSING="$TREAT_DOSAGE_MISSING"
MAC_FILTER="$MAC_FILTER"
MAF_FILTER="$MAF_FILTER"
MAF_MAX_FILTER="$MAF_MAX_FILTER"
MAC_MAX_FILTER="$MAC_MAX_FILTER"
GENO_FILTER="$GENO_FILTER"
MIND_FILTER="$MIND_FILTER"
HWE_FILTER="$HWE_FILTER"
WINDOW="$WINDOW"
SHIFT="$SHIFT"
R2="$R2"
KINSHIP="$KINSHIP"
KIN_MAF="$KIN_MAF"
NUM_THREADS="$NUM_THREADS"
BAD_LD="$BAD_LD"
_dispatch
EOF
else
    _dispatch
fi
