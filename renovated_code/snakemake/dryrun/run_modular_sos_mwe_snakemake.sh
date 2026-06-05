#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_modular_sos_mwe_snakemake.sh [options]

Options:
  --mwe-data PATH          MWE data directory, .tar.gz/.tgz, or zip file. Defaults to ../mwe_data.
  --run-tag TAG           Run label under renovated_code/snakemake/tmp/modular_sos_mwe.
  --cores N               Snakemake cores. Default: 4.
  --target TARGET         Snakemake target. Default: all. Use xqtl_core to skip plots.
  --snakemake-dry-run     Build the DAG without executing jobs.
  --no-pixi               Do not source the local pixi compatibility layer.
  -h, --help              Show this help.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_MWE_DATA="${ROOT}/../mwe_data"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
CORES=4
TARGET="all"
MWE_DATA=""
USE_PIXI=1
SNAKEMAKE_DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mwe-data)
            [[ $# -ge 2 ]] || die "--mwe-data requires a path"
            MWE_DATA="$2"
            shift 2
            ;;
        --run-tag)
            [[ $# -ge 2 ]] || die "--run-tag requires a value"
            RUN_TAG="$2"
            shift 2
            ;;
        --cores)
            [[ $# -ge 2 ]] || die "--cores requires a value"
            CORES="$2"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || die "--target requires a value"
            TARGET="$2"
            shift 2
            ;;
        --snakemake-dry-run)
            SNAKEMAKE_DRY_RUN=1
            shift
            ;;
        --no-pixi)
            USE_PIXI=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

case "${CORES}" in
    ''|*[!0-9]*) die "--cores must be a positive integer" ;;
esac
[[ "${CORES}" -gt 0 ]] || die "--cores must be a positive integer"

SNAKEMAKE_BIN="${SNAKEMAKE_BIN:-snakemake}"
SOS_BIN="${SOS_BIN:-sos}"
if [[ -x "${SCRIPT_DIR}/bin/sos" ]]; then
    SOS_BIN="${SCRIPT_DIR}/bin/sos"
fi

BASE_DIR="${ROOT}/renovated_code/snakemake/tmp/modular_sos_mwe"
RUN_DIR="${BASE_DIR}/${RUN_TAG}"
INPUT_DIR="${RUN_DIR}/inputs"
OUTPUT_DIR="${RUN_DIR}/output"
LOG_DIR="${RUN_DIR}/logs"
CONFIG="${RUN_DIR}/modular_sos_mwe.config.yaml"
SNAKEFILE="${ROOT}/renovated_code/snakemake/modular_sos/Snakefile"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_modular_sos_mwe_inputs.sh"

[[ ! -e "${RUN_DIR}" ]] || die "run directory already exists: ${RUN_DIR}"
mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}" "${RUN_DIR}/runtime_home" "${RUN_DIR}/runtime_root"

MWE_DATA="${MWE_DATA:-${DEFAULT_MWE_DATA}}"
if [[ -d "${MWE_DATA}" ]]; then
    MWE_DIR="$(cd -- "${MWE_DATA}" && pwd)"
elif [[ -f "${MWE_DATA}" ]]; then
    case "${MWE_DATA}" in
        *.tar.gz|*.tgz)
            UNPACK_DIR="${RUN_DIR}/mwe_data_unpacked"
            mkdir -p "${UNPACK_DIR}"
            tar -xzf "${MWE_DATA}" -C "${UNPACK_DIR}"
            marker="$(find "${UNPACK_DIR}" -name AC_sample_fastq.list -print -quit)"
            [[ -n "${marker}" ]] || die "could not find AC_sample_fastq.list inside ${MWE_DATA}"
            MWE_DIR="$(cd -- "$(dirname -- "${marker}")" && pwd)"
            ;;
        *.zip)
            command -v unzip >/dev/null 2>&1 || die "unzip is required for zip input"
            UNPACK_DIR="${RUN_DIR}/mwe_data_unpacked"
            mkdir -p "${UNPACK_DIR}"
            unzip -q "${MWE_DATA}" -d "${UNPACK_DIR}"
            marker="$(find "${UNPACK_DIR}" -name AC_sample_fastq.list -print -quit)"
            [[ -n "${marker}" ]] || die "could not find AC_sample_fastq.list inside ${MWE_DATA}"
            MWE_DIR="$(cd -- "$(dirname -- "${marker}")" && pwd)"
            ;;
        *)
            die "--mwe-data must be a directory, .tar.gz/.tgz, or .zip file"
            ;;
    esac
else
    die "MWE data path does not exist: ${MWE_DATA}"
fi

if [[ "${USE_PIXI}" -eq 1 && -f "${SCRIPT_DIR}/activate_local_pixi.sh" ]]; then
    export XQTL_LOCAL_RUNTIME_HOME="${RUN_DIR}/runtime_home"
    export XQTL_LOCAL_RUNTIME_ROOT="${RUN_DIR}/runtime_root"
    source "${SCRIPT_DIR}/activate_local_pixi.sh" >/dev/null
fi

MWE_DIR="${MWE_DIR}" XQTL_MODULAR_SOS_SKIP_PIXI=1 "${PREPARE_SCRIPT}" "${INPUT_DIR}"

cat > "${CONFIG}" <<EOF
cwd: "${OUTPUT_DIR}"
dry_run: false
pipeline_dir: "${ROOT}/pipeline"
renovated_code_dir: "${ROOT}/renovated_code/script"
sos_bin: "${SOS_BIN}"
minerva: ""
runtime_home: "${RUN_DIR}/runtime_home"
runtime_root: "${RUN_DIR}/runtime_root"
start_from: "fastq"

sos:
  defaults:
    queue: "local"
    J: 1
    job_size: 1
  force: false
  local_rules:
    - fastqc
    - rnaseqc_call
    - bulk_expression_qc
    - bulk_expression_normalization
    - vcf_qc
    - vcf_to_plink
    - merge_plink
    - plink_qc
    - genotype_by_chrom
    - sample_match
    - king_kinship
    - unrelated_qc
    - related_qc
    - flashpca
    - project_samples
    - merge_pca_covariate
    - marchenko_pc
    - peer_factors
    - tensorqtl_cis
    - susie_twas

themes:
  - name: "AC"
    fastq_list: "${INPUT_DIR}/AC_sample_fastq.list"
    bam_list: "${INPUT_DIR}/AC_bam_list.normalized.txt"
    data_dir: "${INPUT_DIR}/fastq"
    paired_end: true
    phenotype_bed: "${INPUT_DIR}/modular_sos_hg_synthetic.expression.bed.gz"
    raw_phenotype_file: ""
    phenotype_id_column: "gene_id"
    molecular_trait_type: "gene"
    sample_participant_lookup: "${INPUT_DIR}/sample_participant_lookup.tsv"
    covariate_file: "${INPUT_DIR}/modular_sos_hg_covariates.cov.gz"
    phenotype_group: ""

genotype:
  vcf_files:
    - "${INPUT_DIR}/genotype.vcf.gz"
  plink_bed: ""

reference:
  fasta: "${INPUT_DIR}/genotype_reference.fa"
  dbsnp: "${INPUT_DIR}/dbsnp.variants.gz"
  star_index: ""
  gtf_ercc: "${INPUT_DIR}/rnaseqc_compatible.gtf"
  gtf_collapsed: "${INPUT_DIR}/rnaseqc_compatible.gtf"
  rsem_index: ""
  adapters: "TruSeq3-PE.fa"

containers:
  bioinfo: "oras://ghcr.io/statfungen/bioinfo_apptainer:latest"
  rnaquant: "oras://ghcr.io/statfungen/rnaseq_apptainer:latest"
  tensorqtl: "oras://ghcr.io/statfungen/tensorqtl_apptainer:latest"
  peer: "oras://ghcr.io/statfungen/peer_apptainer:latest"
  pcatools: "oras://ghcr.io/statfungen/pcatools_apptainer:latest"
  flashpca: "oras://ghcr.io/statfungen/flashpcaR_apptainer:latest"
  susie: "oras://ghcr.io/statfungen/stephenslab_apptainer:latest"

genotype_qc:
  mac_filter: 0
  maf_filter: 0
  geno_filter: 0.5
  mind_filter: 0.5
  hwe_filter: 0
  kinship: 0.0625
  ld_window: 200
  ld_shift: 25
  ld_r2: 0.1
  bad_ld: true
  gt_only_vcf_qc: true

pca:
  n_pcs: 1
  pve_threshold: 0.7
  maha_k: 1
  maha_prob: 0.997

rnaseq_qc:
  low_expr_tpm: 0
  low_expr_tpm_percent: 0
  rle_filter_percent: 0
  ds_filter_percent: 0

normalization:
  method: "tmm_cpm_voom"
  tpm_threshold: 0
  count_threshold: 0
  sample_frac_threshold: 0
  quantile_normalize: true

phenotype_preprocessing:
  run_imputation: false
  impute_method: "gEBMF"
  num_factor: 60
  qc_prior_to_impute: true

covariate:
  n_pcs: 2
  tol_cov: 0.3
  mean_impute: true

hidden_factors:
  method: "Marchenko_PC"
  n_factors: 0
  peer_iterations: 1000
  peer_convergence: "fast"

association:
  cis_window: 1000000
  mac_threshold: 0
  maf_threshold: 0
  pvalue_cutoff: "5e-8"

finemapping:
  L: 2
  max_L: 4
  pip_cutoff: 0.025
  coverage:
    - 0.95
    - 0.70
    - 0.50
  maf: 0
  rcond: 0.01
  small_sample_correction: false

chromosomes:
  - "chr1"
  - "chr2"
  - "chr22"

resources:
  default:
    threads: 1
    mem_mb: 4000
    runtime: 120
  high_mem:
    threads: 1
    mem_mb: 8000
    runtime: 120
  rna_calling:
    threads: 1
    mem_mb: 8000
    runtime: 120
  genotype_qc:
    threads: 1
    mem_mb: 4000
    runtime: 120
  pca:
    threads: 1
    mem_mb: 4000
    runtime: 120
  hidden_factors:
    threads: 1
    mem_mb: 4000
    runtime: 120
  tensorqtl:
    threads: 1
    mem_mb: 8000
    runtime: 120
  finemapping:
    threads: 1
    mem_mb: 8000
    runtime: 120
EOF

printf 'Modular SoS MWE run directory: %s\n' "${RUN_DIR}"
printf 'MWE data source: %s\n' "${MWE_DIR}"
printf 'Config: %s\n' "${CONFIG}"
printf 'Target: %s\n' "${TARGET}"

snakemake_args=(
    --snakefile "${SNAKEFILE}"
    --configfile "${CONFIG}"
    --cores "${CORES}"
    --printshellcmds
)

if [[ "${SNAKEMAKE_DRY_RUN}" -eq 1 ]]; then
    snakemake_args+=(--dry-run)
fi

snakemake_args+=("${TARGET}")

set -o pipefail
"${SNAKEMAKE_BIN}" "${snakemake_args[@]}" 2>&1 | tee "${LOG_DIR}/snakemake.${TARGET}.log"
