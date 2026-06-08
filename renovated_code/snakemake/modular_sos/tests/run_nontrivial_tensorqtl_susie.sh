#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_nontrivial_tensorqtl_susie.sh [options]

Options:
  --data-tar PATH       Fixture tarball. Default: data/modular_sos_nontrivial_tensorqtl_susie.tar.gz.
  --workdir PATH        Output directory. Default: renovated_code/snakemake/tmp/modular_sos_tests/nontrivial_<timestamp>.
  --run-tag TAG         Label used when --workdir is omitted.
  --num-threads N       SoS notebook thread count. Default: 4.
  --no-pixi             Do not source the local pixi compatibility layer.
  -h, --help            Show this help.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
SNAKEMAKE_DIR="${ROOT}/renovated_code/snakemake"
MODULAR_SOS_NOTEBOOKS="${ROOT}/renovated_code/notebook/modular_sos"
LEGACY_TENSORQTL="${ROOT}/code/association_scan/TensorQTL/TensorQTL.ipynb"
LEGACY_MNM="${ROOT}/code/mnm_analysis/mnm_methods/mnm_regression.ipynb"
DATA_TAR="${SCRIPT_DIR}/data/modular_sos_nontrivial_tensorqtl_susie.tar.gz"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
WORKDIR=""
NUM_THREADS=4
USE_PIXI=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-tar)
            [[ $# -ge 2 ]] || die "--data-tar requires a path"
            DATA_TAR="$2"
            shift 2
            ;;
        --workdir)
            [[ $# -ge 2 ]] || die "--workdir requires a path"
            WORKDIR="$2"
            shift 2
            ;;
        --run-tag)
            [[ $# -ge 2 ]] || die "--run-tag requires a value"
            RUN_TAG="$2"
            shift 2
            ;;
        --num-threads)
            [[ $# -ge 2 ]] || die "--num-threads requires a value"
            NUM_THREADS="$2"
            shift 2
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

case "${NUM_THREADS}" in
    ''|*[!0-9]*) die "--num-threads must be a positive integer" ;;
esac
[[ "${NUM_THREADS}" -gt 0 ]] || die "--num-threads must be a positive integer"
[[ -f "${DATA_TAR}" ]] || die "fixture tarball not found: ${DATA_TAR}"

WORKDIR="${WORKDIR:-${SNAKEMAKE_DIR}/tmp/modular_sos_tests/nontrivial_${RUN_TAG}}"
[[ ! -e "${WORKDIR}" ]] || die "workdir already exists: ${WORKDIR}"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/reports" "${WORKDIR}/old" "${WORKDIR}/modular_sos" \
    "${WORKDIR}/runtime_home" "${WORKDIR}/runtime_root"

export XQTL_LOCAL_RUNTIME_HOME="${WORKDIR}/runtime_home"
export XQTL_LOCAL_RUNTIME_ROOT="${WORKDIR}/runtime_root"
if [[ "${USE_PIXI}" -eq 1 && -f "${SNAKEMAKE_DIR}/dryrun/activate_local_pixi.sh" ]]; then
    source "${SNAKEMAKE_DIR}/dryrun/activate_local_pixi.sh" >/dev/null
fi

SOS_BIN="${SOS_BIN:-sos}"
if [[ -x "${SNAKEMAKE_DIR}/dryrun/bin/sos" ]]; then
    SOS_BIN="${SNAKEMAKE_DIR}/dryrun/bin/sos"
fi

tar -xzf "${DATA_TAR}" -C "${WORKDIR}"
FIXTURE="${WORKDIR}/nontrivial_tensorqtl_susie"
[[ -d "${FIXTURE}" ]] || die "fixture directory was not found after unpacking ${DATA_TAR}"

TQTL_DIR="${FIXTURE}/tensorqtl"
SUSIE_DIR="${FIXTURE}/susie"
printf '#id\t#path\nchr1\t%s\n' "${SUSIE_DIR}/fm_genotype.bed" > "${SUSIE_DIR}/genotype_by_chrom_files_normalized.txt"

printf 'Notebook compare workdir: %s\n' "${WORKDIR}"
printf 'Fixture: %s\n' "${FIXTURE}"
printf 'SoS: %s\n' "${SOS_BIN}"

"${SOS_BIN}" run "${LEGACY_TENSORQTL}" cis \
    --cwd "${WORKDIR}/old/tensorqtl_cis" \
    --genotype-file "${TQTL_DIR}/AC.unrelated.plink_qc.prune.bed" \
    --phenotype-file "${TQTL_DIR}/modular_sos_hg_synthetic.expression.bed.gz" \
    --covariate-file "${TQTL_DIR}/modular_sos_hg_covariates.cov.AC.related.plink_qc.extracted.pca.projected.gz" \
    --chromosome 1 \
    --MAC 0 \
    --window 1000000 \
    --numThreads "${NUM_THREADS}" 2>&1 | tee "${WORKDIR}/logs/tensorqtl.old.log"

XQTL_PATCH_TENSORQTL_SORT=1 \
PYTHONPATH="${SNAKEMAKE_DIR}/compat/python${PYTHONPATH:+:${PYTHONPATH}}" \
"${SOS_BIN}" run "${MODULAR_SOS_NOTEBOOKS}/TensorQTL.ipynb" cis \
    --cwd "${WORKDIR}/modular_sos/tensorqtl_cis" \
    --genotype-file "${TQTL_DIR}/AC.unrelated.plink_qc.prune.bed" \
    --phenotype-file "${TQTL_DIR}/modular_sos_hg_synthetic.expression.bed.gz" \
    --covariate-file "${TQTL_DIR}/modular_sos_hg_covariates.cov.AC.related.plink_qc.extracted.pca.projected.gz" \
    --chromosome 1 \
    --MAC 0 \
    --window 1000000 \
    --numThreads "${NUM_THREADS}" 2>&1 | tee "${WORKDIR}/logs/tensorqtl.modular_sos.log"

TENSOR_COMPARE="${WORKDIR}/reports/tensorqtl_cis.compare.tsv"
printf 'file\told_md5\tmodular_sos_md5\tstatus\n' > "${TENSOR_COMPARE}"
for file in \
    modular_sos_hg_synthetic.expression.cis_qtl_pairs.1.parquet \
    modular_sos_hg_synthetic.expression_chr1.cis_qtl.pairs.tsv.gz \
    modular_sos_hg_synthetic.expression_chr1.cis_qtl.pairs.tsv.gz.tbi \
    modular_sos_hg_synthetic.expression_chr1.cis_qtl.regional.tsv.gz \
    modular_sos_hg_synthetic.expression_chr1.cis_qtl.regional.tsv.gz.tbi \
    modular_sos_hg_synthetic.expression_chr1.cis_qtl_regional_significance.summary.txt \
    modular_sos_hg_synthetic.expression_chr1.cis_qtl_regional_significance.tsv.gz
do
    old_file="${WORKDIR}/old/tensorqtl_cis/${file}"
    modular_sos_file="${WORKDIR}/modular_sos/tensorqtl_cis/${file}"
    [[ -f "${old_file}" ]] || die "missing old TensorQTL output: ${old_file}"
    [[ -f "${modular_sos_file}" ]] || die "missing Modular SoS TensorQTL output: ${modular_sos_file}"
    old_md5="$(md5sum "${old_file}" | awk '{print $1}')"
    modular_sos_md5="$(md5sum "${modular_sos_file}" | awk '{print $1}')"
    status="match"
    [[ "${old_md5}" == "${modular_sos_md5}" ]] || status="diff"
    printf '%s\t%s\t%s\t%s\n' "${file}" "${old_md5}" "${modular_sos_md5}" "${status}" >> "${TENSOR_COMPARE}"
done
awk 'NR > 1 && $4 != "match" { bad = 1 } END { exit bad }' "${TENSOR_COMPARE}"

python3 - "${WORKDIR}/modular_sos/tensorqtl_cis/modular_sos_hg_synthetic.expression_chr1.cis_qtl.regional.tsv.gz" <<'PY'
import csv
import gzip
import math
import sys

path = sys.argv[1]
with gzip.open(path, "rt", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        values = [row.get("p_nominal"), row.get("p_perm"), row.get("p_beta")]
        if all(v not in {None, "", "NA", "nan"} for v in values):
            nums = [float(v) for v in values]
            if all(math.isfinite(v) for v in nums):
                print(
                    "TensorQTL nontrivial row: "
                    f"trait={row.get('molecular_trait_object_id')} "
                    f"variant={row.get('variant_id')} "
                    f"p_perm={row.get('p_perm')}"
                )
                sys.exit(0)
raise SystemExit("TensorQTL regional output has no finite p_nominal/p_perm/p_beta row")
PY

"${SOS_BIN}" run "${LEGACY_MNM}" susie_twas \
    --cwd "${WORKDIR}/old/fm_susie_twas" \
    --name fmtest \
    --genoFile "${SUSIE_DIR}/genotype_by_chrom_files_normalized.txt" \
    --phenoFile "${SUSIE_DIR}/gene_chr1_1.mnm.bed.gz" \
    --covFile "${SUSIE_DIR}/cov.gz" \
    --region-name gene_chr1_1 \
    --cis-window 500000 \
    --skip-analysis-pip-cutoff 0.0 \
    --numThreads "${NUM_THREADS}" 2>&1 | tee "${WORKDIR}/logs/susie_twas.old.log"

"${SOS_BIN}" run "${MODULAR_SOS_NOTEBOOKS}/mnm_regression.ipynb" susie_twas \
    --cwd "${WORKDIR}/modular_sos/fm_susie_twas" \
    --name fmtest \
    --genoFile "${SUSIE_DIR}/genotype_by_chrom_files_normalized.txt" \
    --phenoFile "${SUSIE_DIR}/gene_chr1_1.mnm.bed.gz" \
    --covFile "${SUSIE_DIR}/cov.gz" \
    --region-name gene_chr1_1 \
    --cis-window 500000 \
    --skip-analysis-pip-cutoff 0.0 \
    --numThreads "${NUM_THREADS}" \
    --renovated-code-dir "${ROOT}/renovated_code/script" 2>&1 | tee "${WORKDIR}/logs/susie_twas.modular_sos.log"

Rscript "${SCRIPT_DIR}/compare_susie_heads.R" \
    "${WORKDIR}/old/fm_susie_twas/fine_mapping/fmtest.chr1_gene_chr1_1.univariate_bvsr.rds" \
    "${WORKDIR}/modular_sos/fm_susie_twas/fine_mapping/fmtest.chr1_gene_chr1_1.univariate_bvsr.rds" \
    "${WORKDIR}/old/fm_susie_twas/twas_weights/fmtest.chr1_gene_chr1_1.univariate_twas_weights.rds" \
    "${WORKDIR}/modular_sos/fm_susie_twas/twas_weights/fmtest.chr1_gene_chr1_1.univariate_twas_weights.rds" \
    "${WORKDIR}/reports/fm_susie_twas.compare.tsv" \
    "${WORKDIR}/reports/fm_susie_twas.head_report.txt"

printf 'TensorQTL compare: %s\n' "${TENSOR_COMPARE}"
printf 'SuSiE/TWAS compare: %s\n' "${WORKDIR}/reports/fm_susie_twas.compare.tsv"
printf 'SuSiE/TWAS head report: %s\n' "${WORKDIR}/reports/fm_susie_twas.head_report.txt"
