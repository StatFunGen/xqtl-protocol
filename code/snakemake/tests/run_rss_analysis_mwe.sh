#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_rss_analysis_mwe.sh [options]

Options:
  --output-dir PATH       Output directory. Default: code/snakemake/tmp/xqtl_tests/rss_analysis_mwe.
  --regen-fixture         Regenerate the RSS toy fixture under code/snakemake/tests/data/rss_analysis_mwe.
  --skip-fixture          Do not generate missing fixture files.
  --python PATH           Python executable used for `python -m sos`. Default: ${PYTHON:-python}.
  --rscript PATH          Rscript executable. Default: ${RSCRIPT:-Rscript}.
  -h, --help              Show this help.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
RSCRIPT_BIN="${RSCRIPT:-Rscript}"
OUTPUT_DIR="code/snakemake/tmp/xqtl_tests/rss_analysis_mwe"
REGEN_FIXTURE=0
SKIP_FIXTURE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { printf 'error: --output-dir requires a value\n' >&2; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --regen-fixture)
            REGEN_FIXTURE=1
            shift
            ;;
        --skip-fixture)
            SKIP_FIXTURE=1
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || { printf 'error: --python requires a value\n' >&2; exit 1; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --rscript)
            [[ $# -ge 2 ]] || { printf 'error: --rscript requires a value\n' >&2; exit 1; }
            RSCRIPT_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cd "${REPO_ROOT}"

rscript_resolved="$(command -v "${RSCRIPT_BIN}")"
export PATH="$(dirname -- "${rscript_resolved}"):${PATH}"

fixture_files=(
    "code/snakemake/tests/data/rss_analysis_mwe/ld_reference/protocol_example.ld_meta_file.tsv"
    "code/snakemake/tests/data/rss_analysis_mwe/ld_reference/protocol_example.chr22_49355984_50799822.cor.xz"
    "code/snakemake/tests/data/rss_analysis_mwe/ld_reference/protocol_example.chr22_49355984_50799822.bim"
    "code/snakemake/tests/data/rss_analysis_mwe/rss_analysis/protocol_example.gwas_meta_data.tsv"
    "code/snakemake/tests/data/rss_analysis_mwe/rss_analysis/protocol_example.column_mapping.txt"
    "code/snakemake/tests/data/rss_analysis_mwe/rss_analysis/protocol_example.gwas_sumstats.chr22.tsv.gz"
    "code/snakemake/tests/data/rss_analysis_mwe/rss_analysis/protocol_example.gwas_sumstats.chr22.tsv.gz.tbi"
)

missing_fixture=0
for fixture in "${fixture_files[@]}"; do
    if [[ ! -s "${fixture}" ]]; then
        missing_fixture=1
    fi
done

if [[ "${REGEN_FIXTURE}" == 1 || ( "${missing_fixture}" == 1 && "${SKIP_FIXTURE}" == 0 ) ]]; then
    "${RSCRIPT_BIN}" code/snakemake/tests/generate_rss_mwe_fixture.R "${REPO_ROOT}"
fi

mkdir -p "${OUTPUT_DIR}"

common_args=(
    --ld-meta-data code/snakemake/tests/data/rss_analysis_mwe/ld_reference/protocol_example.ld_meta_file.tsv
    --gwas-meta-data code/snakemake/tests/data/rss_analysis_mwe/rss_analysis/protocol_example.gwas_meta_data.tsv
    --region-name 22:49355984-50799822
    --cwd "${OUTPUT_DIR}"
)

"${PYTHON_BIN}" -m sos run pipeline/rss_analysis.ipynb get_analysis_regions -s force "${common_args[@]}"

"${PYTHON_BIN}" -m sos run pipeline/rss_analysis.ipynb univariate_rss -s force \
    "${common_args[@]}" \
    --qc-method slalom \
    --impute \
    --finemapping-method susie_rss \
    --skip-analysis-pip-cutoff 0

rds_path="${OUTPUT_DIR}/univariate_rss/SLALOM_RAISS_imputed.chr22_49355984_50799822.univariate_susie_rss.rds"

"${PYTHON_BIN}" -m sos run pipeline/rss_analysis.ipynb univariate_plot -s force \
    "${common_args[@]}" \
    --input "${rds_path}"

png_path="${OUTPUT_DIR}/SLALOM_RAISS_imputed.chr22_49355984_50799822.univariate_susie_rss.png"

"${RSCRIPT_BIN}" code/snakemake/tests/check_rss_mwe_output.R \
    "${rds_path}" \
    "${png_path}" \
    "chr22_49355984_50799822"
