#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_local_pixi_common.sh"

xqtl_activate_local_pixi

status=0
check_home="$(mktemp -d "${TMPDIR:-/tmp}/xqtl-local-pixi-check.XXXXXX")"
trap 'rm -rf "${check_home}"' EXIT

pass() {
    printf 'PASS  %s: %s\n' "$1" "$2"
}

warn() {
    printf 'WARN  %s: %s\n' "$1" "$2"
}

fail() {
    printf 'FAIL  %s: %s\n' "$1" "$2" >&2
    status=1
}

first_line() {
    printf '%s\n' "$1" | head -n 1
}

check_command_path() {
    local name="$1"
    local resolved

    resolved="$(command -v "${name}" 2>/dev/null || true)"
    if [[ -z "${resolved}" ]]; then
        fail "${name}-path" "command not found"
        return
    fi

    case "${resolved}" in
        "${XQTL_LOCAL_PIXI_HELPER_BIN}"/*|"${PIXI_HOME}"/*)
            pass "${name}-path" "${resolved}"
            ;;
        *)
            warn "${name}-path" "resolved outside local Pixi layer: ${resolved}"
            ;;
    esac
}

run_check() {
    local label="$1"
    shift
    local output

    if output="$("$@" 2>&1)"; then
        pass "${label}" "$(first_line "${output:-ok}")"
    else
        fail "${label}" "$(first_line "${output}")"
    fi
}

run_check_match() {
    local label="$1"
    local needle="$2"
    shift 2
    local output
    local cmd_status

    set +e
    output="$("$@" 2>&1)"
    cmd_status=$?
    set -e

    if printf '%s\n' "${output}" | grep -Fq "${needle}"; then
        pass "${label}" "$(first_line "${output}")"
    else
        fail "${label}" "$(first_line "${output:-exit ${cmd_status}}")"
    fi
}

run_r_package_check() {
    local package_name="$1"
    run_check "r-package-${package_name}" env HOME="${check_home}" Rscript -e "ok <- requireNamespace('${package_name}', quietly=TRUE); cat(if (ok) 'present' else 'missing'); quit(status = if (ok) 0 else 1)"
}

printf 'PIXI_HOME=%s\n' "${PIXI_HOME}"
printf 'PATH helper=%s\n' "${XQTL_LOCAL_PIXI_HELPER_BIN}"

for cmd in python R Rscript sos snakemake bcftools fastqc rnaseqc plink plink2 king samtools STAR picard multiqc trimmomatic; do
    check_command_path "${cmd}"
done

run_check "python-version" python --version
run_check "r-version" R --version
run_check "rscript-smoke" env HOME="${check_home}" Rscript -e "cat('rscript-ok')"
run_check "sos-version" env HOME="${check_home}" PYTHONWARNINGS=ignore "$(xqtl_local_python_bin)" -c "import sos; print(sos.__version__)"
run_check "snakemake-version" snakemake --version
run_check "python-imports" python -c "import importlib.util, numpy, pandas; assert importlib.util.find_spec('tensorqtl') is not None; print('python-imports-ok')"
run_check "bcftools-version" bcftools --version
run_check "fastqc-version" fastqc --version
run_check_match "rnaseqc-help" "RNASeQC" rnaseqc --help
run_check "plink-version" plink --version
run_check "plink2-version" plink2 --version
run_check_match "king-version" "KING " king --version
run_check "samtools-version" samtools --version
run_check "star-version" STAR --version
run_check_match "picard-version" "Version:" picard MarkDuplicates --version
run_r_package_check "flashpcaR"
run_r_package_check "peer"

if [[ "${status}" -ne 0 ]]; then
    printf '\nLocal Pixi compatibility layer is active, but one or more dry-run dependencies are still missing or stale.\n' >&2
    exit "${status}"
fi

printf '\nAll checked commands resolved through the local Pixi compatibility layer.\n'
