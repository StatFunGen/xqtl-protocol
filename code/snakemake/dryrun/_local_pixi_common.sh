#!/usr/bin/env bash

xqtl_local_pixi_support_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
xqtl_local_pixi_repo_root="$(cd -- "${xqtl_local_pixi_support_dir}/../../.." && pwd)"
xqtl_local_pixi_workspace_root="$(cd -- "${xqtl_local_pixi_repo_root}/.." && pwd)"
xqtl_local_pixi_default_home="${xqtl_local_pixi_workspace_root}/mwe_data/.pixi"
xqtl_local_runtime_home_default="${xqtl_local_pixi_repo_root}"
xqtl_local_runtime_root_default="${xqtl_local_pixi_repo_root}/.xqtl-runtime"
xqtl_local_pixi_helper_bin="${xqtl_local_pixi_support_dir}/bin"
xqtl_local_pixi_compat_python="${xqtl_local_pixi_repo_root}/renovated_code/snakemake/compat/python"
xqtl_local_pixi_preferred_envs=(
    python
    snakemake
    r-base
    bcftools
    fastqc
    rnaseqc
    plink
    plink2
    king
    samtools
    star
    picard
    multiqc
    trimmomatic
    snpsift
    tensorqtl
)

xqtl_local_pixi_die() {
    printf 'error: %s\n' "$*" >&2
    return 1
}

xqtl_local_pixi_mkdirs() {
    local dir

    for dir in "$@"; do
        mkdir -p "${dir}" || xqtl_local_pixi_die "failed to create directory: ${dir}"
    done
}

xqtl_local_pixi_configure() {
    export PIXI_HOME="${PIXI_HOME:-${xqtl_local_pixi_default_home}}"
    export RATTLER_CACHE_DIR="${RATTLER_CACHE_DIR:-${PIXI_HOME}/cache}"

    [[ -d "${PIXI_HOME}" ]] || xqtl_local_pixi_die "PIXI_HOME does not exist: ${PIXI_HOME}"
    [[ -d "${PIXI_HOME}/envs" ]] || xqtl_local_pixi_die "missing env directory: ${PIXI_HOME}/envs"
}

xqtl_local_pixi_collect_prefixes() {
    local preferred_env
    local env_dir

    printf '%s\n' "${xqtl_local_pixi_helper_bin}"

    for preferred_env in "${xqtl_local_pixi_preferred_envs[@]}"; do
        env_dir="${PIXI_HOME}/envs/${preferred_env}"
        [[ -d "${env_dir}/bin" ]] && printf '%s\n' "${env_dir}/bin"
        [[ -d "${env_dir}/lib/jvm/bin" ]] && printf '%s\n' "${env_dir}/lib/jvm/bin"
    done

    while IFS= read -r env_dir; do
        for preferred_env in "${xqtl_local_pixi_preferred_envs[@]}"; do
            [[ "${env_dir}" != "${PIXI_HOME}/envs/${preferred_env}" ]] || continue 2
        done

        [[ -d "${env_dir}/bin" ]] && printf '%s\n' "${env_dir}/bin"
        [[ -d "${env_dir}/lib/jvm/bin" ]] && printf '%s\n' "${env_dir}/lib/jvm/bin"
    done < <(find "${PIXI_HOME}/envs" -mindepth 1 -maxdepth 1 -type d | sort)
}

xqtl_local_pixi_prepend_paths() {
    local entry
    local -a current_parts combined_parts
    local -A seen=()

    while IFS= read -r entry; do
        [[ -n "${entry}" && -d "${entry}" ]] || continue
        if [[ -z "${seen["${entry}"]+x}" ]]; then
            combined_parts+=("${entry}")
            seen["${entry}"]=1
        fi
    done < <(xqtl_local_pixi_collect_prefixes)

    IFS=':' read -r -a current_parts <<< "${PATH:-}"
    for entry in "${current_parts[@]}"; do
        [[ -n "${entry}" ]] || continue
        if [[ -z "${seen["${entry}"]+x}" ]]; then
            combined_parts+=("${entry}")
            seen["${entry}"]=1
        fi
    done

    PATH=""
    for entry in "${combined_parts[@]}"; do
        PATH="${PATH:+${PATH}:}${entry}"
    done
    export PATH
}

xqtl_activate_local_pixi() {
    xqtl_local_pixi_configure || return 1
    xqtl_local_pixi_prepend_paths
    xqtl_local_export_r_runtime
    if [[ -d "${xqtl_local_pixi_compat_python}" ]]; then
        case ":${PYTHONPATH:-}:" in
            *":${xqtl_local_pixi_compat_python}:"*) ;;
            *) export PYTHONPATH="${xqtl_local_pixi_compat_python}${PYTHONPATH:+:${PYTHONPATH}}" ;;
        esac
    fi
    if [[ -z "${BCFTOOLS_PLUGINS:-}" && -d "${PIXI_HOME}/envs/bcftools/libexec/bcftools" ]]; then
        export BCFTOOLS_PLUGINS="${PIXI_HOME}/envs/bcftools/libexec/bcftools"
    fi
    export XQTL_LOCAL_PIXI_ACTIVE_HOME="${PIXI_HOME}"
    export XQTL_LOCAL_PIXI_REPO_ROOT="${xqtl_local_pixi_repo_root}"
    export XQTL_LOCAL_PIXI_HELPER_BIN="${xqtl_local_pixi_helper_bin}"
    export XQTL_LOCAL_PIXI_ACTIVATED=1
}

xqtl_activate_repo_runtime_home() {
    local runtime_home
    local runtime_root

    runtime_home="${XQTL_LOCAL_RUNTIME_HOME:-${xqtl_local_runtime_home_default}}"
    runtime_root="${XQTL_LOCAL_RUNTIME_ROOT:-${xqtl_local_runtime_root_default}}"

    xqtl_local_pixi_mkdirs \
        "${runtime_home}" \
        "${runtime_root}" \
        "${runtime_home}/.sos" \
        "${runtime_root}/ipython" \
        "${runtime_root}/tmp" \
        "${runtime_root}/cache" \
        "${runtime_root}/config" \
        "${runtime_root}/data" \
        "${runtime_root}/state" \
        "${runtime_root}/matplotlib" \
        "${runtime_root}/jupyter" || return 1

    export HOME="${runtime_home}"
    export XQTL_LOCAL_RUNTIME_HOME="${runtime_home}"
    export XQTL_LOCAL_RUNTIME_ROOT="${runtime_root}"
    export IPYTHONDIR="${IPYTHONDIR:-${runtime_root}/ipython}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${runtime_root}/cache}"
    export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${runtime_root}/config}"
    export XDG_DATA_HOME="${XDG_DATA_HOME:-${runtime_root}/data}"
    export XDG_STATE_HOME="${XDG_STATE_HOME:-${runtime_root}/state}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${runtime_root}/matplotlib}"
    export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-${runtime_root}/jupyter}"
    export R_HISTFILE="${R_HISTFILE:-${runtime_root}/Rhistory}"
    # SoS puts step locks under tempfile.gettempdir()/USER/.sos, so force
    # transient state into the route-local runtime instead of inherited /tmp.
    export TMPDIR="${runtime_root}/tmp"
    export TMP="${TMPDIR}"
    export TEMP="${TMPDIR}"
}

xqtl_local_python_bin() {
    printf '%s\n' "${PIXI_HOME}/envs/python/bin/python"
}

xqtl_local_r_home() {
    printf '%s\n' "${PIXI_HOME}/envs/r-base/lib/R"
}

xqtl_local_r_exec() {
    printf '%s\n' "$(xqtl_local_r_home)/bin/exec/R"
}

xqtl_local_export_r_runtime() {
    local r_home
    r_home="$(xqtl_local_r_home)"

    export R_HOME="${r_home}"
    export R_SHARE_DIR="${r_home}/share"
    export R_INCLUDE_DIR="${r_home}/include"
    export R_DOC_DIR="${r_home}/doc"
    export LD_LIBRARY_PATH="${r_home}/lib:${PIXI_HOME}/envs/r-base/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
