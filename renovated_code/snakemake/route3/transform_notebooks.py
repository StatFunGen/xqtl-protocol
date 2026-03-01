"""
Transform SoS pipeline notebooks into Route 3 SoS wrapper notebooks.

For each target notebook:
  - All markdown cells are copied unchanged.
  - Code cells without `task:` blocks are copied unchanged.
  - The [global] code cell gets an extra parameter line inserted
    after the first line (`[global]`).
  - Code cells WITH `task:` blocks: keep everything up to and including
    the `task:` line, then replace ALL language blocks (bash:, python3:,
    R:, python:) with ONE new `bash:` block that calls the corresponding
    modular script directly:
      * R scripts  -> Rscript  ${renovated_code_dir}/script.R  --step STEP ...
      * Python scripts -> python3 ${renovated_code_dir}/script.py --step STEP ...
      * Bash-only (CLI wrappers) -> bash ${renovated_code_dir}/script.sh STEP ...
  - Simple bash task cells that only call CLI tools (tabix, samtools, etc.)
    are NOT transformed — they remain as-is.
"""

import json
import copy
import re
import shutil
from pathlib import Path

DST = Path("/home/user/xqtl-protocol/code/snakemake_pipeline/route3/notebooks")
DST.mkdir(parents=True, exist_ok=True)

# Primary source: existing route3 notebooks (already have the right structure).
# Fallback source: pipeline/ notebooks (for notebooks not yet in route3/notebooks/).
SRC = DST
SRC_PIPELINE = Path("/home/user/xqtl-protocol/pipeline")

RENOV_PARAM = "parameter: renovated_code_dir = path('renovated_code')  # override with --renovated-code-dir\n"


def src_to_list(text):
    """Convert a multiline string to a JSON notebook source list."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        elif line:
            result.append(line)
    return result


def make_bash_block(script_rel, step_cmd, params, stderr_expr, stdout_expr,
                    runner="bash"):
    """
    Build a replacement bash block string.

    runner:
      "bash"    -> bash ${renovated_code_dir}/script.sh STEP --container ...
      "rscript" -> Rscript ${renovated_code_dir}/script.R --step STEP ...
      "python3" -> python3 ${renovated_code_dir}/script.py --step STEP ...

    For rscript/python3 runners the container is handled by SoS via the
    `container = container` block option, so no --container flag is added.

    When step_cmd is None, no --step flag is added (for single-purpose scripts
    like susie_twas.R, mnm.R, fsusie.R, univariate_rss.R).
    """
    block = ('bash: expand= "${ }", stderr = ' + stderr_expr
             + ', stdout = ' + stdout_expr
             + ', container = container, entrypoint = entrypoint\n')

    if runner == "bash":
        block += '    bash ${renovated_code_dir}/' + script_rel + ' ' + step_cmd + ' \\\n'
        block += '        --container "${container}" \\\n'
    elif runner == "rscript":
        block += '    Rscript ${renovated_code_dir}/' + script_rel + ' \\\n'
        if step_cmd is not None:
            block += '        --step ' + step_cmd + ' \\\n'
    elif runner == "python3":
        block += '    python3 ${renovated_code_dir}/' + script_rel + ' \\\n'
        if step_cmd is not None:
            block += '        --step ' + step_cmd + ' \\\n'
    else:
        raise ValueError(f"Unknown runner: {runner!r}")

    for p in params[:-1]:
        block += f'        {p} \\\n'
    if params:
        block += f'        {params[-1]}\n'
    return block


def transform_global_cell(src):
    """Insert renovated_code_dir parameter after the [global] line (idempotent)."""
    if "renovated_code_dir" in src:
        return src  # already present — nothing to do
    lines = src.split("\n")
    result = []
    inserted = False
    for line in lines:
        result.append(line)
        if not inserted and line.strip() == "[global]":
            result.append(RENOV_PARAM.rstrip("\n"))
            inserted = True
    return "\n".join(result)


def transform_task_cell(src, script_rel, step_cmd, params,
                        stderr_expr, stdout_expr, runner="bash"):
    """
    Keep everything up to and including the `task:` line,
    then replace all language blocks with a single bash block.
    """
    task_match = re.search(r'^[ \t]*task\s*:', src, re.MULTILINE)
    if task_match is None:
        return src

    task_line_end = src.index('\n', task_match.start()) + 1
    prefix = src[:task_line_end]
    bash_block = make_bash_block(script_rel, step_cmd, params,
                                 stderr_expr, stdout_expr, runner)
    return prefix + bash_block


def transform_notebook(nb_name, step_transforms, src_dir=None):
    """
    Transform a notebook.

    step_transforms: dict mapping step_header_pattern ->
      (script_rel, step_cmd, params, stderr_expr, stdout_expr)
      or a 6-tuple with runner as 6th element:
      (script_rel, step_cmd, params, stderr_expr, stdout_expr, runner)

    src_dir: override the source directory (default: SRC = route3/notebooks/).
             Use SRC_PIPELINE for notebooks not yet present in route3/notebooks/.

    Cells whose step header does NOT match any pattern keep their
    original source unchanged (including simple bash cells).
    """
    src_path = (src_dir or SRC) / nb_name
    dst_path = DST / nb_name

    with open(src_path) as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb['cells']:
        cell = copy.deepcopy(cell)
        if cell['cell_type'] != 'code':
            new_cells.append(cell)
            continue

        src = ''.join(cell['source'])

        # Handle [global] cell
        if re.match(r'\s*\[global\]', src):
            new_src = transform_global_cell(src)
            cell['source'] = src_to_list(new_src)
            new_cells.append(cell)
            continue

        # Only consider cells with a task: block
        if not re.search(r'^[ \t]*task\s*:', src, re.MULTILINE):
            new_cells.append(cell)
            continue

        matched = False
        for pattern, transform_spec in step_transforms.items():
            if re.search(pattern, src, re.MULTILINE):
                script_rel   = transform_spec[0]
                step_cmd     = transform_spec[1]
                params       = transform_spec[2]
                stderr_expr  = transform_spec[3]
                stdout_expr  = transform_spec[4]
                runner       = transform_spec[5] if len(transform_spec) > 5 else "bash"
                new_src = transform_task_cell(src, script_rel, step_cmd, params,
                                              stderr_expr, stdout_expr, runner)
                cell['source'] = src_to_list(new_src)
                matched = True
                print(f"  [TRANSFORM] step pattern '{pattern}' in {nb_name}")
                break

        if not matched:
            # No matching transform — cell kept as-is (simple bash, etc.)
            pass

        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(dst_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"  [DONE] {nb_name}")


def copy_notebook_as_is(nb_name):
    """Copy a notebook file unchanged (legacy — prefer transform_notebook with {})."""
    src_path = SRC / nb_name
    dst_path = DST / nb_name
    shutil.copy2(src_path, dst_path)
    print(f"  [COPY AS-IS] {nb_name}")


# ---------------------------------------------------------------------------
# Notebook-specific transformations
# ---------------------------------------------------------------------------

# 1. RNA_calling.ipynb  — bash (STAR/FastQC CLI), python3 (GCT merge), Rscript (Picard aggregation)
def transform_RNA_calling():
    transform_notebook("RNA_calling.ipynb", {
        r'^\[fastqc\]': (
            "molecular_phenotypes/calling/RNA_calling.sh",
            "fastqc",
            [
                '--cwd "${cwd}"',
                '--sample-list "${sample_list}"',
                '--data-dir "${data_dir}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "bash",
        ),
        r'^\[rnaseqc_call_1\]': (
            "molecular_phenotypes/calling/RNA_calling.sh",
            "rnaseqc_call",
            [
                '--cwd "${cwd}"',
                '--sample-list "${sample_list}"',
                '--data-dir "${data_dir}"',
                '--gtf "${gtf}"',
                '--reference-fasta "${reference_fasta}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "bash",
        ),
        # rnaseqc_call_2: merge per-sample GCT outputs + aggregate metrics (Python)
        r'^\[rnaseqc_call_2\]': (
            "molecular_phenotypes/calling/RNA_calling.py",
            "rnaseqc_merge",
            [
                '--cwd "${cwd}"',
                '--name "${bam_list:bn}"',
                '--input ${" ".join([str(x) for x in _input])}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "python3",
        ),
        # rsem_call_4 / rnaseqc_call_4: aggregate Picard alignment/RNA/dup metrics (R)
        r'^\[rsem_call_4, rnaseqc_call_4\]': (
            "molecular_phenotypes/calling/RNA_calling.R",
            "aggregate_picard_qc",
            [
                '--input-dir "${_input[-1]:d}"',
                '--output "${_output}"',
                '--is-paired-end ${is_paired_end}',
                '${"--wasp" if varVCFfile else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 2. VCF_QC.ipynb  — bash-only (BCFTools / GATK CLI tools)
def transform_VCF_QC():
    transform_notebook("VCF_QC.ipynb", {
        r'^\[qc_1': (
            "data_preprocessing/genotype/VCF_QC.sh",
            "qc",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--dbsnp-variants "${dbsnp_variants}"',
                '--reference-genome "${reference_genome}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
    })


# 3. genotype_formatting.ipynb  — bash-only (PLINK CLI tools)
def transform_genotype_formatting():
    transform_notebook("genotype_formatting.ipynb", {
        r'^\[vcf_to_plink\]': (
            "data_preprocessing/genotype/genotype_formatting.sh",
            "vcf_to_plink",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--name "${name}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
        r'^\[merge_plink\]': (
            "data_preprocessing/genotype/genotype_formatting.sh",
            "merge_plink",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--name "${name}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
        r'^\[genotype_by_chrom_1\]': (
            "data_preprocessing/genotype/genotype_formatting.sh",
            "genotype_by_chrom",
            [
                '--cwd "${cwd}"',
                '--genoFile "${genoFile}"',
                '--name "${name}"',
                '--chrom ${chrom}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
    })


# 4. GWAS_QC.ipynb  — bash-only (PLINK / KING CLI tools) + Rscript for king_2
def transform_GWAS_QC():
    transform_notebook("GWAS_QC.ipynb", {
        r'^\[qc_no_prune': (
            "data_preprocessing/genotype/GWAS_QC.sh",
            "qc_no_prune",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--name "${name}"',
                '--mac-filter ${mac_filter}',
                '--maf-filter ${maf_filter}',
                '--geno-filter ${geno_filter}',
                '--mind-filter ${mind_filter}',
                '--hwe-filter ${hwe_filter}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
        r'^\[qc_2': (
            "data_preprocessing/genotype/GWAS_QC.sh",
            "qc",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--mac-filter ${mac_filter}',
                '--window ${window}',
                '--shift ${shift}',
                '--r2 ${r2}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "bash",
        ),
        r'^\[king_1\]': (
            "data_preprocessing/genotype/GWAS_QC.sh",
            "king",
            [
                '--cwd "${cwd}"',
                '--genoFile "${genoFile}"',
                '--keep-samples "${keep_samples}"',
                '--name "${name}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output}.stderr'",
            "f'{_output}.stdout'",
            "bash",
        ),
        # king_2: kinship-based sample removal (complex graph + plinkQC R code)
        r'^\[king_2': (
            "data_preprocessing/genotype/GWAS_QC.R",
            "king_2",
            [
                '--input "${_input}"',
                '--output "${_output}"',
                '--kinship ${kinship}',
            ],
            "f'{_output}.stderr'",
            "f'{_output}.stdout'",
            "rscript",
        ),
        r'^\[genotype_phenotype_sample_overlap\]': (
            "data_preprocessing/genotype/GWAS_QC.sh",
            "genotype_phenotype_sample_overlap",
            [
                '--cwd "${cwd}"',
                '--genoFile "${genoFile}"',
                '--phenoFile "${phenoFile}"',
                '--name "${name}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "bash",
        ),
    })


# 5. PCA.ipynb  — Rscript PCA.R
def transform_PCA():
    transform_notebook("PCA.ipynb", {
        r'^\[flashpca_1\]': (
            "data_preprocessing/genotype/PCA.R",
            "flashpca",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input[0]}"',
                '--k ${k}',
                '--maha-k ${maha_k}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[project_samples_1\]': (
            "data_preprocessing/genotype/PCA.R",
            "project_samples",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input[0]}"',
                '--pca-model "${pca_model}"',
                '--maha-k ${maha_k}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 6. covariate_formatting.ipynb  — Rscript covariate_formatting.R
def transform_covariate_formatting():
    transform_notebook("covariate_formatting.ipynb", {
        r'^\[merge_genotype_pc\]': (
            "data_preprocessing/covariate/covariate_formatting.R",
            "merge_genotype_pc",
            [
                '--cwd "${cwd}"',
                '--pcaFile "${pcaFile}"',
                '--covFile "${covFile}"',
                '--k ${k}',
                '--tol-cov ${tol_cov}',
                '${"--mean-impute" if mean_impute else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 7. phenotype_formatting.ipynb
#    [phenotype_by_chrom_1] and [phenotype_by_region_1] are SIMPLE BASH
#    (tabix / bgzip) — they are NOT transformed.
#    [gct_extract_samples] has R code — call Rscript phenotype_formatting.R
def transform_phenotype_formatting():
    transform_notebook("phenotype_formatting.ipynb", {
        r'^\[gct_extract_samples\]': (
            "data_preprocessing/phenotype/phenotype_formatting.R",
            "gct_extract_samples",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input[0]}"',
                '--keep-samples "${keep_samples}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:nn}.stderr'",
            "f'{_output:nn}.stdout'",
            "rscript",
        ),
    })


# 8. covariate_hidden_factor.ipynb  — Rscript covariate_hidden_factor.R
#    Sub-steps match the notebook's structure:
#      [*_1]            -> compute_residual
#      [Marchenko_PC_2] -> Marchenko_PC
#      [PEER_2]         -> PEER_fit
#      [PEER_3]         -> PEER_extract
def transform_covariate_hidden_factor():
    transform_notebook("covariate_hidden_factor.ipynb", {
        r'^\[\*_1': (
            "data_preprocessing/covariate/covariate_hidden_factor.R",
            "compute_residual",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${phenoFile}"',
                '--covFile "${covFile}"',
                '${"--mean-impute-missing" if mean_impute_missing else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:nn}.stderr'",
            "f'{_output:nn}.stdout'",
            "rscript",
        ),
        r'^\[Marchenko_PC_2': (
            "data_preprocessing/covariate/covariate_hidden_factor.R",
            "Marchenko_PC",
            [
                '--cwd "${cwd}"',
                '--residFile "${_input}"',
                '--N ${N}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[PEER_2\]': (
            "data_preprocessing/covariate/covariate_hidden_factor.R",
            "PEER_fit",
            [
                '--cwd "${cwd}"',
                '--residFile "${_input}"',
                '--N ${N}',
                '--iteration ${iteration}',
                '--convergence-mode "${convergence_mode}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
        r'^\[PEER_3\]': (
            "data_preprocessing/covariate/covariate_hidden_factor.R",
            "PEER_extract",
            [
                '--cwd "${cwd}"',
                '--modelFile "${_input}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
    })


# 9. TensorQTL.ipynb  — python3 TensorQTL.py
def transform_TensorQTL():
    transform_notebook("TensorQTL.ipynb", {
        r'^\[cis_1\]': (
            "association_scan/TensorQTL/TensorQTL.py",
            "cis",
            [
                '--cwd "${cwd}"',
                '--genotype-file "${genotype_file}"',
                '--phenotype-file "${phenotype_file}"',
                '--covariate-file "${covariate_file}"',
                '--window ${window}',
                '--MAC ${MAC}',
                '--maf-threshold ${maf_threshold}',
                '--numThreads ${numThreads}',
            ],
            'f\'${_output["nominal"]:nnn}.stderr\'',
            'f\'${_output["nominal"]:nnn}.stdout\'',
            "python3",
        ),
        r'^\[cis_2\]': (
            "association_scan/TensorQTL/TensorQTL.py",
            "cis_postprocess",
            [
                '--cwd "${cwd}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "python3",
        ),
        r'^\[trans\]': (
            "association_scan/TensorQTL/TensorQTL.py",
            "trans",
            [
                '--cwd "${cwd}"',
                '--genotype-file "${genotype_file}"',
                '--phenotype-file "${phenotype_file}"',
                '--covariate-file "${covariate_file}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "python3",
        ),
    })


# 10. phenotype_imputation.ipynb  — Rscript phenotype_imputation.R
#     Source: pipeline/ (not yet in route3/notebooks/)
#     [missxgboost] depends on a custom xgb_imp.R file and is kept as-is.
def transform_phenotype_imputation():
    transform_notebook("phenotype_imputation.ipynb", {
        r'^\[EBMF\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "EBMF",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--prior "${prior}"',
                '--varType "${varType}"',
                '--num-factor ${num_factor}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[gEBMF\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "gEBMF",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--num-factor ${num_factor}',
                '--nCores ${nCores}',
                '--backfit-iter ${backfit_iter}',
                '${"--save-flash" if save_flash else ""}',
                '${"--null-check" if null_check else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[missforest\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "missforest",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[knn\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "knn",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[soft\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "soft",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[mean\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "mean",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[lod\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "lod",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '${"--qc-prior-to-impute" if qc_prior_to_impute else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[bed_filter_na\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "bed_filter_na",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input}"',
                '--rank-max ${rank_max}',
                '--lambda-hyp ${lambda_hyp}',
                '--impute-method "${impute_method}"',
                '--tol-missing ${tol_missing}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        # [missxgboost] requires a custom xgb_imp.R file — kept as-is (no transform)
    }, src_dir=SRC_PIPELINE)


# 11. bulk_expression_QC.ipynb  — Rscript bulk_expression_QC.R
def transform_bulk_expression_QC():
    transform_notebook("bulk_expression_QC.ipynb", {
        r'^\[qc_1': (
            "molecular_phenotypes/QC/bulk_expression_QC.R",
            "qc_1",
            [
                '--cwd "${cwd}"',
                '--tpm-gct "${_input}"',
                '--low-expr-TPM ${low_expr_TPM}',
                '--low-expr-TPM-percent ${low_expr_TPM_percent}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:nnn}.stderr'",
            "f'{_output:nnn}.log'",
            "rscript",
        ),
        r'^\[qc_2': (
            "molecular_phenotypes/QC/bulk_expression_QC.R",
            "qc_2",
            [
                '--cwd "${cwd}"',
                '--tpm-gct "${_input}"',
                '--RLEFilterPercent ${RLEFilterPercent}',
                '--DSFilterPercent ${DSFilterPercent}',
                '--topk-genes ${topk_genes}',
                '--cluster-percent ${cluster_percent}',
                '--pvalue-cutoff ${pvalue_cutoff}',
                '--cluster-level ${cluster_level}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:nnn}.stderr'",
            "f'{_output:nnn}.log'",
            "rscript",
        ),
        r'^\[qc_3': (
            "molecular_phenotypes/QC/bulk_expression_QC.R",
            "qc_3",
            [
                '--cwd "${cwd}"',
                '--tpm-gct "${_input}"',
                '--counts-gct "${counts_gct}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:nn}.stderr'",
            "f'{_output:nn}.stdout'",
            "rscript",
        ),
    })


# 12. bulk_expression_normalization.ipynb  — python3 bulk_expression_normalization.py
def transform_bulk_expression_normalization():
    transform_notebook("bulk_expression_normalization.ipynb", {
        r'^\[normalize\]': (
            "molecular_phenotypes/QC/bulk_expression_normalization.py",
            "normalize",
            [
                '--cwd "${cwd}"',
                '--tpm-gct "${_input[0]}"',
                '--counts-gct "${_input[1]}"',
                '--annotation-gtf "${_input[2]}"',
                '--sample-participant-lookup "${_input[3]}"',
                '--tpm-threshold ${tpm_threshold}',
                '--count-threshold ${count_threshold}',
                '--sample-frac-threshold ${sample_frac_threshold}',
                '--normalization-method "${normalization_method}"',
                '${"--quantile-normalize" if quantile_normalize else ""}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:nn}.stderr'",
            "f'{_output[0]:nn}.stdout'",
            "python3",
        ),
    })


# 13. mnm_regression.ipynb — Rscript susie_twas.R / mnm.R / fsusie.R
#     [get_analysis_regions] is orchestration-only (no task:) — kept as-is.
#     [susie_twas], [mnm], [fsusie] each have an R task block -> call scripts directly.
#     [mnm_genes] and [mvfsusie] are kept as-is (complex / WIP).
def transform_mnm_regression():
    # Common params shared across susie_twas and mnm steps
    _common = [
        '--genotype ${_input[0]:a}',
        '--phenotype "${",".join([str(x.absolute()) for x in _input[1:len(_input)//2+1]])}"',
        '--covariate "${",".join([str(x.absolute()) for x in _input[len(_input)//2+1:]])}"',
        '--region "${_meta_info[0]}"',
        '--window "${_meta_info[1]}"',
        '--region-name "${_meta_info[2]}"',
        '--extract-region-names "${"|".join([x if isinstance(x,str) else ",".join(x) for x in _meta_info[3]])}"',
        '--conditions "${",".join(_meta_info[4:])}"',
        '--skip-analysis-pip-cutoff "${",".join(skip_analysis_pip_cutoff)}"',
        '--maf ${maf}',
        '--mac ${mac}',
        '--imiss ${imiss}',
        '${"--indel" if indel else ""}',
        '${"--keep-samples " + str(keep_samples) if keep_samples.is_file() else ""}',
        '${"--keep-variants " + str(keep_variants) if not keep_variants.is_dir() else ""}',
        '${"--save-data" if save_data else ""}',
        '--pip-cutoff ${pip_cutoff}',
        '--coverage "${",".join([str(x) for x in coverage])}"',
        '--seed ${seed}',
        '--cwd ${cwd:a}',
    ]

    transform_notebook("mnm_regression.ipynb", {
        r'^\[susie_twas\]': (
            "mnm_analysis/mnm_methods/susie_twas.R",
            None,   # no --step flag (single-purpose script)
            _common + [
                '${"--skip-fine-mapping" if skip_fine_mapping else ""}',
                '${"--skip-twas-weights" if skip_twas_weights else ""}',
                '${"--trans-analysis" if trans_analysis else ""}',
                '--init-l ${init_L}',
                '--max-l ${max_L}',
                '--max-cv-variants ${max_cv_variants}',
                '--twas-cv-folds ${twas_cv_folds}',
                '--twas-cv-threads ${twas_cv_threads}',
                '--min-twas-maf ${min_twas_maf}',
                '--min-twas-xvar ${min_twas_xvar}',
                '${"--ld-reference-meta-file " + str(ld_reference_meta_file) if not ld_reference_meta_file.is_dir() else ""}',
                '--output-prefix ${_output[0]:nn}',
            ],
            'f"{_output[0]:nn}.susie_twas.stderr"',
            'f"{_output[0]:nn}.susie_twas.stdout"',
            "rscript",
        ),
        r'^\[mnm\]': (
            "mnm_analysis/mnm_methods/mnm.R",
            None,
            _common + [
                '${"--skip-fine-mapping" if skip_fine_mapping else ""}',
                '${"--skip-twas-weights" if skip_twas_weights else ""}',
                '--xvar-cutoff ${xvar_cutoff}',
                '--mvsusie-max-iter ${mvsusie_max_iter}',
                '--mrmash-max-iter ${mrmash_max_iter}',
                '--max-cv-variants ${max_cv_variants}',
                '--twas-cv-folds ${twas_cv_folds}',
                '--twas-cv-threads ${twas_cv_threads}',
                '${"--ld-reference-meta-file " + str(ld_reference_meta_file) if not ld_reference_meta_file.is_dir() else ""}',
                '${"--mixture-prior " + str(mixture_prior) if mixture_prior.is_file() else ""}',
                '${"--mixture-prior-cv " + str(mixture_prior_cv) if mixture_prior_cv.is_file() else ""}',
                '--prior-weights-min ${prior_weights_min}',
                '${"--prior-canonical-matrices" if prior_canonical_matrices else ""}',
                '${"--sample-partition " + str(sample_partition) if sample_partition.is_file() else ""}',
                '--output-prefix ${_output[0]:nn}',
            ],
            'f"{_output[0]:nn}.mnm.stderr"',
            'f"{_output[0]:nn}.mnm.stdout"',
            "rscript",
        ),
        r'^\[fsusie\]': (
            "mnm_analysis/mnm_methods/fsusie.R",
            None,
            [
                '--genotype ${_input[0]:a}',
                '--phenotype "${",".join([str(x.absolute()) for x in _input[1:len(_input)//2+1]])}"',
                '--covariate "${",".join([str(x.absolute()) for x in _input[len(_input)//2+1:]])}"',
                '--region "${_meta_info[0]}"',
                '--window "${_meta_info[1]}"',
                '--region-name "${_meta_info[2]}"',
                '--conditions "${",".join(_meta_info[4:])}"',
                '--maf ${maf}',
                '--mac ${mac}',
                '--imiss ${imiss}',
                '${"--indel" if indel else ""}',
                '${"--keep-samples " + str(keep_samples) if keep_samples.is_file() else ""}',
                '${"--keep-variants " + str(keep_variants) if not keep_variants.is_dir() else ""}',
                '--prior "${prior}"',
                '--max-snp-em ${max_SNP_EM}',
                '--max-scale ${max_scale}',
                '--min-purity ${min_purity}',
                '--epigenetics-mark-threshold ${epigenetics_mark_treshold}',
                '--susie-top-pc ${susie_top_pc}',
                '--post-processing "${post_processing}"',
                '${"--small-sample-correction" if small_sample_correction else ""}',
                '--pip-cutoff ${pip_cutoff}',
                '--coverage "${",".join([str(x) for x in coverage])}"',
                '--init-l ${init_L}',
                '--max-l ${max_L}',
                '${"--skip-twas-weights" if skip_twas_weights else ""}',
                '--max-cv-variants ${max_cv_variants}',
                '--twas-cv-folds ${twas_cv_folds}',
                '--twas-cv-threads ${twas_cv_threads}',
                '${"--save-data" if save_data else ""}',
                '--output-prefix ${_output:n}',
                '--cwd ${cwd:a}',
            ],
            'f"{_output:n}.stderr"',
            'f"{_output:n}.stdout"',
            "rscript",
        ),
    })


# 14. rss_analysis.ipynb — Rscript univariate_rss.R / univariate_plot.R
#     [get_analysis_regions] is orchestration-only — kept as-is.
#     [univariate_rss]  -> call univariate_rss.R directly.
#     [univariate_plot] -> call univariate_plot.R directly.
def transform_rss_analysis():
    transform_notebook("rss_analysis.ipynb", {
        r'^\[univariate_plot\]': (
            "pecotmr_integration/univariate_plot.R",
            None,  # single-purpose script — no --step flag
            [
                '--input "${_input}"',
                '--output "${_output[0]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
        r'^\[univariate_rss\]': (
            "pecotmr_integration/univariate_rss.R",
            None,
            [
                '--ld-meta-data ${ld_meta_data}',
                '--studies "${",".join(regional_data["GWAS"].keys())}"',
                '--sumstat-paths "${",".join([regional_data["GWAS"][x][regional_data["regions"][_regions][0]][0] for x in regional_data["GWAS"].keys()])}"',
                '--column-file-paths "${",".join([str(regional_data["GWAS"][x][regional_data["regions"][_regions][0]][1]) for x in regional_data["GWAS"].keys()])}"',
                '--n-samples "${",".join([str(regional_data["GWAS"][x][regional_data["regions"][_regions][0]][2]) for x in regional_data["GWAS"].keys()])}"',
                '--n-cases "${",".join([str(regional_data["GWAS"][x][regional_data["regions"][_regions][0]][3]) for x in regional_data["GWAS"].keys()])}"',
                '--n-controls "${",".join([str(regional_data["GWAS"][x][regional_data["regions"][_regions][0]][4]) for x in regional_data["GWAS"].keys()])}"',
                '--region "chr${_regions.replace(":", "_")}"',
                '${"--skip-regions " + ",".join(skip_regions) if skip_regions else ""}',
                '${"--extract-region-name " + extract_region_name if extract_region_name != "NULL" else ""}',
                '${"--region-name-col " + region_name_col if region_name_col != "NULL" else ""}',
                '${"--compute-ld-from-genotype" if compute_LD_from_genotype else ""}',
                '--imiss ${imiss}',
                '--maf ${maf}',
                '--L ${L}',
                '--max-l ${max_L}',
                '--l-step ${l_step}',
                '--pip-cutoff ${pip_cutoff}',
                '--skip-analysis-pip-cutoff ${skip_analysis_pip_cutoff}',
                '--coverage "${",".join([str(x) for x in coverage])}"',
                '${"--finemapping-method " + finemapping_method if finemapping_method else ""}',
                '${"--impute" if impute else ""}',
                '--rcond ${rcond}',
                '--lamb ${lamb}',
                '--r2-threshold ${R2_threshold}',
                '--minimum-ld ${minimum_ld}',
                '${"--qc-method " + qc_method if qc_method else ""}',
                '${"--comment-string " + comment_string if comment_string != "NULL" else ""}',
                '${"--diagnostics" if diagnostics else ""}',
                '--output-prefix ${_output:nn}',
                '--output ${_output}',
            ],
            'f"{_output:n}.stderr"',
            'f"{_output:n}.stdout"',
            "rscript",
        ),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Transforming notebooks ===\n")

    print("1. RNA_calling.ipynb")
    transform_RNA_calling()

    print("\n2. VCF_QC.ipynb")
    transform_VCF_QC()

    print("\n3. genotype_formatting.ipynb")
    transform_genotype_formatting()

    print("\n4. GWAS_QC.ipynb")
    transform_GWAS_QC()

    print("\n5. PCA.ipynb")
    transform_PCA()

    print("\n6. covariate_formatting.ipynb")
    transform_covariate_formatting()

    print("\n7. phenotype_formatting.ipynb")
    transform_phenotype_formatting()

    print("\n8. covariate_hidden_factor.ipynb")
    transform_covariate_hidden_factor()

    print("\n9. TensorQTL.ipynb")
    transform_TensorQTL()

    print("\n10. phenotype_imputation.ipynb  (source: pipeline/)")
    transform_phenotype_imputation()

    print("\n11. bulk_expression_QC.ipynb")
    transform_bulk_expression_QC()

    print("\n12. bulk_expression_normalization.ipynb")
    transform_bulk_expression_normalization()

    print("\n13. mnm_regression.ipynb")
    transform_mnm_regression()

    print("\n14. rss_analysis.ipynb")
    transform_rss_analysis()

    print("\n=== Verification ===")
    expected = [
        "RNA_calling.ipynb",
        "VCF_QC.ipynb",
        "genotype_formatting.ipynb",
        "GWAS_QC.ipynb",
        "PCA.ipynb",
        "covariate_formatting.ipynb",
        "phenotype_formatting.ipynb",
        "covariate_hidden_factor.ipynb",
        "TensorQTL.ipynb",
        "phenotype_imputation.ipynb",
        "bulk_expression_QC.ipynb",
        "bulk_expression_normalization.ipynb",
        "mnm_regression.ipynb",
        "rss_analysis.ipynb",
    ]
    all_ok = True
    for name in expected:
        p = DST / name
        if p.exists():
            print(f"  OK  {name}")
        else:
            print(f"  MISSING  {name}")
            all_ok = False

    if all_ok:
        print("\nAll 14 notebooks created successfully!")
    else:
        print("\nSome notebooks are missing!")
