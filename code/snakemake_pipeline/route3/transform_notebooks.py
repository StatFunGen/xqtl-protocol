"""
Transform SoS pipeline notebooks into Route 3 SoS wrapper notebooks.

For each target notebook:
  - All markdown cells are copied unchanged.
  - Code cells without `task:` blocks are copied unchanged.
  - The [global] code cell is copied unchanged but with an extra parameter line inserted
    after the first line (`[global]`).
  - Code cells WITH `task:` blocks: keep everything up to and including the `task:` line,
    then replace ALL language blocks (bash:, python3:, R:, python:) with ONE new `bash:`
    block that calls the corresponding modular script.
"""

import json
import copy
import re
import shutil
from pathlib import Path

SRC = Path("/home/user/xqtl-protocol/pipeline")
DST = Path("/home/user/xqtl-protocol/code/snakemake_pipeline/route3/notebooks")
DST.mkdir(parents=True, exist_ok=True)

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


def find_first_lang_block_pos(src):
    """Return the start position of the first language block line."""
    m = re.search(r'^[ \t]*(bash|python3?|R)\s*:', src, re.MULTILINE)
    return m.start() if m else -1


def make_bash_block(script_rel, step_cmd, params, stderr_expr, stdout_expr):
    """Build a replacement bash block string."""
    # We need the literal string: bash: expand= "${ }", ...
    # Using concatenation to avoid f-string brace interpretation issues.
    block = 'bash: expand= "${ }", stderr = ' + stderr_expr + ', stdout = ' + stdout_expr + ', container = container, entrypoint = entrypoint\n'
    block += '    bash ${renovated_code_dir}/' + script_rel + ' ' + step_cmd + ' \\\n'
    block += '        --container "${container}" \\\n'
    for p in params[:-1]:
        block += f'        {p} \\\n'
    if params:
        block += f'        {params[-1]}\n'
    return block


def transform_global_cell(src):
    """Insert renovated_code_dir parameter after the [global] line."""
    lines = src.split("\n")
    result = []
    inserted = False
    for i, line in enumerate(lines):
        result.append(line)
        if not inserted and line.strip() == "[global]":
            # Insert after this line
            result.append(RENOV_PARAM.rstrip("\n"))
            inserted = True
    return "\n".join(result)


def transform_task_cell(src, script_rel, step_cmd, params, stderr_expr, stdout_expr):
    """
    Keep everything up to and including the `task:` line,
    then replace all language blocks with a single bash block.
    """
    # Find the task: line
    task_match = re.search(r'^[ \t]*task\s*:', src, re.MULTILINE)
    if task_match is None:
        return src  # no task block, return as-is

    # Find the end of the task: line
    task_line_end = src.index('\n', task_match.start()) + 1

    # Keep everything up to and including the task line
    prefix = src[:task_line_end]

    # Build the new bash block
    bash_block = make_bash_block(script_rel, step_cmd, params, stderr_expr, stdout_expr)

    return prefix + bash_block


def copy_notebook_as_is(nb_name):
    """Copy a notebook file unchanged."""
    src_path = SRC / nb_name
    dst_path = DST / nb_name
    shutil.copy2(src_path, dst_path)
    print(f"  [COPY AS-IS] {nb_name}")


def transform_notebook(nb_name, step_transforms):
    """
    Transform a notebook.

    step_transforms: dict mapping step_header_pattern -> (script_rel, step_cmd, params, stderr_expr, stdout_expr)
      where step_header_pattern is a regex pattern that matches the step header line.
    """
    src_path = SRC / nb_name
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

        # Check if this is the global cell
        if re.match(r'\s*\[global\]', src):
            new_src = transform_global_cell(src)
            cell['source'] = src_to_list(new_src)
            new_cells.append(cell)
            continue

        # Check if this cell has a task: block
        if not re.search(r'^[ \t]*task\s*:', src, re.MULTILINE):
            new_cells.append(cell)
            continue

        # Check if any step transform pattern matches
        matched = False
        for pattern, (script_rel, step_cmd, params, stderr_expr, stdout_expr) in step_transforms.items():
            if re.search(pattern, src, re.MULTILINE):
                new_src = transform_task_cell(src, script_rel, step_cmd, params, stderr_expr, stdout_expr)
                cell['source'] = src_to_list(new_src)
                matched = True
                print(f"  [TRANSFORM] step pattern '{pattern}' in {nb_name}")
                break

        if not matched:
            # No matching transform, keep as-is
            pass

        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(dst_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"  [DONE] {nb_name}")


# ---------------------------------------------------------------------------
# Notebook-specific transformations
# ---------------------------------------------------------------------------

# 1. RNA_calling.ipynb
def transform_RNA_calling():
    transform_notebook("RNA_calling.ipynb", {
        # [fastqc] step
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
        ),
        # [rnaseqc_call_1] step
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
        ),
    })


# 2. VCF_QC.ipynb
def transform_VCF_QC():
    transform_notebook("VCF_QC.ipynb", {
        # [qc_1 (variant preprocessing)]
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
        ),
    })


# 3. genotype_formatting.ipynb
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
        ),
    })


# 4. GWAS_QC.ipynb
def transform_GWAS_QC():
    transform_notebook("GWAS_QC.ipynb", {
        # [qc_no_prune, qc_1 (basic QC filters)]
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
        ),
        # [qc_2 (LD pruning)]
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
        ),
        # [king_1]
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
        ),
        # [genotype_phenotype_sample_overlap]
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
        ),
    })


# 5. PCA.ipynb
def transform_PCA():
    transform_notebook("PCA.ipynb", {
        r'^\[flashpca_1\]': (
            "data_preprocessing/genotype/PCA.sh",
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
        ),
        r'^\[project_samples_1\]': (
            "data_preprocessing/genotype/PCA.sh",
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
        ),
    })


# 6. covariate_formatting.ipynb
def transform_covariate_formatting():
    transform_notebook("covariate_formatting.ipynb", {
        r'^\[merge_genotype_pc\]': (
            "data_preprocessing/covariate/covariate_formatting.sh",
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
        ),
    })


# 7. phenotype_formatting.ipynb
def transform_phenotype_formatting():
    transform_notebook("phenotype_formatting.ipynb", {
        r'^\[phenotype_by_chrom_1\]': (
            "data_preprocessing/phenotype/phenotype_formatting.sh",
            "phenotype_by_chrom",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${phenoFile}"',
                '--name "${name}"',
                '--chrom ${_chrom}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
        ),
    })


# 8. covariate_hidden_factor.ipynb
def transform_covariate_hidden_factor():
    transform_notebook("covariate_hidden_factor.ipynb", {
        # [*_1(computing residual ...)] - matches any step name as first sub-step
        r'^\[\*_1': (
            "data_preprocessing/covariate/covariate_hidden_factor.sh",
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
        ),
        # [Marchenko_PC_2, PCA_2]
        r'^\[Marchenko_PC_2': (
            "data_preprocessing/covariate/covariate_hidden_factor.sh",
            "Marchenko_PC",
            [
                '--cwd "${cwd}"',
                '--residFile "${_input}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
        ),
        # [PEER_2]
        r'^\[PEER_2\]': (
            "data_preprocessing/covariate/covariate_hidden_factor.sh",
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
        ),
        # [PEER_3]
        r'^\[PEER_3\]': (
            "data_preprocessing/covariate/covariate_hidden_factor.sh",
            "PEER_extract",
            [
                '--cwd "${cwd}"',
                '--modelFile "${_input}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
        ),
    })


# 9. TensorQTL.ipynb
def transform_TensorQTL():
    transform_notebook("TensorQTL.ipynb", {
        r'^\[cis_1\]': (
            "association_scan/TensorQTL/TensorQTL.sh",
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
        ),
        r'^\[cis_2\]': (
            "association_scan/TensorQTL/TensorQTL.sh",
            "cis_postprocess",
            [
                '--cwd "${cwd}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
        ),
        r'^\[trans\]': (
            "association_scan/TensorQTL/TensorQTL.sh",
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

    print("\n10. mnm_regression.ipynb (copy as-is)")
    copy_notebook_as_is("mnm_regression.ipynb")

    print("\n11. rss_analysis.ipynb (copy as-is)")
    copy_notebook_as_is("rss_analysis.ipynb")

    print("\n12. bulk_expression_QC.ipynb (copy as-is)")
    copy_notebook_as_is("bulk_expression_QC.ipynb")

    print("\n13. bulk_expression_normalization.ipynb (copy as-is)")
    copy_notebook_as_is("bulk_expression_normalization.ipynb")

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
        "mnm_regression.ipynb",
        "rss_analysis.ipynb",
        "bulk_expression_QC.ipynb",
        "bulk_expression_normalization.ipynb",
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
        print("\nAll 13 notebooks created successfully!")
    else:
        print("\nSome notebooks are missing!")
