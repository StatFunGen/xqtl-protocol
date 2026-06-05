"""
Transform SoS pipeline notebooks into Modular SoS SoS wrapper notebooks.

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

DST = Path(__file__).resolve().parents[3] / "renovated_code" / "snakemake" / "modular_sos" / "notebooks"
DST.mkdir(parents=True, exist_ok=True)

# Primary source: existing modular_sos notebooks (already have the right structure).
# Fallback source: pipeline/ notebooks (for notebooks not yet in modular_sos/notebooks/).
SRC = DST
SRC_PIPELINE = Path(__file__).resolve().parents[3] / "pipeline"

RENOV_PARAM = "parameter: renovated_code_dir = path('renovated_code/script')  # override with --renovated-code-dir\n"


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


def transform_language_cell(src, script_rel, step_cmd, params,
                            stderr_expr, stdout_expr, runner="bash"):
    """
    Replace one or more inline SoS language blocks in a matched step cell
    even when the cell is not declared as a `task:`.
    """
    lang_match = re.search(r'^[ \t]*(bash|python3?|R)\s*:', src, re.MULTILINE)
    if lang_match is None:
        return src

    prefix = src[:lang_match.start()]
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

    src_dir: override the source directory (default: SRC = modular_sos/notebooks/).
             Use SRC_PIPELINE for notebooks not yet present in modular_sos/notebooks/.

    Cells whose step header does NOT match any pattern keep their
    original source unchanged (including simple bash cells).
    """
    src_path = (src_dir or SRC) / nb_name
    if not src_path.exists():
        src_path = SRC_PIPELINE / nb_name
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

        matched = False
        for pattern, transform_spec in step_transforms.items():
            if re.search(pattern, src, re.MULTILINE):
                script_rel   = transform_spec[0]
                step_cmd     = transform_spec[1]
                params       = transform_spec[2]
                stderr_expr  = transform_spec[3]
                stdout_expr  = transform_spec[4]
                runner       = transform_spec[5] if len(transform_spec) > 5 else "bash"
                if re.search(r'^[ \t]*task\s*:', src, re.MULTILINE):
                    new_src = transform_task_cell(src, script_rel, step_cmd, params,
                                                  stderr_expr, stdout_expr, runner)
                else:
                    new_src = transform_language_cell(src, script_rel, step_cmd, params,
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


def _step_name(cell):
    if cell.get("cell_type") != "code":
        return None
    src = "".join(cell.get("source", []))
    match = re.search(r"^\[([^\]]+)\]", src, re.MULTILINE)
    return match.group(1) if match else None


def restore_step_cells_from_pipeline(nb_name, step_names, *, insert_after=None):
    """Replace or insert specific step cells from the legacy pipeline notebook."""
    dst_path = DST / nb_name
    src_path = SRC_PIPELINE / nb_name

    with open(dst_path) as f:
        dst_nb = json.load(f)
    with open(src_path) as f:
        src_nb = json.load(f)

    src_cells = {
        name: copy.deepcopy(cell)
        for cell in src_nb["cells"]
        if (name := _step_name(cell)) in step_names
    }
    dst_indexes = {
        name: idx
        for idx, cell in enumerate(dst_nb["cells"])
        if (name := _step_name(cell))
    }

    for step_name in step_names:
        if step_name not in src_cells:
            raise ValueError(f"Could not find step {step_name!r} in pipeline/{nb_name}")
        if step_name in dst_indexes:
            dst_nb["cells"][dst_indexes[step_name]] = src_cells[step_name]
        else:
            anchor = (insert_after or {}).get(step_name)
            if not anchor or anchor not in dst_indexes:
                raise ValueError(
                    f"Cannot insert step {step_name!r} into {nb_name}; missing anchor {anchor!r}"
                )
            insert_idx = dst_indexes[anchor] + 1
            dst_nb["cells"].insert(insert_idx, src_cells[step_name])
            dst_indexes = {
                name: idx
                for idx, cell in enumerate(dst_nb["cells"])
                if (name := _step_name(cell))
            }

    with open(dst_path, "w") as f:
        json.dump(dst_nb, f, indent=1)
    print(f"  [RESTORE] {nb_name}: {', '.join(step_names)}")


def copy_notebook_as_is(nb_name):
    """Copy a notebook file unchanged (legacy — prefer transform_notebook with {})."""
    src_path = SRC / nb_name
    if not src_path.exists():
        src_path = SRC_PIPELINE / nb_name
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
        r'^\[picard_qc, STAR_align_2\]': (
            "molecular_phenotypes/calling/RNA_calling.sh",
            "picard_qc_star_align_2",
            [
                '--cwd "${cwd}"',
                '--gtf "${gtf}"',
                '--ref-flat "${ref_flat}"',
                '--reference-fasta "${reference_fasta}"',
                '--java-mem "${java_mem}"',
                '--optical-distance ${optical_distance}',
                '--input-bam "${_input[0]}"',
                '--sample-id "${_sample_id}"',
                '--strand "${_strand}"',
                '--picard-metrics "${_output["picard_metrics"]}"',
                '--picard-rna-metrics "${_output["picard_rna_metrics"]}"',
                '--md-bam "${_output["md_bam"]}"',
                '--md-metrics "${_output["md_metrics"]}"',
                '--bigwig "${_output["bigwig"]}"',
                '--output-summary "${_output["output_summary"]}"',
                '--var-vcf-file "${varVCFfile}"',
                '--zap-raw-bam "${zap_raw_bam}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "bash",
        ),
        r'^\[STAR_align_3\]': (
            "molecular_phenotypes/calling/RNA_calling.py",
            "star_align_3",
            [
                '--output "${_output}"',
                '--input ${" ".join([str(x) for x in _input])}',
                '--sample-id ${" ".join([str(x) for x in sample_id])}',
                '--strand ${" ".join([str(x) for x in strand])}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "python3",
        ),
        r'^\[rsem_call_2\]': (
            "molecular_phenotypes/calling/RNA_calling.py",
            "rsem_call_2",
            [
                '--cwd "${cwd}"',
                '--name "${bam_list:bn}"',
                '--input ${" ".join([str(x) for x in _input])}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "python3",
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
        r'^\[rename_chrs:': (
            "data_preprocessing/genotype/VCF_QC.sh",
            "rename_chrs",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--output "${_output}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:nn}.stderr'",
            "f'{_output:nn}.stdout'",
            "bash",
        ),
        r'^\[dbsnp_annotate\]': (
            "data_preprocessing/genotype/VCF_QC.sh",
            "dbsnp_annotate",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--output "${_output}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
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
        r'^\[qc_3': (
            "data_preprocessing/genotype/VCF_QC.sh",
            "qc_3",
            [
                '--cwd "${cwd}"',
                '--genoFile "${_input}"',
                '--novel-sumstats "${_output[0]}"',
                '--known-sumstats "${_output[1]}"',
                '--novel-tstv "${_output[2]}"',
                '--known-tstv "${_output[3]}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
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
                '--mem "${mem}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
        r'^\[ld_by_region_plink_1\]': (
            "data_preprocessing/genotype/genotype_formatting.py",
            "ld_by_region_plink_1",
            [
                '--genoFile "${_input}"',
                '--region-chrom "${_regions[0]}"',
                '--region-start "${_regions[1]}"',
                '--region-end "${_regions[2]}"',
                '--float-type ${float_type}',
                '--output "${_output}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "python3",
        ),
        r'^\[genotype_by_chrom_1\]': (
            "data_preprocessing/genotype/genotype_formatting.sh",
            "genotype_by_chrom",
            [
                '--cwd "${cwd}"',
                '--genoFile "${genoFile}"',
                '--name "${name}"',
                '--chrom ${_chrom}',
                '--mem "${mem}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "bash",
        ),
        r'^\[write_data_list\]': (
            "data_preprocessing/genotype/genotype_formatting.py",
            "write_data_list",
            [
                '--data-files "${"::".join([str(x) for x in _input])}"',
                '--ext "${ext}"',
                '--output "${_output}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "python3",
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
                '--maf-max-filter ${maf_max_filter}',
                '--mac-max-filter ${mac_max_filter}',
                '--geno-filter ${geno_filter}',
                '--mind-filter ${mind_filter}',
                '--hwe-filter ${hwe_filter}',
                '${"--keep-samples " + str(keep_samples) if keep_samples.is_file() else ""}',
                '${"--remove-samples " + str(remove_samples) if remove_samples.is_file() else ""}',
                '${"--exclude-variants " + str(exclude_variants) if exclude_variants.is_file() else ""}',
                '${"--keep-variants " + str(keep_variants) if keep_variants.is_file() else ""}',
                '${"--meta-only" if meta_only else ""}',
                '${"--rm-dups" if rm_dups else ""}',
                '${"--treat-dosage-missing" if treat_dosage_missing else ""}',
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
        r'^\[plot_pca\]': (
            "data_preprocessing/genotype/PCA.R",
            "plot_pca",
            [
                '--cwd "${cwd}"',
                '--plot-data "${_input}"',
                '--outlier-file "${outlier_file}"',
                '--min-axis "${min_axis}"',
                '--max-axis "${max_axis}"',
                '--pop-col "${pop_col}"',
                '--label-col "${label_col}"',
                '--pops "${",".join([str(x) for x in pops])}"',
                '--k ${k}',
                '--output-pc-plot "${_output[0]}"',
                '--output-scree-plot "${_output[1]}"',
                '--output-scree-text "${_output[2]}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
        r'^\[detect_outliers\]': (
            "data_preprocessing/genotype/PCA.R",
            "detect_outliers",
            [
                '--cwd "${cwd}"',
                '--pca-result "${_input}"',
                '--prob ${prob}',
                '--pval ${pval}',
                '--robust ${"TRUE" if robust else "FALSE"}',
                '--pop-col "${pop_col}"',
                '--k ${k}',
                '--distance-output "${_output[\'distance\']}"',
                '--identified-outliers-output "${_output[\'identified_outliers\']}"',
                '--analysis-summary-output "${_output[\'analysis_summary\']}"',
                '--qqplot-output "${_output[\'qqplot_mahalanobis\']}"',
                '--hist-output "${_output[\'hist_mahalanobis\']}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
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
                '--name "${name}"',
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
        r'^\[phenotype_annotate_by_tad\]': (
            "data_preprocessing/phenotype/phenotype_formatting.R",
            "phenotype_annotate_by_tad",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input[0]}"',
                '--TAD_list "${_input[1]}"',
                '--phenotype-per-tad ${phenotype_per_tad}',
                '--output "${_output}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 8. covariate_hidden_factor.ipynb  — Rscript covariate_hidden_factor.R
#    Sub-steps match the notebook's structure:
#      [*_1]            -> compute_residual
#      [Marchenko_PC_2] -> Marchenko_PC
#      [PEER_2]         -> PEER_fit
#      [PEER_3]         -> PEER_extract
#      [BiCV_2]         -> BiCV_2
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
                '--covFile "${covFile}"',
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
        r'^\[BiCV_2\]': (
            "data_preprocessing/covariate/covariate_hidden_factor.R",
            "BiCV_2",
            [
                '--cwd "${cwd}"',
                '--residFile "${_input}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output}.stderr'",
            "f'{_output}.stdout'",
            "rscript",
        ),
        r'^\[BiCV_3\]': (
            "data_preprocessing/covariate/covariate_hidden_factor.R",
            "BiCV_3",
            [
                '--cwd "${cwd}"',
                '--residFile "${_input[0]}"',
                '--vcfFile "${_input[1]}"',
                '--covFile "${covFile}"',
                '--output "${_output}"',
                '--N ${N}',
                '--iteration ${iteration}',
                '--numThreads ${numThreads}',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
    })


# 9. TensorQTL.ipynb  — python3 TensorQTL.py
def transform_TensorQTL():
    transform_notebook("TensorQTL.ipynb", {
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
    restore_step_cells_from_pipeline("TensorQTL.ipynb", ["cis_1", "cis_2"])


# 10. phenotype_imputation.ipynb  — Rscript phenotype_imputation.R
#     Source: pipeline/ (not yet in modular_sos/notebooks/)
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
        r'^\[missxgboost\]': (
            "data_preprocessing/phenotype/phenotype_imputation.R",
            "missxgboost",
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


# 13. mnm_regression.ipynb — Rscript susie_twas.R / mnm.R / fsusie.R / mnm_genes.R / mvfsusie.R
#     [get_analysis_regions] is orchestration-only (no task:) — kept as-is.
#     [susie_twas], [mnm], [fsusie], [mnm_genes], [mvfsusie] each have an R task block -> call scripts directly.
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
                '${"--estimate-residual-method " + str(estimate_residual_method) if estimate_residual_method else ""}',
                '${"--small-sample-correction" if small_sample_correction else ""}',
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
        r'^\[mnm_genes\]': (
            "mnm_analysis/mnm_methods/mnm_genes.R",
            None,
            [
                '--genotype ${_input[0]:a}',
                '--phenotype "${",".join([str(x.absolute()) for x in _input[1:len(_input)//2+1]])}"',
                '--covariate "${",".join([str(x.absolute()) for x in _input[len(_input)//2+1:]])}"',
                '--region-coord "${_meta_info[0]}"',
                '--grange "${_meta_info[1]}"',
                '--region-name "${_meta_info[2]}"',
                '--raw-region-names "${"|".join([x if isinstance(x,str) else ",".join(x) for x in _meta_info[3]])}"',
                '--condition-name "${_meta_info[-1]}"',
                '--skip-analysis-pip-cutoff "${",".join(skip_analysis_pip_cutoff)}"',
                '--pheno-id-map-file ${pheno_id_map_file:r}',
                '--fine-mapping-meta ${fine_mapping_meta:r}',
                '--maf ${maf}',
                '--mac ${mac}',
                '--imiss ${imiss}',
                '${"--indel" if indel else ""}',
                '${"--keep-samples " + str(keep_samples) if keep_samples.is_file() else ""}',
                '${"--keep-variants " + str(keep_variants) if not keep_variants.is_dir() else ""}',
                '${"--data-driven-prior" if data_driven_prior else ""}',
                '--n-random ${n_random}',
                '--n-null ${n_null}',
                '${"--independent-variant-list " + str(independent_variant_list) if independent_variant_list.is_file() else ""}',
                '--prior-weights-min ${prior_weights_min}',
                '${"--prior-canonical-matrices" if prior_canonical_matrices else ""}',
                '${"--sample-partition " + str(sample_partition) if sample_partition.is_file() else ""}',
                '--mvsusie-max-iter ${mvsusie_max_iter}',
                '--mrmash-max-iter ${mrmash_max_iter}',
                '${"--save-data" if save_data else ""}',
                '${"--skip-fine-mapping" if skip_fine_mapping else ""}',
                '${"--skip-twas-weights" if skip_twas_weights else ""}',
                '--pip-cutoff ${pip_cutoff}',
                '--coverage "${",".join([str(x) for x in coverage])}"',
                '--max-cv-variants ${max_cv_variants}',
                '--twas-cv-folds ${twas_cv_folds}',
                '--twas-cv-threads ${twas_cv_threads}',
                '--seed ${seed}',
                '--output-files "${",".join([str(x) for x in _output])}"',
                '--cwd ${cwd:a}',
            ],
            'f"{_output[0]:nn}.mnm_genes.stderr"',
            'f"{_output[0]:nn}.mnm_genes.stdout"',
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
        r'^\[mvfsusie\]': (
            "mnm_analysis/mnm_methods/mvfsusie.R",
            None,
            [
                '--genotype ${_input[0]:a}',
                '--phenotype "${",".join([str(x.absolute()) for x in _input[1::2]])}"',
                '--covariate "${",".join([str(x.absolute()) for x in _input[2::2]])}"',
                '--region "${_meta_info[1]}:${_meta_info[2]}-${_meta_info[3]}"',
                '--maf ${maf}',
                '--mac ${mac}',
                '--imiss ${imiss}',
                '--max-l ${max_L}',
                '--output ${_output:a}',
            ],
            'f"{_output:n}.stderr"',
            'f"{_output:n}.stdout"',
            "rscript",
        ),
    })


# 14. rss_analysis.ipynb — keep univariate_plot identical to source semantics
#     [get_analysis_regions] is orchestration-only — kept as-is.
#     [univariate_rss]  -> call univariate_rss.R directly.
#     [univariate_plot] -> restored from pipeline notebook for maximum fidelity.
def transform_rss_analysis():
    transform_notebook("rss_analysis.ipynb", {
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
    restore_step_cells_from_pipeline("rss_analysis.ipynb", ["univariate_plot"])


def transform_mash_preprocessing():
    transform_notebook("mash_preprocessing.ipynb", {
        r'^\[susie_to_mash_2\]': (
            "mnm_analysis/mnm_methods/mash_preprocessing.R",
            "susie_to_mash_2",
            [
                '--input-files "${",".join([str(x) for x in _input])}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })
    restore_step_cells_from_pipeline(
        "mash_preprocessing.ipynb",
        ["susie_to_mash_1", "random_null_tensorqtl_1"],
    )


# 15. splicing_calling.ipynb — Rscript splicing_calling.R for psichomics steps
def transform_splicing_calling():
    transform_notebook("splicing_calling.ipynb", {
        r'^\[psichomics_1\]': (
            "molecular_phenotypes/calling/splicing_calling.R",
            "psichomics_1",
            [
                '--input-files "${",".join([str(x.absolute()) for x in _input])}"',
                '--cwd "${cwd:a}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[psichomics_2\]': (
            "molecular_phenotypes/calling/splicing_calling.R",
            "psichomics_2",
            [
                '--junction-file "${cwd}/psichomics_junctions.txt"',
                '--splicing-annotation "${splicing_annotation:a}"',
                '--cwd "${cwd:a}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[phenotype_annotate_by_tad\]': (
            "data_preprocessing/phenotype/phenotype_formatting.R",
            "phenotype_annotate_by_tad",
            [
                '--cwd "${cwd}"',
                '--phenoFile "${_input[0]}"',
                '--TAD_list "${_input[1]}"',
                '--phenotype-per-tad ${phenotype_per_tad}',
                '--output "${_output}"',
                '--numThreads ${numThreads}',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 16. methylation_calling.ipynb — Rscript methylation_calling.R
def transform_methylation_calling():
    transform_notebook("methylation_calling.ipynb", {
        r'^\[sesame_1\]': (
            "molecular_phenotypes/calling/methylation_calling.R",
            "sesame_1",
            [
                '--sample-sheet "${_input}"',
                '--idat-folder "${idat_folder:a}"',
                '${"--keep-only-cpg-probes" if keep_only_cpg_probes else ""}',
                '--samples-frac-dt-cutoff ${samples_frac_dt_cutoff}',
                '--sample-sheet-header-rows ${sample_sheet_header_rows}',
                '--n-cores ${n_cores}',
                '--output-rds "${_output[0]}"',
                '--output-beta "${_output[1]}"',
                '--output-m "${_output[2]}"',
                '--output-qc "${_output[3]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
        r'^\[minfi_1\]': (
            "molecular_phenotypes/calling/methylation_calling.R",
            "minfi_1",
            [
                '--sample-sheet "${_input}"',
                '${"--keep-only-cpg-probes" if keep_only_cpg_probes else ""}',
                '--samples-pval-cutoff ${samples_pval_cutoff}',
                '--probe-pval-cutoff ${probe_pval_cutoff}',
                '--cross-reactive-probes "${cross_reactive_probes:a}"',
                '--hg-build ${hg_build}',
                '--output-rds "${_output[0]}"',
                '--output-beta "${_output[1]}"',
                '--output-m "${_output[2]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
        r'^\[\*_2\]': (
            "molecular_phenotypes/calling/methylation_calling.R",
            "bed_format",
            [
                '--beta-input "${_input[1]}"',
                '--m-input "${_input[2]}"',
                '--output-beta-bed "${_output[0]}"',
                '--output-m-bed "${_output[1]}"',
                '--output-annot "${_output[2]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
    })


# 17. qtl_association_postprocessing.ipynb — Rscript qtl_association_postprocessing.R
def transform_qtl_association_postprocessing():
    transform_notebook("qtl_association_postprocessing.ipynb", {
        r'^\[default\]': (
            "association_scan/qtl_association_postprocessing.R",
            None,
            [
                '--pecotmr-path "${pecotmr_path:a}"',
                '--workdir "${work_dir}"',
                '--maf-cutoff ${maf_cutoff}',
                '--cis-window ${cis_window}',
                '--pvalue-cutoff ${pvalue_cutoff}',
                '--fdr-threshold ${fdr_threshold}',
                '--gene-coordinates "${gene_coordinates:a}"',
                '--output-dir "${output_dir}"',
                '--archive-dir "${archive_dir}"',
                '--regional-pattern "${regional_pattern}"',
                '--n-variants-suffix "${n_variants_suffix}"',
                '--qtl-pattern "${qtl_pattern}"',
                '--pvalue-pattern "${pvalue_pattern}"',
                '--qvalue-pattern "${qvalue_pattern}"',
                '--start-distance-col "${tss_dist_col}"',
                '--end-distance-col "${tes_dist_col}"',
                '--af-col "${af_col}"',
                '--molecular-id-col "${molecular_id_col}"',
                '${"--enable-archive" if enable_archive else ""}',
                '--additional-pvalue-cols "${additional_pvalue_cols}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 18. intact.ipynb — Rscript intact.R
def transform_intact():
    transform_notebook("intact.ipynb", {
        r'^\[intact\]': (
            "pecotmr_integration/intact.R",
            None,
            [
                '--ptwas-file "${ptwas_file}"',
                '--fastenloc-file "${fastenloc_file}"',
                '--alpha ${alpha}',
                '--output "${_output}"',
            ],
            "f'{_output:nn}.stderr'",
            "f'{_output:nn}.stdout'",
            "rscript",
        ),
    })


# 19. gene_annotation.ipynb — Python/R script-backed modular_sos conversion
def transform_gene_annotation():
    transform_notebook("gene_annotation.ipynb", {
        r'^\[annotate_coord\]': (
            "data_preprocessing/phenotype/gene_annotation.py",
            "annotate_coord",
            [
                '--phenoFile "${_input[0]}"',
                '--coordinate-annotation "${_input[1]}"',
                '--sample-participant-lookup "${sample_participant_lookup}"',
                '--molecular-trait-type "${molecular_trait_type}"',
                '--phenotype-id-column "${phenotype_id_column}"',
                '--auxiliary-id-mapping "${auxiliary_id_mapping}"',
                '--sep "${sep}"',
                '${"--strip-id" if strip_id else ""}',
                '--output-bed "${_output[0]}"',
                '--output-region-list "${_output[1]}"',
                '--gene-list-output "${cwd:a}/${_input[0]:bn}.gene_list.tsv"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "python3",
        ),
        r'^\[annotate_coord_biomart\]': (
            "data_preprocessing/phenotype/gene_annotation.R",
            "annotate_coord_biomart",
            [
                '--phenoFile "${_input[0]}"',
                '--ensembl-version ${ensembl_version}',
                '--output-bed "${_output[0]}"',
                '--output-region-list "${_output[1]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
        r'^\[map_leafcutter_cluster_to_gene\]': (
            "data_preprocessing/phenotype/gene_annotation.py",
            "map_leafcutter_cluster_to_gene",
            [
                '--intron-count "${_input[0]}"',
                '--annotation-gtf "${_input[1]}"',
                '--map-stra "${map_stra}"',
                '--overlap-ratio ${overlap_ratio}',
                '--output-exon-list "${_output[0]}"',
                '--output-cluster-map "${_output[1]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "python3",
        ),
        r'^\[annotate_leafcutter_isoforms\]': (
            "data_preprocessing/phenotype/gene_annotation.py",
            "annotate_leafcutter_isoforms",
            [
                '--phenoFile "${_input[0]}"',
                '--annotation-gtf "${_input[1]}"',
                '--cluster-map "${_input[3]}"',
                '--sample-participant-lookup "${sample_participant_lookup}"',
                '--output-bed "${_output[0]}"',
                '--output-phenotype-group "${_output[1]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "python3",
        ),
        r'^\[annotate_psichomics_isoforms\]': (
            "data_preprocessing/phenotype/gene_annotation.py",
            "annotate_psichomics_isoforms",
            [
                '--phenoFile "${_input[0]}"',
                '--annotation-gtf "${_input[1]}"',
                '--sample-participant-lookup "${sample_participant_lookup}"',
                '--output-bed "${_output[0]}"',
                '--output-phenotype-group "${_output[1]}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "python3",
        ),
    })


# 20. SuSiE_enloc.ipynb — Rscript SuSiE_enloc.R
def transform_SuSiE_enloc():
    transform_notebook("SuSiE_enloc.ipynb", {
        r'^\[xqtl_gwas_enrichment\]': (
            "pecotmr_integration/SuSiE_enloc.R",
            "xqtl_gwas_enrichment",
            [
                '--xqtl-files "${",".join([str(x.absolute()) for x in _input])}"',
                '--gwas-files "${",".join([str(x) for x in _meta[1:]])}"',
                '--context "${_meta[0]}"',
                '--xqtl-finemapping-obj "${",".join([str(x) for x in xqtl_finemapping_obj]) if len(xqtl_finemapping_obj) != 0 else ""}"',
                '--xqtl-varname-obj "${",".join([str(x) for x in xqtl_varname_obj]) if len(xqtl_varname_obj) != 0 else ""}"',
                '--gwas-finemapping-obj "${",".join([str(x) for x in gwas_finemapping_obj]) if len(gwas_finemapping_obj) != 0 else ""}"',
                '--gwas-varname-obj "${",".join([str(x) for x in gwas_varname_obj]) if len(gwas_varname_obj) != 0 else ""}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[susie_coloc\]': (
            "pecotmr_integration/SuSiE_enloc.R",
            "susie_coloc",
            [
                '--xqtl-files "${",".join([str(x.absolute()) for x in _input])}"',
                '--gwas-files "${",".join([str(x) for x in _meta[1:]])}"',
                '--context "${_meta[0]}"',
                '--xqtl-finemapping-obj "${",".join([str(x) for x in xqtl_finemapping_obj]) if len(xqtl_finemapping_obj) != 0 else ""}"',
                '--xqtl-varname-obj "${",".join([str(x) for x in xqtl_varname_obj]) if len(xqtl_varname_obj) != 0 else ""}"',
                '--gwas-finemapping-obj "${",".join([str(x) for x in gwas_finemapping_obj]) if len(gwas_finemapping_obj) != 0 else ""}"',
                '--gwas-varname-obj "${",".join([str(x) for x in gwas_varname_obj]) if len(gwas_varname_obj) != 0 else ""}"',
                '--xqtl-region-obj "${",".join([str(x) for x in xqtl_region_obj]) if len(xqtl_region_obj) != 0 else ""}"',
                '--gwas-region-obj "${",".join([str(x) for x in gwas_region_obj]) if len(gwas_region_obj) != 0 else ""}"',
                '${"--skip-enrich" if skip_enrich else ""}',
                '${"--filter-lbf-cs" if not skip_enrich else ""}',
                '${"--ld-meta-file-path " + str(ld_meta_file_path) if ld_meta_file_path.is_file() else ""}',
                '--cwd "${cwd:a}"',
                '--name "${name}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 21. qr_and_twas.ipynb — Rscript qr_and_twas.R
def transform_qr_and_twas():
    transform_notebook("qr_and_twas.ipynb", {
        r'^\[quantile_qtl_twas_weight\]': (
            "association_scan/quantile_models/qr_and_twas.R",
            "quantile_qtl_twas_weight",
            [
                '--genotype "${_input[0]:a}"',
                '--phenotype "${",".join([str(x.absolute()) for x in _input[1:len(_input)//2+1]])}"',
                '--covariate "${",".join([str(x.absolute()) for x in _input[len(_input)//2+1:]])}"',
                '--conditions "${",".join([str(x) for x in _meta_info[4:]])}"',
                '--region "${_meta_info[0] if int(_meta_info[0].split("-")[-1]) > 0 else ""}"',
                '--window "${_meta_info[1]}"',
                '--region-name "${_meta_info[2]}"',
                '--extract-region-names "${"|".join([x if isinstance(x, str) else ",".join(x) for x in _meta_info[3]])}"',
                '--phenotype-header ${4 if int(_meta_info[0].split("-")[-1]) > 0 else 1}',
                '--region-name-col ${4 if int(_meta_info[0].split("-")[-1]) > 0 else 1}',
                '${"--keep-samples " + str(keep_samples) if keep_samples.is_file() else ""}',
                '${"--keep-variants " + str(keep_variants) if keep_variants.is_file() else ""}',
                '--maf ${maf}',
                '--mac ${mac}',
                '--imiss ${imiss}',
                '${"--indel" if indel else ""}',
                '--min-twas-maf ${min_twas_maf}',
                '--screen-threshold ${screen_threshold}',
                '${"--ld-reference-meta-file " + str(ld_reference_meta_file) if not ld_reference_meta_file.is_dir() else ""}',
                '${"--save-data" if save_data else ""}',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 22. gsea.ipynb — Rscript gsea.R
def transform_gsea():
    transform_notebook("gsea.ipynb", {
        r'^\[pathway_analysis\]': (
            "association_scan/gsea.R",
            "pathway_analysis",
            [
                '--genes-file "${genes_file}"',
                '--organism "${organism}"',
                '--pvalue-cutoff ${pvalue_cutoff}',
                '--output "${_output[\'pathway_results\']}"',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
    })


# 23. eoo_enrichment.ipynb — Rscript eoo_enrichment.R
def transform_eoo_enrichment():
    transform_notebook("eoo_enrichment.ipynb", {
        r'^\[enrichment\]': (
            "association_scan/eoo_enrichment.R",
            "enrichment",
            [
                '--significant-variants-path "${significant_variants_path}"',
                '--baseline-anno-path "${baseline_anno_path}"',
                '--annotations-start ${annotations_start}',
                '--output "${_output[\'enrichment\']}"',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
    })


# 24. apa_impute.ipynb — Rscript apa_impute.R
def transform_apa_impute():
    transform_notebook("apa_impute.ipynb", {
        r'^\[APAimpute\]': (
            "molecular_phenotypes/QC/apa_impute.R",
            "APAimpute",
            [
                '--cwd "${cwd}"',
                '--chrlist "${",".join([str(x) for x in chrlist])}"',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
        r'^\[APArename_1\]': (
            "molecular_phenotypes/QC/apa_impute.R",
            "APArename_1",
            [
                '--input "${_input}"',
                '--match "${match}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[APArename_2\]': (
            "molecular_phenotypes/QC/apa_impute.R",
            "APArename_2",
            [
                '--input "${_input}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
        r'^\[APArename_3\]': (
            "molecular_phenotypes/QC/apa_impute.R",
            "APArename_3",
            [
                '--input "${_input}"',
                '--match "${match}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })
    restore_step_cells_from_pipeline(
        "apa_impute.ipynb",
        ["APArename_3"],
    )


# 25. GRM.ipynb — Rscript GRM.R for APEX formatting
def transform_GRM():
    transform_notebook("GRM.ipynb", {
        r'^\[grm_3\]': (
            "data_preprocessing/genotype/GRM.R",
            "grm_3",
            [
                '--input "${_input}"',
                '--output "${_output}"',
            ],
            "f'{_output}.stderr'",
            "f'{_output}.stdout'",
            "rscript",
        ),
        r'^\[grm_4\]': (
            "data_preprocessing/genotype/GRM.R",
            "grm_4",
            [
                '--inputs "${",".join([str(x) for x in _input])}"',
                '--chroms "${",".join([str(x) for x in genotypes.keys()])}"',
                '--output "${_output}"',
            ],
            "f'{_output}.stderr'",
            "f'{_output}.stdout'",
            "rscript",
        ),
    })


# 26. METAL.ipynb — Rscript METAL.R for output reformatting
def transform_METAL():
    transform_notebook("METAL.ipynb", {
        r'^\[METAL_3,Output_reformatting\]': (
            "association_scan/METAL.R",
            "Output_reformatting",
            [
                '--input "${_input}"',
                '--name "${name}"',
                '--output-vcf "${_output[0]}"',
                '--output-sumstat "${_output[1]}"',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
        r'^\[METAL_4,recipe\]': (
            "association_scan/METAL.R",
            "recipe",
            [
                '--name "${name}"',
                '--sumstat-list-path "${sumstat_list_path}"',
                '--input-files "${"::".join([str(x) for x in _input])}"',
                '--output "${_output}"',
            ],
            "f'{_output}.stderr'",
            "f'{_output}.stdout'",
            "rscript",
        ),
    })


# 27. ld_prune_reference.ipynb — Rscript ld_prune_reference.R
def transform_ld_prune_reference():
    transform_notebook("ld_prune_reference.ipynb", {
        r'^\[LD_pruning_2\]': (
            "data_preprocessing/genotype/ld_prune_reference.R",
            "LD_pruning_2",
            [
                '--input-files "${"::".join([str(x) for x in _input])}"',
                '--output "${_output}"',
            ],
            "f'{_output:n}.stderr'",
            "f'{_output:n}.stdout'",
            "rscript",
        ),
    })


# 28. gregor.ipynb — Rscript gregor.R
def transform_gregor():
    transform_notebook("gregor.ipynb", {
        r'^\[gregor_3\]': (
            "association_scan/gregor.R",
            "gregor_3",
            [
                '--input-dir "${_input:ad}"',
                '--output-counts "${_output[0]}"',
                '--output-results "${_output[1]}"',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
        r'^\[gregor_fisher_plot\]': (
            "association_scan/gregor.R",
            "gregor_fisher_plot",
            [
                '--fisher1 "${_input[0]}"',
                '--fisher2 "${_input[1]}"',
                '--output "${_output}"',
            ],
            "f'{_output[0]:n}.stderr'",
            "f'{_output[0]:n}.stdout'",
            "rscript",
        ),
    })


# 29. pseudobulk_mega_expression_QC_and_normalization.ipynb
def transform_pseudobulk_mega_expression_QC_and_normalization():
    transform_notebook("pseudobulk_mega_expression_QC_and_normalization.ipynb", {
        r'^\[mergedata\]': (
            "molecular_phenotypes/QC/pseudobulk_mega_expression_QC_and_normalization.R",
            "mergedata",
            [
                '--input-files "${"::".join([str(x) for x in _input])}"',
                '--output "${_output[\'normalized_log2cpm\']}"',
            ],
            "f'{_output[0]}.stderr'",
            "f'{_output[0]}.stdout'",
            "rscript",
        ),
    })


# 30. pseudobulk_expression_QC_and_normalization.ipynb
def transform_pseudobulk_expression_QC_and_normalization():
    transform_notebook("pseudobulk_expression_QC_and_normalization.ipynb", {
        r'^\[qc_1\]': (
            "molecular_phenotypes/QC/pseudobulk_expression_QC_and_normalization.R",
            "qc_1",
            [
                '--input "${_input}"',
                '--brain-region-list "${BrainRegionList}"',
                '--region "${_region}"',
                '--low-cell-count-filter-threshold ${low_cell_count_filter_threshold}',
                '--low-expr-gene-count-filter-threshold ${low_expr_gene_count_filter_threshold}',
                '--low-expr-gene-count-gene-filter-percent ${low_expr_gene_count_gene_filter_percent}',
                '--log2cpm-gene-filter-threshold ${log2cpm_gene_filter_threshold}',
                '--log2cpm-gene-filter-percent ${log2cpm_gene_filter_percent}',
                '--output-gct "${_output}"',
            ],
            "f'{_output:nnn}.stderr'",
            "f'{_output:nnn}.log'",
            "rscript",
        ),
        r'^\[SE_qc_1\]': (
            "molecular_phenotypes/QC/pseudobulk_expression_QC_and_normalization.R",
            "SE_qc_1",
            [
                '--input "${_input}"',
                '--region "${_region}"',
                '--celltype "${_celltypes}"',
                '--output-rds "${_output[0]}"',
                '--output-gct "${_output[1]}"',
            ],
            "f'{_output[0]:nnn}.stderr'",
            "f'{_output[0]:nnn}.log'",
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

    print("\n14. mash_preprocessing.ipynb")
    transform_mash_preprocessing()

    print("\n15. rss_analysis.ipynb")
    transform_rss_analysis()

    print("\n16. splicing_calling.ipynb")
    transform_splicing_calling()

    print("\n17. methylation_calling.ipynb")
    transform_methylation_calling()

    print("\n18. qtl_association_postprocessing.ipynb")
    transform_qtl_association_postprocessing()

    print("\n19. intact.ipynb")
    transform_intact()

    print("\n20. gene_annotation.ipynb")
    transform_gene_annotation()

    print("\n21. SuSiE_enloc.ipynb")
    transform_SuSiE_enloc()

    print("\n22. qr_and_twas.ipynb")
    transform_qr_and_twas()

    print("\n23. gsea.ipynb")
    transform_gsea()

    print("\n24. eoo_enrichment.ipynb")
    transform_eoo_enrichment()

    print("\n25. apa_impute.ipynb")
    transform_apa_impute()

    print("\n26. GRM.ipynb")
    transform_GRM()

    print("\n27. METAL.ipynb")
    transform_METAL()

    print("\n28. ld_prune_reference.ipynb")
    transform_ld_prune_reference()

    print("\n29. gregor.ipynb")
    transform_gregor()

    print("\n30. pseudobulk_mega_expression_QC_and_normalization.ipynb")
    transform_pseudobulk_mega_expression_QC_and_normalization()

    print("\n31. pseudobulk_expression_QC_and_normalization.ipynb")
    transform_pseudobulk_expression_QC_and_normalization()

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
        "mash_preprocessing.ipynb",
        "rss_analysis.ipynb",
        "splicing_calling.ipynb",
        "methylation_calling.ipynb",
        "qtl_association_postprocessing.ipynb",
        "intact.ipynb",
        "gene_annotation.ipynb",
        "SuSiE_enloc.ipynb",
        "qr_and_twas.ipynb",
        "gsea.ipynb",
        "eoo_enrichment.ipynb",
        "apa_impute.ipynb",
        "GRM.ipynb",
        "METAL.ipynb",
        "ld_prune_reference.ipynb",
        "gregor.ipynb",
        "pseudobulk_mega_expression_QC_and_normalization.ipynb",
        "pseudobulk_expression_QC_and_normalization.ipynb",
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
        print("\nAll 31 notebooks created successfully!")
    else:
        print("\nSome notebooks are missing!")
