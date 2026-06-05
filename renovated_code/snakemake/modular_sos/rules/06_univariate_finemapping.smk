# ============================================================
# Rule Module 06: Univariate Fine-mapping  (Modular SoS)
# ============================================================
# Covers: Univariate SuSiE fine-mapping + TWAS weight estimation
#
# SoS notebooks called (Modular SoS wrappers in modular_sos/notebooks/):
#   - mnm_regression.ipynb (susie_twas)
#   - rss_analysis.ipynb   (univariate_plot)
# ============================================================

# ------------------------------------
# Step 6.1 — Univariate SuSiE fine-mapping + TWAS weights
# ------------------------------------
rule susie_twas:
    """Run univariate SuSiE fine-mapping and compute TWAS weights for all loci."""
    input:
        geno_list      = lambda wc: get_genotype_chrom_list(),
        pheno_region_list = lambda wc: get_phenotype_region_list(wc.theme),
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        done = "{cwd}/finemapping/{theme}/susie_twas/.done_susie_twas",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("susie_twas"),
        outer_lsf     = str(sos_uses_lsf("susie_twas")).lower(),
        outer_queue   = sos_queue("susie_twas"),
        outer_walltime = lsf_walltime(config["resources"]["finemapping"]["runtime"], sos_queue("susie_twas")),
        outer_mem_gb  = lsf_mem_gb(config["resources"]["finemapping"]["mem_mb"]),
        outer_job_name = lambda wc: f"modular_sos_susie_{wc.theme}",
        runtime_home  = RUNTIME_HOME,
        runtime_root  = RUNTIME_ROOT,
        activate_pixi = str(ACTIVATE_PIXI),
        run_workdir   = RUN_WORKDIR,
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/finemapping/{theme}/susie_twas",
        cis_window    = config["association"]["cis_window"],
        L             = config["finemapping"]["L"],
        max_L         = config["finemapping"]["max_L"],
        pip_cutoff    = config["finemapping"]["pip_cutoff"],
        min_twas_maf  = config["finemapping"]["maf"],
        small_sample_correction = "--small-sample-correction" if config["finemapping"].get("small_sample_correction", False) else "",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["finemapping"]["mem_mb"],
        runtime  = config["resources"]["finemapping"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        if [ "{params.outer_lsf}" = "true" ]; then
            outer_script="{params.outdir}/susie_twas.outer_lsf.sh"
            outer_stdout="{params.outdir}/susie_twas.outer_lsf.%J.out"
            outer_stderr="{params.outdir}/susie_twas.outer_lsf.%J.err"
            cat > "$outer_script" <<'OUTER_LSF'
#!/bin/bash
set -euo pipefail
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load singularity/3.11.0 || true
elif command -v ml >/dev/null 2>&1; then
    ml singularity/3.11.0 || true
fi
export SINGULARITY_BIND="/sc:/sc"
export XQTL_LOCAL_RUNTIME_HOME="{params.runtime_home}"
export XQTL_LOCAL_RUNTIME_ROOT="{params.runtime_root}"
source "{params.activate_pixi}"
cd "{params.run_workdir}"
{params.sos_bin} run {params.notebooks_dir}/mnm_regression.ipynb susie_twas \
    --cwd {params.outdir} \
    --name {wildcards.theme} \
    --genoFile {input.geno_list} \
    --phenoFile {input.pheno_region_list} \
    --covFile {input.hidden_factors} \
    --cis-window {params.cis_window} \
    --init-L {params.L} \
    --max-L {params.max_L} \
    --pip-cutoff {params.pip_cutoff} \
    --min_twas_maf {params.min_twas_maf} \
    {params.small_sample_correction} \
    --renovated-code-dir {params.renovated_dir} \
    --numThreads {threads} {params.dry_run} {params.sos_sched}
OUTER_LSF
            chmod +x "$outer_script"
            bsub -K \
                -q "{params.outer_queue}" \
                -P acc_load \
                -W "{params.outer_walltime}" \
                -J "{params.outer_job_name}" \
                -oo "$outer_stdout" \
                -eo "$outer_stderr" \
                -n {threads} \
                -R "span[hosts=1] rusage[mem={params.outer_mem_gb}GB]" \
                < "$outer_script"
        else
            {params.sos_bin} run {params.notebooks_dir}/mnm_regression.ipynb susie_twas \
                --cwd {params.outdir} \
                --name {wildcards.theme} \
                --genoFile {input.geno_list} \
                --phenoFile {input.pheno_region_list} \
                --covFile {input.hidden_factors} \
                --cis-window {params.cis_window} \
                --init-L {params.L} \
                --max-L {params.max_L} \
	                --pip-cutoff {params.pip_cutoff} \
	                --min_twas_maf {params.min_twas_maf} \
	                {params.small_sample_correction} \
	                --renovated-code-dir {params.renovated_dir} \
	                --numThreads {threads} {params.dry_run} {params.sos_sched}
        fi
	        python3 - "{params.outdir}" "{input.pheno_region_list}" <<'PY'
import os
import sys

outdir, region_list = sys.argv[1], sys.argv[2]
expected = 0
with open(region_list) as handle:
    for line in handle:
        if line.strip() and not line.startswith("#"):
            expected += 1

if expected == 0:
    print("ERROR: susie_twas phenotype region list has no analyzable rows", file=sys.stderr)
    sys.exit(1)

fine_dir = os.path.join(outdir, "fine_mapping")
twas_dir = os.path.join(outdir, "twas_weights")
fine_count = sum(
    1 for root, _, files in os.walk(fine_dir)
    for name in files if name.endswith(".univariate_bvsr.rds")
)
twas_count = sum(
    1 for root, _, files in os.walk(twas_dir)
    for name in files if name.endswith(".univariate_twas_weights.rds")
)

if fine_count < expected or twas_count < expected:
    print(
        "ERROR: susie_twas incomplete outputs: expected %d fine-mapping RDS and %d TWAS-weight RDS, found %d and %d"
        % (expected, expected, fine_count, twas_count),
        file=sys.stderr,
    )
    sys.exit(1)
PY
	        susie_guard_status=$?
	        if [ "$susie_guard_status" -ne 0 ]; then
	            exit "$susie_guard_status"
	        fi
	        touch {output.done}
	        """

# ------------------------------------
# Step 6.2 — Fine-mapping credible set plots
# ------------------------------------
rule finemapping_plots:
    """Generate PIP plots for fine-mapped credible sets."""
    input:
        done = "{cwd}/finemapping/{theme}/susie_twas/.done_susie_twas",
    output:
        plots_done = "{cwd}/finemapping/{theme}/susie_twas_plots/.done_plots",
    params:
        plot_script     = RENOVATED + "/pecotmr_integration/univariate_plot.R",
        finemapping_dir = "{cwd}/finemapping/{theme}/susie_twas",
        outdir          = "{cwd}/finemapping/{theme}/susie_twas_plots",
        pip_cutoff      = config["finemapping"]["pip_cutoff"],
        dry_run         = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        mapfile -t rds_files < <(find {params.finemapping_dir}/fine_mapping -name "*.univariate_bvsr.rds" | sort)
        if [ "${{#rds_files[@]}}" -eq 0 ]; then
            echo "ERROR: no fine-mapping BVSR RDS files found under {params.finemapping_dir}/fine_mapping" >&2
            exit 1
        fi
        for rds in "${{rds_files[@]}}"; do
            png_name="$(basename "$rds" .rds).png"
            Rscript {params.plot_script} \
                --input "$rds" \
                --output "{params.outdir}/$png_name"
        done
        mapfile -t plot_files < <(find {params.outdir} -maxdepth 1 -name "*.png" | sort)
        if [ "${{#plot_files[@]}}" -lt "${{#rds_files[@]}}" ]; then
            echo "ERROR: incomplete fine-mapping plots: expected ${{#rds_files[@]}}, found ${{#plot_files[@]}}" >&2
            exit 1
        fi
        touch {output.plots_done}
        """
