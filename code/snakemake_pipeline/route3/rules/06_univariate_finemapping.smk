# ============================================================
# Rule Module 06: Univariate Fine-mapping  (Route 3)
# ============================================================
# Covers: Univariate SuSiE fine-mapping + TWAS weight estimation
#
# SoS notebooks called (Route 3 wrappers in route3/notebooks/):
#   - mnm_regression.ipynb (susie_twas)   — copied as-is; native script pending
#   - rss_analysis.ipynb   (univariate_plot) — copied as-is; native script pending
#
# When renovated_code/mnm_analysis/mnm_regression.sh and rss_analysis.sh
# are implemented, the notebook task blocks will be updated to call them.
# ============================================================

# ------------------------------------
# Step 6.1 — Univariate SuSiE fine-mapping + TWAS weights
# ------------------------------------
rule susie_twas:
    """Run univariate SuSiE fine-mapping and compute TWAS weights for all loci."""
    input:
        tensorqtl_done = "{cwd}/association_scan/{theme}/TensorQTL/.done_cis",
        geno_list      = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.genotype_by_chrom_files.txt",
        pheno_list     = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.phenotype_by_chrom_files.txt",
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        done = "{cwd}/finemapping/{theme}/susie_twas/.done_susie_twas",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        container     = config["containers"]["susie"],
        outdir        = "{cwd}/finemapping/{theme}/susie_twas",
        cis_window    = config["association"]["cis_window"],
        L             = config["finemapping"]["L"],
        max_L         = config["finemapping"]["max_L"],
        pip_cutoff    = config["finemapping"]["pip_cutoff"],
        min_twas_maf  = config["finemapping"]["maf"],
        dry_run       = DRY_RUN_SOS,
    threads: config["resources"]["finemapping"]["threads"]
    resources:
        mem_mb   = config["resources"]["finemapping"]["mem_mb"],
        runtime  = config["resources"]["finemapping"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        PHENO_BEDS=$(awk 'NR>1 {{print $2}}' {input.pheno_list} | tr '\n' ' ')
        N_PHENO=$(awk 'END{{print NR-1}}' {input.pheno_list})
        COV_FILES=$(python3 -c "print(' '.join(['{input.hidden_factors}'] * $N_PHENO))")
        sos run {params.notebooks_dir}/mnm_regression.ipynb susie_twas \
            --cwd {params.outdir} \
            --name {wildcards.theme} \
            --genoFile {input.geno_list} \
            --phenoFile $PHENO_BEDS \
            --covFile $COV_FILES \
            --cis-window {params.cis_window} \
            --init-L {params.L} \
            --max-L {params.max_L} \
            --pip-cutoff {params.pip_cutoff} \
            --min_twas_maf {params.min_twas_maf} \
            --container {params.container} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
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
        notebooks_dir   = NOTEBOOKS,
        renovated_dir   = RENOVATED,
        container       = config["containers"]["susie"],
        finemapping_dir = "{cwd}/finemapping/{theme}/susie_twas",
        outdir          = "{cwd}/finemapping/{theme}/susie_twas_plots",
        pip_cutoff      = config["finemapping"]["pip_cutoff"],
        dry_run         = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        find {params.finemapping_dir} -name "*.rds" | sort | while IFS= read -r rds; do
            sos run {params.notebooks_dir}/rss_analysis.ipynb univariate_plot \
                "$rds" \
                --cwd {params.outdir} \
                --container {params.container} \
                --renovated-code-dir {params.renovated_dir} \
                --numThreads {threads} {params.dry_run}
        done
        touch {output.plots_done}
        """
