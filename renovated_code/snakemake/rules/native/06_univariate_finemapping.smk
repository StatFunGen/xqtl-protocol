# ============================================================
# Rule Module 06: Univariate Fine-mapping (native scripts)
# ============================================================
# NOTE: Native scripts for mnm_regression and rss_analysis are not yet
# implemented in renovated_code/. Rules here fall back to the SoS backend
# until standalone scripts are added. Remove the fallback sos run lines
# and replace with the renovated script path once available.
#
# Expected future script locations:
#   {renovated_code_dir}/mnm_analysis/mnm_regression.sh
#   {renovated_code_dir}/mnm_analysis/rss_analysis.sh
# ============================================================

RENOVATED = config["renovated_code_dir"]

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
        pipeline_dir = config["pipeline_dir"],   # SoS fallback until native script exists
        container    = config["containers"]["susie"],
        outdir       = "{cwd}/finemapping/{theme}/susie_twas",
        L            = config["finemapping"]["L"],
        max_L        = config["finemapping"]["max_L"],
        pip_cutoff   = config["finemapping"]["pip_cutoff"],
        min_twas_maf = config["finemapping"]["maf"],
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["finemapping"]["threads"]
    resources:
        mem_mb  = config["resources"]["finemapping"]["mem_mb"],
        runtime = config["resources"]["finemapping"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        # TODO: replace with native script when available:
        #   bash {RENOVATED}/mnm_analysis/mnm_regression.sh susie_twas ...
        sos run {params.pipeline_dir}/mnm_regression.ipynb susie_twas \
            --cwd {params.outdir} \
            --genoFile {input.geno_list} \
            --phenoFile {input.pheno_list} \
            --covFile {input.hidden_factors} \
            --L {params.L} \
            --max-L {params.max_L} \
            --pip-cutoff {params.pip_cutoff} \
            --min_twas_maf {params.min_twas_maf} \
            --container {params.container} \
            --numThreads {threads}
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
        pipeline_dir    = config["pipeline_dir"],   # SoS fallback until native script exists
        container       = config["containers"]["susie"],
        finemapping_dir = "{cwd}/finemapping/{theme}/susie_twas",
        outdir          = "{cwd}/finemapping/{theme}/susie_twas_plots",
        pip_cutoff      = config["finemapping"]["pip_cutoff"],
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        # TODO: replace with native script when available:
        #   bash {RENOVATED}/mnm_analysis/rss_analysis.sh univariate_plot ...
        sos run {params.pipeline_dir}/rss_analysis.ipynb univariate_plot \
            --cwd {params.outdir} \
            --finemapping-dir {params.finemapping_dir} \
            --pip-cutoff {params.pip_cutoff} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.plots_done}
        """
