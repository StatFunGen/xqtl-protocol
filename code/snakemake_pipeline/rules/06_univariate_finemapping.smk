# ============================================================
# Rule Module 06: Univariate Fine-mapping (SuSiE via mnm_regression)
# ============================================================
# Covers: Univariate SuSiE fine-mapping + TWAS weight estimation
#
# Mirrors: eQTL_analysis_commands.ipynb and mnm_regression.ipynb
#
# SoS notebook called:
#   - pipeline/mnm_regression.ipynb (susie_twas)
#
# Design note:
#   rss_analysis.ipynb is for LD-reference-panel-based RSS fine-mapping
#   (i.e., when you only have summary statistics + an external LD panel).
#   For genotype-based fine-mapping (the default in this protocol), the
#   correct notebook is mnm_regression.ipynb::susie_twas, which takes the
#   same genotype/phenotype/covariate inputs as TensorQTL and runs SuSiE
#   directly on individual-level data.
#
# Dependency:
#   tensorqtl_cis (done file) → susie_twas → finemapping_plots
# ============================================================

# ------------------------------------
# Step 6.1 — Univariate SuSiE fine-mapping + TWAS weights
# ------------------------------------
# Runs SuSiE on each cis-QTL region (defined by the TensorQTL output)
# and simultaneously estimates TWAS weights via cross-validation.
#
# Inputs:
#   --genoFile:    genotype_by_chrom_files.txt  (per-chrom plink list)
#   --phenoFile:   phenotype_by_chrom_files.txt (per-chrom BED list)
#   --covFile:     hidden factor file (Marchenko PC or PEER)
#   tensorqtl_done: ensures TensorQTL finishes before fine-mapping
#
# Output: sentinel done file; actual RDS results are in {outdir}/susie_twas/
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
        pipeline_dir  = config["pipeline_dir"],
        container     = config["containers"]["susie"],
        outdir        = "{cwd}/finemapping/{theme}/susie_twas",
        L             = config["finemapping"]["L"],
        max_L         = config["finemapping"]["max_L"],
        pip_cutoff    = config["finemapping"]["pip_cutoff"],
        maf           = config["finemapping"]["maf"],
        # min_twas_maf uses underscore (SoS preserves it for this parameter)
        min_twas_maf  = config["finemapping"]["maf"],
    threads: config["resources"]["finemapping"]["threads"]
    resources:
        mem_mb   = config["resources"]["finemapping"]["mem_mb"],
        walltime = config["resources"]["finemapping"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
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
# Generates PIP lollipop plots for each credible set identified during
# SuSiE fine-mapping. Uses the univariate_plot step from rss_analysis.ipynb,
# which reads the RDS output from susie_twas.
rule finemapping_plots:
    """Generate PIP plots for fine-mapped credible sets."""
    input:
        done     = "{cwd}/finemapping/{theme}/susie_twas/.done_susie_twas",
    output:
        plots_done = "{cwd}/finemapping/{theme}/susie_twas_plots/.done_plots",
    params:
        pipeline_dir    = config["pipeline_dir"],
        container       = config["containers"]["susie"],
        finemapping_dir = "{cwd}/finemapping/{theme}/susie_twas",
        outdir          = "{cwd}/finemapping/{theme}/susie_twas_plots",
        pip_cutoff      = config["finemapping"]["pip_cutoff"],
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/rss_analysis.ipynb univariate_plot \
            --cwd {params.outdir} \
            --finemapping-dir {params.finemapping_dir} \
            --pip-cutoff {params.pip_cutoff} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.plots_done}
        """
