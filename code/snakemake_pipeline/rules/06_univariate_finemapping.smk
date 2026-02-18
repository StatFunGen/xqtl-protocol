# ============================================================
# Rule Module 06: Univariate Fine-mapping (SuSiE / RSS)
# ============================================================
# Covers: Prepare analysis regions → SuSiE fine-mapping with RSS →
#         Generate credible set plots
#
# Mirrors: code/mnm_analysis/mnm_methods/rss_analysis.ipynb
#          (univariate_fine_mapping step via mnm_regression.ipynb susie_twas)
#
# SoS notebooks called:
#   - pipeline/mnm_regression.ipynb (susie_twas)
#   - pipeline/rss_analysis.ipynb (get_analysis_regions, univariate_fine_mapping, rss_susie_plot)
# ============================================================

# ------------------------------------
# Step 6.1 — Extract per-region data for fine-mapping
# ------------------------------------
# Reads the CIS-QTL significant regions from TensorQTL output and
# prepares genotype LD matrices, summary statistics, and region
# boundaries for each significant locus.
rule get_analysis_regions:
    """Extract regional genotype and summary statistic data for fine-mapping."""
    input:
        sig_table = "{cwd}/association_scan/{theme}/TensorQTL/TensorQTL.cis.regional_significance.tsv.gz",
        geno_list = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.plink_files_list.txt",
        hidden_factors = lambda wc: get_hidden_factors(wc),
        phenotype_bed  = lambda wc: get_phenotype_bed(wc),
    output:
        regions_done = "{cwd}/finemapping/{theme}/univariate/.done_regions",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        outdir       = "{cwd}/finemapping/{theme}/univariate",
        cis_window   = config["association"]["cis_window"],
        maf          = config["finemapping"]["maf"],
    threads: config["resources"]["finemapping"]["threads"]
    resources:
        mem_mb   = config["resources"]["finemapping"]["mem_mb"],
        walltime = config["resources"]["finemapping"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/rss_analysis.ipynb get_analysis_regions \
            --cwd {params.outdir} \
            --genoFile {input.geno_list} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.hidden_factors} \
            --sumstat-file {input.sig_table} \
            --cis-window {params.cis_window} \
            --maf {params.maf} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.regions_done}
        """

# ------------------------------------
# Step 6.2 — Univariate SuSiE fine-mapping with RSS
# ------------------------------------
# Runs SuSiE with Regression on Summary Statistics (RSS) for each
# significant region. Estimates posterior inclusion probabilities (PIPs)
# and credible sets.
#
# Also runs the susie_twas step from mnm_regression.ipynb to compute
# TWAS weights alongside fine-mapping.
rule univariate_finemapping:
    """Run SuSiE-RSS univariate fine-mapping on all significant CIS-QTL regions."""
    input:
        regions_done = "{cwd}/finemapping/{theme}/univariate/.done_regions",
        geno_list    = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.plink_files_list.txt",
        hidden_factors = lambda wc: get_hidden_factors(wc),
        phenotype_bed  = lambda wc: get_phenotype_bed(wc),
    output:
        finemapping_done = "{cwd}/finemapping/{theme}/univariate/.done_finemapping",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        outdir       = "{cwd}/finemapping/{theme}/univariate",
        L            = config["finemapping"]["L"],
        max_L        = config["finemapping"]["max_L"],
        pip_cutoff   = config["finemapping"]["pip_cutoff"],
        coverage     = " ".join(str(c) for c in config["finemapping"]["coverage"]),
        maf          = config["finemapping"]["maf"],
        rcond        = config["finemapping"]["rcond"],
    threads: config["resources"]["finemapping"]["threads"]
    resources:
        mem_mb   = config["resources"]["finemapping"]["mem_mb"],
        walltime = config["resources"]["finemapping"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/rss_analysis.ipynb univariate_fine_mapping \
            --cwd {params.outdir} \
            --genoFile {input.geno_list} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.hidden_factors} \
            --L {params.L} \
            --max-L {params.max_L} \
            --pip-cutoff {params.pip_cutoff} \
            --coverage {params.coverage} \
            --maf {params.maf} \
            --rcond {params.rcond} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.finemapping_done}
        """

# ------------------------------------
# Step 6.3 — SuSiE TWAS weights (from mnm_regression.ipynb)
# ------------------------------------
# Computes TWAS (Transcriptome-Wide Association Study) weights
# alongside the SuSiE fine-mapping. The weights are derived from
# cross-validated prediction models of expression.
rule susie_twas_weights:
    """Compute SuSiE-derived TWAS weights for all fine-mapped regions."""
    input:
        finemapping_done = "{cwd}/finemapping/{theme}/univariate/.done_finemapping",
        geno_list        = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.plink_files_list.txt",
        hidden_factors   = lambda wc: get_hidden_factors(wc),
        phenotype_bed    = lambda wc: get_phenotype_bed(wc),
    output:
        twas_done = "{cwd}/finemapping/{theme}/univariate_twas/.done_twas",
    params:
        pipeline_dir   = config["pipeline_dir"],
        container      = config["containers"]["susie"],
        outdir         = "{cwd}/finemapping/{theme}/univariate_twas",
        L              = config["finemapping"]["L"],
        max_L          = config["finemapping"]["max_L"],
        pip_cutoff     = config["finemapping"]["pip_cutoff"],
        maf            = config["finemapping"]["maf"],
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
            --phenoFile {input.phenotype_bed} \
            --covFile {input.hidden_factors} \
            --L {params.L} \
            --max-L {params.max_L} \
            --pip-cutoff {params.pip_cutoff} \
            --maf {params.maf} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.twas_done}
        """

# ------------------------------------
# Step 6.4 — Credible set visualization plots
# ------------------------------------
# Generates PIP lollipop plots and Z-score plots for each
# credible set identified during fine-mapping.
rule finemapping_plots:
    """Generate PIP and Z-score plots for fine-mapped credible sets."""
    input:
        finemapping_done = "{cwd}/finemapping/{theme}/univariate/.done_finemapping",
    output:
        plots_done = "{cwd}/finemapping/{theme}/univariate_plots/.done_plots",
    params:
        pipeline_dir   = config["pipeline_dir"],
        container      = config["containers"]["susie"],
        finemapping_dir = "{cwd}/finemapping/{theme}/univariate",
        outdir          = "{cwd}/finemapping/{theme}/univariate_plots",
        pip_cutoff      = config["finemapping"]["pip_cutoff"],
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/rss_analysis.ipynb rss_susie_plot \
            --cwd {params.outdir} \
            --finemapping-dir {params.finemapping_dir} \
            --pip-cutoff {params.pip_cutoff} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.plots_done}
        """
