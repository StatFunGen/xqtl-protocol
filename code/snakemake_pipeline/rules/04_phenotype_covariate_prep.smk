# ============================================================
# Rule Module 04: Phenotype & Covariate Preparation
# ============================================================
# Covers:
#   - Merge genotype PCs with fixed covariates
#   - Partition phenotype BED by chromosome
#   - Regress covariates out of phenotype (residualized expression)
#   - Compute hidden confounding factors (Marchenko PCA or PEER)
#
# Mirrors: eQTL_analysis_commands.ipynb stages:
#   phenotype_partition_by_chrom → merge_pca_covariate →
#   resid_exp → factor
#
# SoS notebooks called:
#   - pipeline/phenotype_formatting.ipynb (partition_by_chrom)
#   - pipeline/covariate_formatting.ipynb (merge_pca_covariate, compute_residual)
#   - pipeline/covariate_hidden_factor.ipynb (Marchenko_pc, PEER)
# ============================================================

# ------------------------------------
# Step 4.1 — Merge genotype PCs with fixed covariates
# ------------------------------------
# Combines the projected PCA scores with the user-supplied covariate file
# (sex, age, batch variables, etc.) into a single covariate matrix.
rule merge_pca_covariate:
    """Merge projected genotype PCs with phenotype-level fixed covariates."""
    input:
        projected_rds = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.pca.projected.rds",
        covariate_file = lambda wc: next(
            t["covariate_file"] for t in config["themes"] if t["name"] == wc.theme
        ),
    output:
        merged_cov = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/covariates",
        n_pcs        = config["covariate"]["n_pcs"],
        tol_cov      = config["covariate"]["tol_cov"],
        mean_impute  = lambda _: "--mean-impute" if config["covariate"]["mean_impute"] else "",
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/covariate_formatting.ipynb merge_pca_covariate \
            --cwd {params.outdir} \
            --pcaFile {input.projected_rds} \
            --covFile {input.covariate_file} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --k {params.n_pcs} \
            --tol-cov {params.tol_cov} \
            {params.mean_impute} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.2 — Partition phenotype BED file by chromosome
# ------------------------------------
# Splits the normalized expression BED file into per-chromosome files,
# then writes a file-list used by TensorQTL for parallel CIS analysis.
rule phenotype_by_chrom:
    """Partition phenotype BED.gz into per-chromosome files for TensorQTL."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        chrom_recipe = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.per_chrom.recipe",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["rnaquant"],
        outdir       = "{cwd}/data_preprocessing/{theme}/phenotype_data",
        chroms       = " ".join(config["chromosomes"]),
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/phenotype_formatting.ipynb partition_by_chrom \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --chrom {params.chroms} \
            --container {params.container} \
            --numThreads {threads}
        # Write recipe file listing per-chrom phenotype paths
        ls {params.outdir}/{wildcards.theme}.*.bed.gz > {output.chrom_recipe}
        """

# ------------------------------------
# Step 4.3 — Compute residual expression
# ------------------------------------
# Regresses the merged PCA+covariate matrix out of the expression phenotype.
# The residuals are then used for hidden factor estimation.
rule compute_residual:
    """Regress merged covariates (genotype PCs + fixed covariates) out of expression."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        residual_bed = "{cwd}/data_preprocessing/{theme}/resid_phenotype/{theme}.cov_pca.resid.bed.gz",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/resid_phenotype",
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/covariate_hidden_factor.ipynb computing_residual \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.4a — Hidden factor analysis: Marchenko-Pastur PCA (default)
# ------------------------------------
# Uses the Marchenko-Pastur law to automatically determine the optimal
# number of principal components to extract from the residualized expression.
rule marchenko_pc:
    """Estimate hidden confounding factors via Marchenko-Pastur PCA."""
    input:
        residual_bed = "{cwd}/data_preprocessing/{theme}/resid_phenotype/{theme}.cov_pca.resid.bed.gz",
        merged_cov   = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{theme}.cov_pca.resid.Marchenko_PC.gz",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["pcatools"],
        outdir       = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors    = config["hidden_factors"]["n_factors"],
    threads: config["resources"]["hidden_factors"]["threads"]
    resources:
        mem_mb   = config["resources"]["hidden_factors"]["mem_mb"],
        walltime = config["resources"]["hidden_factors"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/covariate_hidden_factor.ipynb Marchenko_pc \
            --cwd {params.outdir} \
            --phenoFile {input.residual_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.4b — Hidden factor analysis: PEER (alternative method)
# ------------------------------------
# Uses Probabilistic Estimation of Expression Residuals (PEER) to learn
# hidden factor structure. Activated only when config hidden_factors.method = "PEER".
rule peer_factors:
    """Estimate hidden confounding factors via PEER factor analysis."""
    input:
        residual_bed = "{cwd}/data_preprocessing/{theme}/resid_phenotype/{theme}.cov_pca.resid.bed.gz",
        merged_cov   = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{theme}.cov_pca.resid.PEER.gz",
    params:
        pipeline_dir    = config["pipeline_dir"],
        container       = config["containers"]["peer"],
        outdir          = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors       = config["hidden_factors"]["n_factors"],
        iterations      = config["hidden_factors"]["peer_iterations"],
        convergence     = config["hidden_factors"]["peer_convergence"],
    threads: config["resources"]["hidden_factors"]["threads"]
    resources:
        mem_mb   = config["resources"]["hidden_factors"]["mem_mb"],
        walltime = config["resources"]["hidden_factors"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/covariate_hidden_factor.ipynb PEER \
            --cwd {params.outdir} \
            --phenoFile {input.residual_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            --iteration {params.iterations} \
            --convergence-mode {params.convergence} \
            --container {params.container} \
            --numThreads {threads}
        """
