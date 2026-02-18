# ============================================================
# Rule Module 04: Phenotype & Covariate Preparation
# ============================================================
# Covers:
#   - Merge genotype PCs with fixed covariates
#   - Partition phenotype BED by chromosome
#   - Compute hidden confounding factors (Marchenko PCA or PEER)
#
# Mirrors: eQTL_analysis_commands.ipynb stages:
#   phenotype_partition_by_chrom → merge_pca_covariate → factor
#
# SoS notebooks called:
#   - pipeline/phenotype_formatting.ipynb (phenotype_by_chrom)
#   - pipeline/covariate_formatting.ipynb (merge_genotype_pc)
#   - pipeline/covariate_hidden_factor.ipynb (Marchenko_PC, PEER)
#
# NOTE on hidden factors:
#   When you run `sos run covariate_hidden_factor.ipynb Marchenko_PC` (or PEER),
#   the SoS workflow internally runs a shared *_1 step that computes residuals
#   from the merged covariates, then runs the factor step on those residuals.
#   There is therefore NO separate compute_residual rule — the factor rules
#   take the original phenotype BED + merged covariates as input directly.
# ============================================================

# ------------------------------------
# Step 4.1 — Merge genotype PCs with fixed covariates
# ------------------------------------
# Combines the projected PCA scores with the user-supplied covariate file
# (sex, age, batch variables, etc.) into a single covariate matrix.
# Actual SoS step name: merge_genotype_pc (not merge_pca_covariate).
rule merge_pca_covariate:
    """Merge projected genotype PCs with phenotype-level fixed covariates."""
    input:
        projected_rds  = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.pca.projected.rds",
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
        sos run {params.pipeline_dir}/covariate_formatting.ipynb merge_genotype_pc \
            --cwd {params.outdir} \
            --pcaFile {input.projected_rds} \
            --covFile {input.covariate_file} \
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
# Actual SoS step name: phenotype_by_chrom (not partition_by_chrom).
# The SoS notebook produces: {name}.phenotype_by_chrom_files.txt
rule phenotype_by_chrom:
    """Partition phenotype BED.gz into per-chromosome files for TensorQTL."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        chrom_list = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.phenotype_by_chrom_files.txt",
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
        sos run {params.pipeline_dir}/phenotype_formatting.ipynb phenotype_by_chrom \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --name {wildcards.theme} \
            --chrom {params.chroms} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.3a — Hidden factor analysis: Marchenko-Pastur PCA (default)
# ------------------------------------
# Calling `sos run covariate_hidden_factor.ipynb Marchenko_PC` runs two
# internal sub-steps:
#   *_1: regress merged covariates out of phenotype → residual BED
#   Marchenko_PC_2: apply Marchenko-Pastur law to residuals → PC factor file
# Both steps run automatically when you invoke the Marchenko_PC workflow.
# Note: step name is Marchenko_PC (capital PC), NOT Marchenko_pc.
# Note: --mean-impute-missing is the flag name (not --mean-impute).
rule marchenko_pc:
    """Estimate hidden confounding factors via Marchenko-Pastur PCA."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{theme}.Marchenko_PC.gz",
    params:
        pipeline_dir      = config["pipeline_dir"],
        container         = config["containers"]["pcatools"],
        outdir            = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors         = config["hidden_factors"]["n_factors"],
        mean_impute_flag  = lambda _: "--mean-impute-missing" if config["covariate"]["mean_impute"] else "",
    threads: config["resources"]["hidden_factors"]["threads"]
    resources:
        mem_mb   = config["resources"]["hidden_factors"]["mem_mb"],
        walltime = config["resources"]["hidden_factors"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/covariate_hidden_factor.ipynb Marchenko_PC \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            {params.mean_impute_flag} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.3b — Hidden factor analysis: PEER (alternative method)
# ------------------------------------
# Calling `sos run covariate_hidden_factor.ipynb PEER` runs:
#   *_1: regress merged covariates out of phenotype → residual BED
#   PEER_2: PEER factor analysis → .PEER_MODEL.hd5
#   PEER_3: Extract PEER factors → .PEER.gz
# All three steps run automatically when you invoke the PEER workflow.
rule peer_factors:
    """Estimate hidden confounding factors via PEER factor analysis."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{theme}.PEER.gz",
    params:
        pipeline_dir     = config["pipeline_dir"],
        container        = config["containers"]["peer"],
        outdir           = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors        = config["hidden_factors"]["n_factors"],
        iterations       = config["hidden_factors"]["peer_iterations"],
        convergence_mode = config["hidden_factors"]["peer_convergence"],
    threads: config["resources"]["hidden_factors"]["threads"]
    resources:
        mem_mb   = config["resources"]["hidden_factors"]["mem_mb"],
        walltime = config["resources"]["hidden_factors"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/covariate_hidden_factor.ipynb PEER \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            --iteration {params.iterations} \
            --convergence_mode {params.convergence_mode} \
            --container {params.container} \
            --numThreads {threads}
        """
