# ============================================================
# Rule Module 04: Phenotype & Covariate Preparation  (Modular SoS)
# ============================================================
# Covers:
#   - Merge genotype PCs with fixed covariates
#   - Partition phenotype BED by chromosome
#   - Compute hidden confounding factors (Marchenko PCA or PEER)
#
# SoS notebooks called (Modular SoS wrappers in modular_sos/notebooks/):
#   - covariate_formatting.ipynb    (merge_genotype_pc)
#   - phenotype_formatting.ipynb    (phenotype_by_chrom)
#   - covariate_hidden_factor.ipynb (Marchenko_PC, PEER)
# ============================================================

# ------------------------------------
# Step 4.1 — Merge genotype PCs with fixed covariates
# ------------------------------------
rule merge_pca_covariate:
    """Merge projected genotype PCs with phenotype-level fixed covariates."""
    input:
        projected_rds  = lambda wc: get_projected_pca_rds(wc.theme),
        covariate_file = lambda wc: next(
            t["covariate_file"] for t in config["themes"] if t["name"] == wc.theme
        ),
    output:
        merged_cov = "{cwd}/data_preprocessing/{theme}/covariates/{merged_cov_base}.pca.gz",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("merge_pca_covariate"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/covariates",
        output_name   = lambda wc: f"{get_merged_cov_base(wc.theme)}.pca",
        n_pcs         = config["covariate"]["n_pcs"],
        tol_cov       = config["covariate"]["tol_cov"],
        mean_impute   = lambda _: "--mean-impute" if config["covariate"]["mean_impute"] else "",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        {params.sos_bin} run {params.notebooks_dir}/covariate_formatting.ipynb merge_genotype_pc \
            --cwd {params.outdir} \
            --pcaFile {input.projected_rds} \
            --covFile {input.covariate_file} \
            --name {params.output_name} \
            --k {params.n_pcs} \
            --tol-cov {params.tol_cov} \
            {params.mean_impute} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """

# ------------------------------------
# Step 4.2 — Partition phenotype BED file by chromosome
# ------------------------------------
rule phenotype_by_chrom:
    """Partition phenotype BED.gz into per-chromosome files for TensorQTL."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        chrom_list   = "{cwd}/data_preprocessing/{theme}/phenotype_data/{pheno_base}.phenotype_by_chrom_files.txt",
        region_list  = "{cwd}/data_preprocessing/{theme}/phenotype_data/{pheno_base}.phenotype_by_chrom_files.region_list.txt",
    params:
        sos_bin       = SOS_BIN,
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/phenotype_data",
        output_name   = "{pheno_base}",
        chroms        = " ".join(config["chromosomes"]),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        {params.sos_bin} run {params.notebooks_dir}/phenotype_formatting.ipynb phenotype_by_chrom \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --name {params.output_name} \
            --chrom {params.chroms} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 4.3a — Hidden factors: Marchenko-Pastur PCA (default)
# ------------------------------------
rule marchenko_pc:
    """Estimate hidden confounding factors via Marchenko-Pastur PCA."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = lambda wc: get_merged_cov_path(wc.theme),
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{hidden_factor_base}.Marchenko_PC.gz",
    params:
        sos_bin          = SOS_BIN,
        sos_sched        = sos_sched("marchenko_pc"),
        notebooks_dir    = NOTEBOOKS,
        renovated_dir    = RENOVATED,
        outdir           = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors        = config["hidden_factors"]["n_factors"],
        mean_impute_flag = lambda _: "--mean-impute-missing" if config["covariate"]["mean_impute"] else "",
        dry_run          = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["hidden_factors"]["mem_mb"],
        runtime  = config["resources"]["hidden_factors"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/covariate_hidden_factor.ipynb Marchenko_PC \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            {params.mean_impute_flag} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """

# ------------------------------------
# Step 4.3b — Hidden factors: PEER (alternative method)
# ------------------------------------
rule peer_factors:
    """Estimate hidden confounding factors via PEER factor analysis."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = lambda wc: get_merged_cov_path(wc.theme),
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{hidden_factor_base}.PEER.gz",
    params:
        sos_bin          = SOS_BIN,
        sos_sched        = sos_sched("peer_factors"),
        notebooks_dir    = NOTEBOOKS,
        renovated_dir    = RENOVATED,
        outdir           = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors        = config["hidden_factors"]["n_factors"],
        iterations       = config["hidden_factors"]["peer_iterations"],
        convergence_mode = config["hidden_factors"]["peer_convergence"],
        dry_run          = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["hidden_factors"]["mem_mb"],
        runtime  = config["resources"]["hidden_factors"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/covariate_hidden_factor.ipynb PEER \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            --iteration {params.iterations} \
            --convergence_mode {params.convergence_mode} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """
