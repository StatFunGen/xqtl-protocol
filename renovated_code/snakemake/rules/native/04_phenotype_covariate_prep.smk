# ============================================================
# Rule Module 04: Phenotype & Covariate Preparation (native scripts)
# ============================================================
# Calls renovated_code/ shell wrappers instead of SoS notebooks.
# Interface flags are identical to the SoS versions.
# ============================================================

RENOVATED = config["renovated_code_dir"]

# ------------------------------------
# Step 4.1 — Merge genotype PCs with fixed covariates
# ------------------------------------
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
        container   = config["containers"]["bioinfo"],
        outdir      = "{cwd}/data_preprocessing/{theme}/covariates",
        n_pcs       = config["covariate"]["n_pcs"],
        tol_cov     = config["covariate"]["tol_cov"],
        mean_impute = lambda _: "--mean-impute" if config["covariate"]["mean_impute"] else "",
        script      = f"{RENOVATED}/data_preprocessing/covariate/covariate_formatting.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} merge_genotype_pc {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --pcaFile {input.projected_rds} \
            --covFile {input.covariate_file} \
            --k {params.n_pcs} \
            --tol-cov {params.tol_cov} \
            {params.mean_impute} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.2 — Partition phenotype BED file by chromosome
# ------------------------------------
rule phenotype_by_chrom:
    """Partition phenotype BED.gz into per-chromosome files for TensorQTL."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        chrom_list = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.phenotype_by_chrom_files.txt",
    params:
        container = config["containers"]["rnaquant"],
        outdir    = "{cwd}/data_preprocessing/{theme}/phenotype_data",
        chroms    = " ".join(config["chromosomes"]),
        script    = f"{RENOVATED}/data_preprocessing/phenotype/phenotype_formatting.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} phenotype_by_chrom {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --name {wildcards.theme} \
            --chrom {params.chroms} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.3a — Hidden factor analysis: Marchenko-Pastur PCA
# ------------------------------------
rule marchenko_pc:
    """Estimate hidden confounding factors via Marchenko-Pastur PCA."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{theme}.Marchenko_PC.gz",
    params:
        container        = config["containers"]["pcatools"],
        outdir           = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors        = config["hidden_factors"]["n_factors"],
        mean_impute_flag = lambda _: "--mean-impute-missing" if config["covariate"]["mean_impute"] else "",
        script           = f"{RENOVATED}/data_preprocessing/covariate/covariate_hidden_factor.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["hidden_factors"]["threads"]
    resources:
        mem_mb  = config["resources"]["hidden_factors"]["mem_mb"],
        runtime = config["resources"]["hidden_factors"]["runtime"],
    shell:
        """
        bash {params.script} Marchenko_PC {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            {params.mean_impute_flag} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 4.3b — Hidden factor analysis: PEER
# ------------------------------------
rule peer_factors:
    """Estimate hidden confounding factors via PEER factor analysis."""
    input:
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
        merged_cov    = "{cwd}/data_preprocessing/{theme}/covariates/xqtl_protocol_data.plink_qc.{theme}.pca.gz",
    output:
        hidden_factors = "{cwd}/data_preprocessing/{theme}/covariates/{theme}.PEER.gz",
    params:
        container        = config["containers"]["peer"],
        outdir           = "{cwd}/data_preprocessing/{theme}/covariates",
        n_factors        = config["hidden_factors"]["n_factors"],
        iterations       = config["hidden_factors"]["peer_iterations"],
        convergence_mode = config["hidden_factors"]["peer_convergence"],
        script           = f"{RENOVATED}/data_preprocessing/covariate/covariate_hidden_factor.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["hidden_factors"]["threads"]
    resources:
        mem_mb  = config["resources"]["hidden_factors"]["mem_mb"],
        runtime = config["resources"]["hidden_factors"]["runtime"],
    shell:
        """
        bash {params.script} PEER {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            --covFile {input.merged_cov} \
            --N {params.n_factors} \
            --iteration {params.iterations} \
            --convergence-mode {params.convergence_mode} \
            --numThreads {threads}
        """
