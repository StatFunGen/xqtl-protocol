# ============================================================
# Rule Module 02: TWAS and Colocalization Analyses
# ============================================================
# These rules are GWAS-dependent and only execute when
#   config["gwas"]["enabled"] is true.
#
# Analyses included:
#   2.1  TWAS           — cTWAS gene-level trait association
#   2.2  SuSiE-ENLOC    — Bayesian xQTL–GWAS colocalization
#                         (prior enrichment step + colocalization step)
#   2.3  ColocBoost     — Multi-trait colocalization using
#                         individual-level xQTL data
#
# SoS notebooks called:
#   - pipeline/twas_ctwas.ipynb       (twas)
#   - pipeline/SuSiE_enloc.ipynb      (xqtl_gwas_enrichment, susie_coloc)
#   - pipeline/colocboost.ipynb       (colocboost)
#
# Required per-theme config keys (under themes[].gwas):
#   xqtl_meta_data   — TSV: study_id, context, region, rds_path
#                      (points to the susie_twas *.rds files)
#   gwas_meta_data   — TSV: study_id, chrom, sumstats_path, n_sample
#   ld_meta_data     — TSV: chrom, start, end, ld_path
#   region_list      — BED file with analysis regions
#
# Additional per-theme keys required only for ColocBoost
# (individual-level data for joint fine-mapping):
#   geno_file        — path to genotype_by_chrom_files.txt
#   pheno_file       — path to {theme}.phenotype_by_chrom_files.txt
#   cov_file         — path to hidden-factor covariate file
# ============================================================


# ------------------------------------
# Step 2.1 — TWAS / cTWAS
# ------------------------------------
# Tests gene-trait associations by imputing gene expression into
# GWAS using the per-theme finemapping TWAS weights.
# Uses twas_ctwas.ipynb::twas.
rule twas:
    """Run TWAS gene-trait association for one tissue against all GWAS traits."""
    input:
        tsv_done     = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
        xqtl_meta   = lambda wc: _get_theme_gwas_param(wc, "xqtl_meta_data"),
        gwas_meta   = lambda wc: _get_theme_gwas_param(wc, "gwas_meta_data"),
        ld_meta     = lambda wc: _get_theme_gwas_param(wc, "ld_meta_data"),
        region_list = lambda wc: _get_theme_gwas_param(wc, "region_list"),
    output:
        done = "{cwd}/twas/{theme}/.done_twas",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        outdir       = "{cwd}/twas/{theme}",
        name         = "{theme}",
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["twas"]["threads"]
    resources:
        mem_mb  = config["resources"]["twas"]["mem_mb"],
        runtime = config["resources"]["twas"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/twas_ctwas.ipynb twas {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --xqtl_meta_data {input.xqtl_meta} \\
            --gwas_meta_data {input.gwas_meta} \\
            --ld_meta_data {input.ld_meta} \\
            --regions {input.region_list} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 2.2a — SuSiE-ENLOC enrichment priors
# ------------------------------------
# Estimates enrichment of xQTL signals in GWAS associations to
# derive data-driven colocalization priors (p1, p2, p12).
# This step is optional: set config["gwas"]["coloc"]["skip_enrich"]
# to true to use default priors and skip straight to susie_coloc.
rule susie_enloc_enrichment:
    """Estimate xQTL–GWAS enrichment priors for SuSiE-ENLOC colocalization."""
    input:
        tsv_done    = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
        xqtl_meta  = lambda wc: _get_theme_gwas_param(wc, "xqtl_meta_data"),
        gwas_meta  = lambda wc: _get_theme_gwas_param(wc, "gwas_meta_data"),
        ld_meta    = lambda wc: _get_theme_gwas_param(wc, "ld_meta_data"),
        region_list = lambda wc: _get_theme_gwas_param(wc, "region_list"),
    output:
        done = "{cwd}/colocalization/{theme}/enloc_enrichment/.done_enrich",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        outdir       = "{cwd}/colocalization/{theme}/enloc_enrichment",
        name         = "{theme}",
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["coloc"]["threads"]
    resources:
        mem_mb  = config["resources"]["coloc"]["mem_mb"],
        runtime = config["resources"]["coloc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/SuSiE_enloc.ipynb \\
            xqtl_gwas_enrichment {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --xqtl_meta_data {input.xqtl_meta} \\
            --gwas_meta_data {input.gwas_meta} \\
            --ld_meta_file_path {input.ld_meta} \\
            --region_list {input.region_list} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 2.2b — SuSiE-ENLOC colocalization
# ------------------------------------
# Runs Bayesian colocalization between xQTL and GWAS fine-mapped
# signals, using the enrichment priors derived in step 2.2a.
# Set --skip_enrich to bypass the prior enrichment step and use
# default priors (p12 = 5e-6).
rule susie_coloc:
    """Run SuSiE-ENLOC colocalization between xQTL and GWAS signals."""
    input:
        enrich_done = "{cwd}/colocalization/{theme}/enloc_enrichment/.done_enrich",
        xqtl_meta  = lambda wc: _get_theme_gwas_param(wc, "xqtl_meta_data"),
        gwas_meta  = lambda wc: _get_theme_gwas_param(wc, "gwas_meta_data"),
        ld_meta    = lambda wc: _get_theme_gwas_param(wc, "ld_meta_data"),
        region_list = lambda wc: _get_theme_gwas_param(wc, "region_list"),
    output:
        done = "{cwd}/colocalization/{theme}/susie_coloc/.done_coloc",
    params:
        pipeline_dir  = config["pipeline_dir"],
        container     = config["containers"]["susie"],
        enrich_dir    = "{cwd}/colocalization/{theme}/enloc_enrichment",
        outdir        = "{cwd}/colocalization/{theme}/susie_coloc",
        name          = "{theme}",
        dry_run       = DRY_RUN_SOS,
    threads: config["resources"]["coloc"]["threads"]
    resources:
        mem_mb  = config["resources"]["coloc"]["mem_mb"],
        runtime = config["resources"]["coloc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/SuSiE_enloc.ipynb \\
            susie_coloc {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --xqtl_meta_data {input.xqtl_meta} \\
            --gwas_meta_data {input.gwas_meta} \\
            --ld_meta_file_path {input.ld_meta} \\
            --region_list {input.region_list} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 2.3 — ColocBoost multi-trait colocalization
# ------------------------------------
# Joint fine-mapping across multiple molecular traits (and
# optionally GWAS) using individual-level genotype data.
# Runs in --xqtl-coloc mode by default; add --joint-gwas or
# --separate-gwas when gwas_meta_data is provided.
#
# Requires per-theme individual-level inputs:
#   geno_file  — genotype_by_chrom_files.txt
#   pheno_file — {theme}.phenotype_by_chrom_files.txt
#   cov_file   — hidden-factor covariate file
rule colocboost:
    """Run ColocBoost multi-trait colocalization using individual-level data."""
    input:
        tsv_done    = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
        geno_file   = lambda wc: _get_theme_param(wc, "geno_file"),
        pheno_file  = lambda wc: _get_theme_param(wc, "pheno_file"),
        cov_file    = lambda wc: _get_theme_param(wc, "cov_file"),
        region_list = lambda wc: _get_theme_gwas_param(wc, "region_list"),
    output:
        done = "{cwd}/colocalization/{theme}/colocboost/.done_colocboost",
    params:
        pipeline_dir  = config["pipeline_dir"],
        container     = config["containers"]["susie"],
        outdir        = "{cwd}/colocalization/{theme}/colocboost",
        name          = "{theme}",
        maf           = config["gwas"]["colocboost"]["maf"],
        mac           = config["gwas"]["colocboost"]["mac"],
        # Optionally pass GWAS meta if provided; empty string = xQTL-only
        gwas_meta_arg = lambda wc: (
            f"--gwas_meta_data {_get_theme_gwas_param(wc, 'gwas_meta_data')}"
            if config["gwas"].get("enabled", False) else ""
        ),
        ld_meta_arg   = lambda wc: (
            f"--ld_meta_data {_get_theme_gwas_param(wc, 'ld_meta_data')}"
            if config["gwas"].get("enabled", False) else ""
        ),
        dry_run       = DRY_RUN_SOS,
    threads: config["resources"]["colocboost"]["threads"]
    resources:
        mem_mb  = config["resources"]["colocboost"]["mem_mb"],
        runtime = config["resources"]["colocboost"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/colocboost.ipynb \\
            colocboost {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --genoFile {input.geno_file} \\
            --phenoFile {input.pheno_file} \\
            --covFile {input.cov_file} \\
            --region_list {input.region_list} \\
            --maf {params.maf} \\
            --mac {params.mac} \\
            --xqtl_coloc True \\
            {params.gwas_meta_arg} \\
            {params.ld_meta_arg} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """
