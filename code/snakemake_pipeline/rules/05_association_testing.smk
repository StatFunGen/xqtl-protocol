# ============================================================
# Rule Module 05: QTL Association Testing (TensorQTL)
# ============================================================
# Covers: CIS-QTL association testing with TensorQTL
#
# Mirrors: eQTL_analysis_commands.ipynb stage: TensorQTL
#
# SoS notebooks called:
#   - pipeline/TensorQTL.ipynb (cis)
# ============================================================

# Note: get_hidden_factors() and get_phenotype_bed() are defined in the main Snakefile
# and are available to all included rule modules.

# ------------------------------------
# Step 5.1 — TensorQTL CIS: genome-wide cis-QTL nominal scan
# ------------------------------------
# Runs TensorQTL for CIS-QTL analysis in two sub-steps:
#   cis_1: per-chromosome nominal and permutation passes → parquet/TSV files
#   cis_2: aggregation of per-chromosome results → significance table
#
# Inputs:
#   - Per-chromosome plink genotype files (list)
#   - Per-chromosome phenotype BED files (recipe/list)
#   - Hidden factor covariate file (Marchenko PCA or PEER)
# Outputs:
#   - CIS-QTL results recipe TSV pointing to all per-chromosome results
rule tensorqtl_cis:
    """Run TensorQTL cis-QTL nominal + permutation scan across all chromosomes."""
    input:
        geno_list     = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.plink_files_list.txt",
        pheno_recipe  = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.per_chrom.recipe",
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        cis_recipe = "{cwd}/association_scan/{theme}/TensorQTL/TensorQTL.cis._recipe.tsv",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["tensorqtl"],
        outdir       = "{cwd}/association_scan/{theme}/TensorQTL",
        cis_window   = config["association"]["cis_window"],
        mac          = config["association"]["mac_threshold"],
        maf          = config["association"]["maf_threshold"],
        pvalue_cut   = config["association"]["pvalue_cutoff"],
    threads: config["resources"]["tensorqtl"]["threads"]
    resources:
        mem_mb   = config["resources"]["tensorqtl"]["mem_mb"],
        walltime = config["resources"]["tensorqtl"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        # Step cis_1: per-chromosome nominal scan + permutation
        sos run {params.pipeline_dir}/TensorQTL.ipynb cis \
            --cwd {params.outdir} \
            --genotype-file {input.geno_list} \
            --phenotype-file {input.pheno_recipe} \
            --covariate-file {input.hidden_factors} \
            --window {params.cis_window} \
            --MAC {params.mac} \
            --maf-threshold {params.maf} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 5.2 — Post-process CIS results: p-value adjustment
# ------------------------------------
# Aggregates per-chromosome results, applies hierarchical multiple
# testing correction (BH/Bonferroni on permutation-based p-values),
# and generates a genome-wide significance summary.
rule tensorqtl_cis_postprocess:
    """Aggregate CIS-QTL results and apply multiple-testing correction."""
    input:
        cis_recipe = "{cwd}/association_scan/{theme}/TensorQTL/TensorQTL.cis._recipe.tsv",
    output:
        sig_table = "{cwd}/association_scan/{theme}/TensorQTL/TensorQTL.cis.regional_significance.tsv.gz",
        summary   = "{cwd}/association_scan/{theme}/TensorQTL/TensorQTL.cis.regional_significance.summary.txt",
    params:
        pipeline_dir  = config["pipeline_dir"],
        container     = config["containers"]["tensorqtl"],
        outdir        = "{cwd}/association_scan/{theme}/TensorQTL",
        pvalue_cutoff = config["association"]["pvalue_cutoff"],
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/TensorQTL.ipynb cis_postprocess \
            --cwd {params.outdir} \
            --cis-recipe {input.cis_recipe} \
            --pvalue-cutoff {params.pvalue_cutoff} \
            --container {params.container} \
            --numThreads {threads}
        """
