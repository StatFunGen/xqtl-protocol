# ============================================================
# Rule Module 05: QTL Association Testing (TensorQTL)
# ============================================================
# Covers: CIS-QTL association testing with TensorQTL
#
# Mirrors: eQTL_analysis_commands.ipynb stage: TensorQTL
#
# SoS notebooks called:
#   - pipeline/TensorQTL.ipynb (cis)
#
# Note: get_hidden_factors() and get_phenotype_bed() are defined in the main
# Snakefile and are available to all included rule modules.
#
# TensorQTL `cis` workflow runs two internal sub-steps automatically:
#   cis_1: per-chromosome nominal scan + permutation → parquet + TSV files
#   cis_2: aggregate per-chromosome results, compute q-values → significance TSV
# There is NO separate cis_postprocess step to call.
# ============================================================

# ------------------------------------
# Step 5.1 — TensorQTL CIS association scan
# ------------------------------------
# Inputs:
#   --genotype-file: path to genotype_by_chrom_files.txt (list of per-chrom plinks)
#   --phenotype-file: path to {theme}.phenotype_by_chrom_files.txt (list of per-chrom BEDs)
#   --covariate-file: hidden factor file (Marchenko PCA or PEER)
#
# Output: uses a sentinel done file because TensorQTL produces many files
# whose names depend on the input file basenames. The done file is written
# only after cis_2 completes (both nominal + significance aggregation).
rule tensorqtl_cis:
    """Run TensorQTL cis-QTL nominal + permutation scan across all chromosomes."""
    input:
        geno_list      = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.genotype_by_chrom_files.txt",
        pheno_list     = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.phenotype_by_chrom_files.txt",
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        done = "{cwd}/association_scan/{theme}/TensorQTL/.done_cis",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["tensorqtl"],
        outdir       = "{cwd}/association_scan/{theme}/TensorQTL",
        cis_window   = config["association"]["cis_window"],
        mac          = config["association"]["mac_threshold"],
        maf          = config["association"]["maf_threshold"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["tensorqtl"]["threads"]
    resources:
        mem_mb   = config["resources"]["tensorqtl"]["mem_mb"],
        runtime = config["resources"]["tensorqtl"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/TensorQTL.ipynb cis \
            --cwd {params.outdir} \
            --genotype-file {input.geno_list} \
            --phenotype-file {input.pheno_list} \
            --covariate-file {input.hidden_factors} \
            --window {params.cis_window} \
            --MAC {params.mac} \
            --maf-threshold {params.maf} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        touch {output.done}
        """
