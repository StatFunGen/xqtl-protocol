# ============================================================
# Rule Module 05: QTL Association Testing  (Route 3)
# ============================================================
# Covers: CIS-QTL association testing with TensorQTL
#
# SoS notebook called (Route 3 wrapper in route3/notebooks/):
#   - TensorQTL.ipynb (cis)
#
# The notebook task block calls:
#   renovated_code/association_scan/TensorQTL/TensorQTL.sh cis
# ============================================================

# ------------------------------------
# Step 5.1 — TensorQTL CIS association scan
# ------------------------------------
rule tensorqtl_cis:
    """Run TensorQTL cis-QTL nominal + permutation scan across all chromosomes."""
    input:
        geno_list      = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.genotype_by_chrom_files.txt",
        pheno_list     = "{cwd}/data_preprocessing/{theme}/phenotype_data/{theme}.phenotype_by_chrom_files.txt",
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        done = "{cwd}/association_scan/{theme}/TensorQTL/.done_cis",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        container     = config["containers"]["tensorqtl"],
        outdir        = "{cwd}/association_scan/{theme}/TensorQTL",
        cis_window    = config["association"]["cis_window"],
        mac           = config["association"]["mac_threshold"],
        maf           = config["association"]["maf_threshold"],
        dry_run       = DRY_RUN_SOS,
    threads: config["resources"]["tensorqtl"]["threads"]
    resources:
        mem_mb   = config["resources"]["tensorqtl"]["mem_mb"],
        runtime  = config["resources"]["tensorqtl"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.notebooks_dir}/TensorQTL.ipynb cis \
            --cwd {params.outdir} \
            --genotype-file {input.geno_list} \
            --phenotype-file {input.pheno_list} \
            --covariate-file {input.hidden_factors} \
            --window {params.cis_window} \
            --MAC {params.mac} \
            --maf-threshold {params.maf} \
            --container {params.container} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
        touch {output.done}
        """
