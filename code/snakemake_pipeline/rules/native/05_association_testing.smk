# ============================================================
# Rule Module 05: QTL Association Testing (native scripts)
# ============================================================
# Calls TensorQTL.sh (which runs TensorQTL.py) directly.
# Interface flags are identical to the SoS version.
# ============================================================

RENOVATED = config["renovated_code_dir"]

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
        container = config["containers"]["tensorqtl"],
        outdir    = "{cwd}/association_scan/{theme}/TensorQTL",
        cis_window = config["association"]["cis_window"],
        mac        = config["association"]["mac_threshold"],
        maf        = config["association"]["maf_threshold"],
        script     = f"{RENOVATED}/association_scan/TensorQTL/TensorQTL.sh",
    threads: config["resources"]["tensorqtl"]["threads"]
    resources:
        mem_mb  = config["resources"]["tensorqtl"]["mem_mb"],
        runtime = config["resources"]["tensorqtl"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} cis \
            --container {params.container} \
            --cwd {params.outdir} \
            --genotype-file {input.geno_list} \
            --phenotype-file {input.pheno_list} \
            --covariate-file {input.hidden_factors} \
            --window {params.cis_window} \
            --MAC {params.mac} \
            --maf-threshold {params.maf} \
            --numThreads {threads}
        touch {output.done}
        """
