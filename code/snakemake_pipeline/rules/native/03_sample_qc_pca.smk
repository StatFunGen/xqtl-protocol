# ============================================================
# Rule Module 03: Per-tissue Sample QC, Kinship & PCA (native scripts)
# ============================================================
# Calls renovated_code/ shell wrappers instead of SoS notebooks.
# Interface flags are identical to the SoS versions.
# ============================================================

RENOVATED = config["renovated_code_dir"]

# ------------------------------------
# Step 3.0 — Sample matching
# ------------------------------------
rule sample_match:
    """Identify genotype-phenotype sample overlap and write a keep-samples list."""
    input:
        bed           = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        sample_genotypes = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.sample_genotypes.txt",
        sample_overlap   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.sample_overlap.txt",
    params:
        container = config["containers"]["bioinfo"],
        outdir    = "{cwd}/data_preprocessing/{theme}/genotype_data",
        script    = f"{RENOVATED}/data_preprocessing/genotype/GWAS_QC.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} genotype_phenotype_sample_overlap {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --phenoFile {input.phenotype_bed} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.1 — KING kinship
# ------------------------------------
rule king_kinship:
    """Run KING kinship analysis and split into related/unrelated sample sets."""
    input:
        bed              = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        sample_genotypes = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.sample_genotypes.txt",
    output:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.bed",
        related_bed   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.bed",
    params:
        container = config["containers"]["bioinfo"],
        outdir    = "{cwd}/data_preprocessing/{theme}/genotype_data",
        script    = f"{RENOVATED}/data_preprocessing/genotype/GWAS_QC.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb  = config["resources"]["high_mem"]["mem_mb"],
        runtime = config["resources"]["high_mem"]["runtime"],
    shell:
        """
        bash {params.script} king {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --keep-samples {input.sample_genotypes} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.2 — QC on unrelated samples
# ------------------------------------
rule unrelated_qc:
    """Apply QC filters and LD pruning to the unrelated sample set."""
    input:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.bed",
    output:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.bed",
        pruned_in  = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.in",
    params:
        container  = config["containers"]["bioinfo"],
        outdir     = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter = config["genotype_qc"]["mac_filter"],
        ld_window  = config["genotype_qc"]["ld_window"],
        ld_shift   = config["genotype_qc"]["ld_shift"],
        ld_r2      = config["genotype_qc"]["ld_r2"],
        script     = f"{RENOVATED}/data_preprocessing/genotype/GWAS_QC.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb  = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        bash {params.script} qc {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.unrelated_bed} \
            --mac-filter {params.mac_filter} \
            --window {params.ld_window} \
            --shift {params.ld_shift} \
            --r2 {params.ld_r2} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.3 — QC on related samples
# ------------------------------------
rule related_qc:
    """Filter related samples using the pruned variant list from unrelated QC."""
    input:
        related_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.bed",
        pruned_in   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.in",
    output:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.plink_qc.extracted.bed",
    params:
        container  = config["containers"]["bioinfo"],
        outdir     = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter = config["genotype_qc"]["mac_filter"],
        script     = f"{RENOVATED}/data_preprocessing/genotype/GWAS_QC.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb  = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        bash {params.script} qc_no_prune {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.related_bed} \
            --keep-variants {input.pruned_in} \
            --mac-filter {params.mac_filter} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.4 — FlashPCA on unrelated LD-pruned samples
# ------------------------------------
rule flashpca:
    """Compute genotype PCs using flashPCA on the unrelated, LD-pruned sample set."""
    input:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.bed",
    output:
        pca_rds = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.pca.rds",
    params:
        container = config["containers"]["flashpca"],
        outdir    = "{cwd}/data_preprocessing/{theme}/pca",
        n_pcs     = config["pca"]["n_pcs"],
        maha_k    = config["pca"]["maha_k"],
        maha_prob = config["pca"]["maha_prob"],
        script    = f"{RENOVATED}/data_preprocessing/genotype/PCA.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["pca"]["threads"]
    resources:
        mem_mb  = config["resources"]["pca"]["mem_mb"],
        runtime = config["resources"]["pca"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} flashpca {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.pruned_bed} \
            --k {params.n_pcs} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.5 — Project related samples onto unrelated PCA space
# ------------------------------------
rule project_samples:
    """Project related samples onto the PCA space of unrelated samples."""
    input:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.plink_qc.extracted.bed",
        pca_rds        = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.pca.rds",
    output:
        projected_rds = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.pca.projected.rds",
    params:
        container = config["containers"]["flashpca"],
        outdir    = "{cwd}/data_preprocessing/{theme}/pca",
        maha_k    = config["pca"]["maha_k"],
        maha_prob = config["pca"]["maha_prob"],
        script    = f"{RENOVATED}/data_preprocessing/genotype/PCA.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["pca"]["threads"]
    resources:
        mem_mb  = config["resources"]["pca"]["mem_mb"],
        runtime = config["resources"]["pca"]["runtime"],
    shell:
        """
        bash {params.script} project_samples {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.related_qc_bed} \
            --pca-model {input.pca_rds} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --numThreads {threads}
        """
