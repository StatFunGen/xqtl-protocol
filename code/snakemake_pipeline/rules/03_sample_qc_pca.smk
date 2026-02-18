# ============================================================
# Rule Module 03: Per-tissue Sample QC, Kinship & PCA
# ============================================================
# Covers: Sample matching → KING kinship → Unrelated/related QC →
#         FlashPCA (unrelated) → Project related samples onto PC space
#
# Mirrors: eQTL_analysis_commands.ipynb stages:
#   sample_match → king → unrelated_QC → related_QC → pca → projected_sample
#
# SoS notebooks called:
#   - pipeline/GWAS_QC.ipynb (king, qc, qc_no_prune)
#   - pipeline/PCA.ipynb (flashpca, project_samples)
# ============================================================

# ------------------------------------
# Step 3.1 — KING kinship analysis: identify related vs. unrelated samples
# ------------------------------------
# Restricts the genotype file to samples present in the phenotype data,
# then runs KING to classify samples as related or unrelated.
rule king_kinship:
    """Run KING kinship analysis and split into related/unrelated sample sets."""
    input:
        bed          = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.bed",
        related_bed   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.bed",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        kinship      = config["genotype_qc"]["kinship"],
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["high_mem"]["mem_mb"],
        walltime = config["resources"]["high_mem"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/GWAS_QC.ipynb king \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --phenoFile {input.phenotype_bed} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --kinship {params.kinship} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.2 — QC on unrelated samples: variant filters + LD pruning
# ------------------------------------
# Applies MAF/MAC filters and LD pruning to produce a clean
# variant set for PCA.
rule unrelated_qc:
    """Apply QC filters and LD pruning to the unrelated sample set."""
    input:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.bed",
    output:
        pruned_bed  = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.bed",
        pruned_in   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.in",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter   = config["genotype_qc"]["mac_filter"],
        ld_window    = config["genotype_qc"]["ld_window"],
        ld_shift     = config["genotype_qc"]["ld_shift"],
        ld_r2        = config["genotype_qc"]["ld_r2"],
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        walltime = config["resources"]["genotype_qc"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/GWAS_QC.ipynb qc \
            --cwd {params.outdir} \
            --genoFile {input.unrelated_bed} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme}.unrelated.filtered \
            --mac-filter {params.mac_filter} \
            --window {params.ld_window} \
            --shift {params.ld_shift} \
            --r2 {params.ld_r2} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.3 — QC on related samples using unrelated-derived variant set
# ------------------------------------
# Extracts the pruned variant list from unrelated samples and applies
# it to related samples (no re-pruning, preserves variant consistency).
rule related_qc:
    """Filter related samples using the variant list derived from unrelated QC."""
    input:
        related_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.bed",
        pruned_in   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.in",
    output:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.filtered.extracted.bed",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter   = config["genotype_qc"]["mac_filter"],
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        walltime = config["resources"]["genotype_qc"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/GWAS_QC.ipynb qc_no_prune \
            --cwd {params.outdir} \
            --genoFile {input.related_bed} \
            --extract {input.pruned_in} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme}.related.filtered.extracted \
            --mac-filter {params.mac_filter} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.4 — FlashPCA on unrelated samples
# ------------------------------------
# Computes principal components from the LD-pruned unrelated genotypes.
rule flashpca:
    """Compute genotype PCs using flashPCA on the unrelated, LD-pruned sample set."""
    input:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.bed",
    output:
        pca_rds   = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.pca.rds",
        scree_txt = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.pca.scree.txt",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["flashpca"],
        outdir       = "{cwd}/data_preprocessing/{theme}/pca",
        n_pcs        = config["pca"]["n_pcs"],
        maha_k       = config["pca"]["maha_k"],
        maha_prob    = config["pca"]["maha_prob"],
    threads: config["resources"]["pca"]["threads"]
    resources:
        mem_mb   = config["resources"]["pca"]["mem_mb"],
        walltime = config["resources"]["pca"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/PCA.ipynb flashpca \
            --cwd {params.outdir} \
            --genoFile {input.pruned_bed} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme}.unrelated.filtered.prune \
            --k {params.n_pcs} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 3.5 — Project related samples onto unrelated PCA space
# ------------------------------------
# Projects related individuals onto the PC space defined by unrelated samples,
# then selects the number of PCs explaining >= pve_threshold of variance.
rule project_samples:
    """Project related samples onto the PCA space of unrelated samples."""
    input:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.filtered.extracted.bed",
        pca_rds        = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.pca.rds",
        scree_txt      = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.filtered.prune.pca.scree.txt",
    output:
        projected_rds   = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.pca.projected.rds",
        projected_scree = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.pca.projected.scree.txt",
    params:
        pipeline_dir  = config["pipeline_dir"],
        container     = config["containers"]["flashpca"],
        outdir        = "{cwd}/data_preprocessing/{theme}/pca",
        pve_threshold = config["pca"]["pve_threshold"],
        maha_k        = config["pca"]["maha_k"],
        maha_prob     = config["pca"]["maha_prob"],
    threads: config["resources"]["pca"]["threads"]
    resources:
        mem_mb   = config["resources"]["pca"]["mem_mb"],
        walltime = config["resources"]["pca"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/PCA.ipynb project_samples \
            --cwd {params.outdir} \
            --genoFile {input.related_qc_bed} \
            --pcaFile {input.pca_rds} \
            --scree-file {input.scree_txt} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --pve-threshold {params.pve_threshold} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --container {params.container} \
            --numThreads {threads}
        """
