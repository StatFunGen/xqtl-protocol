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
#   - pipeline/GWAS_QC.ipynb (genotype_phenotype_sample_overlap, king, qc, qc_no_prune)
#   - pipeline/PCA.ipynb (flashpca, project_samples)
#
# Dependency chain:
#   sample_match → king_kinship → unrelated_qc ──→ flashpca
#                              └→ related_qc  ──→ project_samples
# ============================================================

# ------------------------------------
# Step 3.0 — Sample matching: extract phenotype sample IDs from genotype
# ------------------------------------
# Identifies the overlap between genotype samples (.fam file) and
# phenotype samples (BED.gz header). Produces sample_genotypes.txt,
# which is the keep-samples list passed to KING.
# SoS step: genotype_phenotype_sample_overlap
rule sample_match:
    """Identify genotype-phenotype sample overlap and write a keep-samples list."""
    input:
        bed           = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        sample_genotypes = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.sample_genotypes.txt",
        sample_overlap   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.sample_overlap.txt",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/GWAS_QC.ipynb genotype_phenotype_sample_overlap \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --phenoFile {input.phenotype_bed} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 3.1 — KING kinship: identify related vs. unrelated samples
# ------------------------------------
# Restricts the genotype file to the phenotype-matched samples (via
# --keep-samples), then runs KING to split samples into related and
# unrelated sets. Output naming: {genoFile_basename}.{name}.{un}related.bed.
rule king_kinship:
    """Run KING kinship analysis and split into related/unrelated sample sets."""
    input:
        bed              = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        sample_genotypes = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.sample_genotypes.txt",
    output:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.bed",
        related_bed   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.bed",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["high_mem"]["mem_mb"],
        runtime = config["resources"]["high_mem"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/GWAS_QC.ipynb king \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --keep-samples {input.sample_genotypes} \
            --name xqtl_protocol_data.plink_qc.{wildcards.theme} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 3.2 — QC on unrelated samples: variant filters + LD pruning
# ------------------------------------
# Applies MAC filter and LD pruning to the unrelated sample set to
# produce a clean variant set for PCA.
# SoS step: qc  (runs qc_1 + qc_2, the latter performing LD pruning)
rule unrelated_qc:
    """Apply QC filters and LD pruning to the unrelated sample set."""
    input:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.bed",
    output:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.bed",
        pruned_in  = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.in",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter   = config["genotype_qc"]["mac_filter"],
        ld_window    = config["genotype_qc"]["ld_window"],
        ld_shift     = config["genotype_qc"]["ld_shift"],
        ld_r2        = config["genotype_qc"]["ld_r2"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/GWAS_QC.ipynb qc \
            --cwd {params.outdir} \
            --genoFile {input.unrelated_bed} \
            --mac-filter {params.mac_filter} \
            --window {params.ld_window} \
            --shift {params.ld_shift} \
            --r2 {params.ld_r2} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 3.3 — QC on related samples using unrelated-derived variant set
# ------------------------------------
# Applies the LD-pruned variant list from unrelated samples to the
# related sample set, ensuring variant consistency for PCA projection.
# Uses --keep-variants (not --extract) — verified from GWAS_QC.ipynb.
rule related_qc:
    """Filter related samples using the pruned variant list from unrelated QC."""
    input:
        related_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.bed",
        pruned_in   = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.in",
    output:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.plink_qc.extracted.bed",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter   = config["genotype_qc"]["mac_filter"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/GWAS_QC.ipynb qc_no_prune \
            --cwd {params.outdir} \
            --genoFile {input.related_bed} \
            --keep-variants {input.pruned_in} \
            --mac-filter {params.mac_filter} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 3.4 — FlashPCA on unrelated LD-pruned samples
# ------------------------------------
# Computes principal components from the LD-pruned unrelated genotypes.
# Output: {genoFile_basename}.pca.rds (the PCA model used for projection).
rule flashpca:
    """Compute genotype PCs using flashPCA on the unrelated, LD-pruned sample set."""
    input:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.bed",
    output:
        pca_rds = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.pca.rds",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["flashpca"],
        outdir       = "{cwd}/data_preprocessing/{theme}/pca",
        n_pcs        = config["pca"]["n_pcs"],
        maha_k       = config["pca"]["maha_k"],
        maha_prob    = config["pca"]["maha_prob"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["pca"]["threads"]
    resources:
        mem_mb   = config["resources"]["pca"]["mem_mb"],
        runtime = config["resources"]["pca"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/PCA.ipynb flashpca \
            --cwd {params.outdir} \
            --genoFile {input.pruned_bed} \
            --k {params.n_pcs} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 3.5 — Project related samples onto unrelated PCA space
# ------------------------------------
# Projects related individuals onto the PC space learned from unrelated
# samples using --pca-model (the RDS file from flashpca).
# Note: --pcaFile, --scree-file, --pve-threshold do NOT exist in PCA.ipynb.
# The pca_model path is passed explicitly via --pca-model.
rule project_samples:
    """Project related samples onto the PCA space of unrelated samples."""
    input:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/xqtl_protocol_data.plink_qc.{theme}.related.plink_qc.extracted.bed",
        pca_rds        = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.unrelated.plink_qc.prune.pca.rds",
    output:
        projected_rds = "{cwd}/data_preprocessing/{theme}/pca/xqtl_protocol_data.plink_qc.{theme}.pca.projected.rds",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["flashpca"],
        outdir       = "{cwd}/data_preprocessing/{theme}/pca",
        maha_k       = config["pca"]["maha_k"],
        maha_prob    = config["pca"]["maha_prob"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["pca"]["threads"]
    resources:
        mem_mb   = config["resources"]["pca"]["mem_mb"],
        runtime = config["resources"]["pca"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/PCA.ipynb project_samples \
            --cwd {params.outdir} \
            --genoFile {input.related_qc_bed} \
            --pca-model {input.pca_rds} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --container {params.container} \
            --numThreads {threads} {params.dry_run}
        """
