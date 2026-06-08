# ============================================================
# Rule Module 01: Molecular Phenotype Quantification  (Modular SoS)
# ============================================================
# Covers: FastQC → STAR alignment + RNA-SeQC quantification →
#         Multi-sample QC → TMM normalization
#
# SoS notebooks called (Modular SoS wrappers in renovated_code/notebook/modular_sos/):
#   - RNA_calling.ipynb          (fastqc, rnaseqc_call)
#   - bulk_expression_QC.ipynb   (qc)
#   - bulk_expression_normalization.ipynb (normalize)
#
# Each notebook step now calls the modular script from renovated_code/
# instead of running inline R/Python code.
# ============================================================

# ------------------------------------
# Step 1.1 — FastQC: raw read quality control
# ------------------------------------
rule fastqc:
    """Run FastQC on raw FASTQ files to assess read quality and detect adapters."""
    input:
        sample_list = lambda wc: next(
            t["fastq_list"] for t in config["themes"] if t["name"] == wc.theme
        ),
    output:
        done = "{cwd}/{theme}/molecular_phenotypes/fastqc/.done_fastqc",
    params:
        sos_bin          = SOS_BIN,
        sos_sched        = sos_sched("fastqc"),
        notebooks_dir    = NOTEBOOKS,
        renovated_dir    = RENOVATED,
        data_dir         = lambda wc: next(
            t["data_dir"] for t in config["themes"] if t["name"] == wc.theme
        ),
        outdir           = "{cwd}/{theme}/molecular_phenotypes/fastqc",
        dry_run          = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        runtime  = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        {params.sos_bin} run {params.notebooks_dir}/RNA_calling.ipynb fastqc \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        touch {output.done}
        """

# ------------------------------------
# Step 1.2 — STAR alignment + RNA-SeQC quantification
# ------------------------------------
rule rnaseqc_call:
    """Align RNA-seq reads with STAR and quantify with RNA-SeQC."""
    input:
        sample_list  = lambda wc: next(
            t["fastq_list"] for t in config["themes"] if t["name"] == wc.theme
        ),
        bam_list     = lambda wc: next(
            t["bam_list"] for t in config["themes"] if t["name"] == wc.theme
        ),
        fastqc_done  = "{cwd}/{theme}/molecular_phenotypes/fastqc/.done_fastqc",
    output:
        tpm_gct    = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.gene_tpm.gct.gz",
        counts_gct = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.gene_readsCount.gct.gz",
        exon_gct   = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.exon_readsCount.gct.gz",
        metrics    = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.metrics.tsv",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("rnaseqc_call"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        data_dir      = lambda wc: next(
            t["data_dir"] for t in config["themes"] if t["name"] == wc.theme
        ),
        gtf           = config["reference"]["gtf_collapsed"],
        fasta         = config["reference"]["fasta"],
        outdir        = "{cwd}/{theme}/molecular_phenotypes",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        runtime  = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        {params.sos_bin} run {params.notebooks_dir}/RNA_calling.ipynb rnaseqc_call \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --gtf {params.gtf} \
            --reference-fasta {params.fasta} \
            --bam-list {input.bam_list} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """

# ------------------------------------
# Step 1.3 — Multi-sample QC
# ------------------------------------
rule bulk_expression_qc:
    """Filter low-expression genes and remove outlier samples."""
    input:
        tpm_gct    = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.gene_tpm.gct.gz",
        counts_gct = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.gene_readsCount.gct.gz",
    output:
        tpm_filt   = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.low_expression_filtered.outlier_removed.tpm.gct.gz",
        count_filt = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.low_expression_filtered.outlier_removed.geneCount.gct.gz",
    params:
        sos_bin             = SOS_BIN,
        sos_sched           = sos_sched("bulk_expression_qc"),
        notebooks_dir        = NOTEBOOKS,
        renovated_dir        = RENOVATED,
        outdir               = "{cwd}/{theme}/molecular_phenotypes",
        low_expr_tpm         = config["rnaseq_qc"]["low_expr_tpm"],
        low_expr_tpm_percent = config["rnaseq_qc"]["low_expr_tpm_percent"],
        rle_filter_percent   = config["rnaseq_qc"]["rle_filter_percent"],
        ds_filter_percent    = config["rnaseq_qc"]["ds_filter_percent"],
        dry_run              = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/bulk_expression_QC.ipynb qc \
            --cwd {params.outdir} \
            --tpm-gct {input.tpm_gct} \
            --counts-gct {input.counts_gct} \
            --low-expr-TPM {params.low_expr_tpm} \
            --low-expr-TPM-percent {params.low_expr_tpm_percent} \
            --RLEFilterPercent {params.rle_filter_percent} \
            --DSFilterPercent {params.ds_filter_percent} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """

# ------------------------------------
# Step 1.4 — Normalization
# ------------------------------------
rule bulk_expression_normalization:
    """Normalize expression with TMM-CPM-voom and output BED.gz for QTL analysis."""
    input:
        tpm_filt   = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.low_expression_filtered.outlier_removed.tpm.gct.gz",
        count_filt = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.low_expression_filtered.outlier_removed.geneCount.gct.gz",
    output:
        phenotype_bed = "{cwd}/{theme}/molecular_phenotypes/{rnaseqc_prefix}.low_expression_filtered.outlier_removed.{norm_method}.expression.bed.gz",
    params:
        sos_bin               = SOS_BIN,
        sos_sched             = sos_sched("bulk_expression_normalization"),
        notebooks_dir         = NOTEBOOKS,
        renovated_dir         = RENOVATED,
        outdir                = "{cwd}/{theme}/molecular_phenotypes",
        gtf                   = config["reference"]["gtf_ercc"],
        norm_method           = config["normalization"]["method"],
        count_threshold       = config["normalization"]["count_threshold"],
        sample_frac_threshold = config["normalization"]["sample_frac_threshold"],
        sample_lookup         = lambda wc: next(
            t.get("sample_participant_lookup", "")
            for t in config["themes"] if t["name"] == wc.theme
        ),
        dry_run               = DRY_RUN_SOS,
    wildcard_constraints:
        norm_method = "[a-z_]+",
    threads: 1
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        runtime  = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/bulk_expression_normalization.ipynb normalize \
            --cwd {params.outdir} \
            --counts-gct {input.count_filt} \
            --tpm-gct {input.tpm_filt} \
            --annotation-gtf {params.gtf} \
            --sample_participant_lookup {params.sample_lookup} \
            --count-threshold {params.count_threshold} \
            --sample-frac-threshold {params.sample_frac_threshold} \
            --normalization-method {params.norm_method} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """
