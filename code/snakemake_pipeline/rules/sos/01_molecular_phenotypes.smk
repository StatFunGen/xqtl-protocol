# ============================================================
# Rule Module 01: Molecular Phenotype Quantification
# ============================================================
# Covers: FastQC → STAR alignment + RNA-SeQC quantification →
#         Multi-sample QC → TMM normalization
#
# Mirrors: code/commands_generator/bulk_expression_commands.ipynb
# SoS notebooks called:
#   - pipeline/RNA_calling.ipynb (fastqc, rnaseqc_call)
#   - pipeline/bulk_expression_QC.ipynb (qc)
#   - pipeline/bulk_expression_normalization.ipynb (normalize)
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
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["rnaquant"],
        data_dir     = lambda wc: next(
            t["data_dir"] for t in config["themes"] if t["name"] == wc.theme
        ),
        outdir       = "{cwd}/{theme}/molecular_phenotypes/fastqc",
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        runtime = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.dry_run} {params.pipeline_dir}/RNA_calling.ipynb fastqc \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.done}
        """

# ------------------------------------
# Step 1.2 — STAR alignment + RNA-SeQC quantification
# ------------------------------------
# The rnaseqc_call step handles STAR alignment internally via fastp trimming.
# It takes --sample-list (FASTQs), --data-dir, --gtf, and --reference-fasta.
# Note: --STAR-index is NOT a parameter of rnaseqc_call; STAR index path is
# embedded in the reference_fasta parameter handling within the SoS notebook.
rule rnaseqc_call:
    """Align RNA-seq reads with STAR and quantify with RNA-SeQC."""
    input:
        sample_list  = lambda wc: next(
            t["fastq_list"] for t in config["themes"] if t["name"] == wc.theme
        ),
        fastqc_done  = "{cwd}/{theme}/molecular_phenotypes/fastqc/.done_fastqc",
    output:
        tpm_gct    = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_tpm.gct.gz",
        counts_gct = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_reads.gct.gz",
        exon_gct   = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.exon_reads.gct.gz",
        metrics    = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.metrics.tsv",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["rnaquant"],
        data_dir     = lambda wc: next(
            t["data_dir"] for t in config["themes"] if t["name"] == wc.theme
        ),
        gtf          = config["reference"]["gtf_collapsed"],
        fasta        = config["reference"]["fasta"],
        outdir       = "{cwd}/{theme}/molecular_phenotypes",
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        runtime = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.dry_run} {params.pipeline_dir}/RNA_calling.ipynb rnaseqc_call \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --gtf {params.gtf} \
            --reference-fasta {params.fasta} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 1.3 — Multi-sample QC
# ------------------------------------
# Filters low-expression genes and removes outlier samples using RLE
# and D-statistic methods.
rule bulk_expression_qc:
    """Filter low-expression genes and remove outlier samples."""
    input:
        tpm_gct    = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_tpm.gct.gz",
        counts_gct = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_reads.gct.gz",
    output:
        tpm_filt   = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.tpm.gct.gz",
        count_filt = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.geneCount.gct.gz",
    params:
        pipeline_dir         = config["pipeline_dir"],
        container            = config["containers"]["rnaquant"],
        outdir               = "{cwd}/{theme}/molecular_phenotypes",
        low_expr_tpm         = config["rnaseq_qc"]["low_expr_tpm"],
        low_expr_tpm_percent = config["rnaseq_qc"]["low_expr_tpm_percent"],
        rle_filter_percent   = config["rnaseq_qc"]["rle_filter_percent"],
        ds_filter_percent    = config["rnaseq_qc"]["ds_filter_percent"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        sos run {params.dry_run} {params.pipeline_dir}/bulk_expression_QC.ipynb qc \
            --cwd {params.outdir} \
            --tpm-gct {input.tpm_gct} \
            --counts-gct {input.counts_gct} \
            --low-expr-TPM {params.low_expr_tpm} \
            --low-expr-TPM-percent {params.low_expr_tpm_percent} \
            --RLEFilterPercent {params.rle_filter_percent} \
            --DSFilterPercent {params.ds_filter_percent} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 1.4 — Normalization: TMM-CPM + quantile normalization
# ------------------------------------
# Produces the final phenotype BED file consumed by the QTL pipeline.
# Note: --sample_participant_lookup uses an underscore (not hyphen) — SoS
# preserves the underscore for this parameter name.
rule bulk_expression_normalization:
    """Normalize expression with TMM-CPM-voom and output BED.gz for QTL analysis."""
    input:
        tpm_filt   = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.tpm.gct.gz",
        count_filt = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.geneCount.gct.gz",
    output:
        phenotype_bed = "{cwd}/{theme}/molecular_phenotypes/{theme}.{norm_method}.expression.bed.gz",
    params:
        pipeline_dir          = config["pipeline_dir"],
        container             = config["containers"]["rnaquant"],
        outdir                = "{cwd}/{theme}/molecular_phenotypes",
        gtf                   = config["reference"]["gtf_ercc"],
        norm_method           = config["normalization"]["method"],
        count_threshold       = config["normalization"]["count_threshold"],
        sample_frac_threshold = config["normalization"]["sample_frac_threshold"],
        # sample_participant_lookup keeps underscores (verified from actual notebook)
        sample_lookup         = lambda wc: next(
            t.get("sample_participant_lookup", "")
            for t in config["themes"] if t["name"] == wc.theme
        ),
        dry_run     = DRY_RUN_SOS,
    wildcard_constraints:
        norm_method = "[a-z_]+",
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        runtime = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        sos run {params.dry_run} {params.pipeline_dir}/bulk_expression_normalization.ipynb normalize \
            --cwd {params.outdir} \
            --counts-gct {input.count_filt} \
            --tpm-gct {input.tpm_filt} \
            --annotation-gtf {params.gtf} \
            --sample_participant_lookup {params.sample_lookup} \
            --count-threshold {params.count_threshold} \
            --sample-frac-threshold {params.sample_frac_threshold} \
            --normalization-method {params.norm_method} \
            --container {params.container} \
            --numThreads {threads}
        """
