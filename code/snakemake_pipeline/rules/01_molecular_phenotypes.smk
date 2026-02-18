# ============================================================
# Rule Module 01: Molecular Phenotype Quantification
# ============================================================
# Covers: FastQC → FastP trimming → STAR alignment →
#         RNA-SeQC quantification → Multi-sample QC → TMM normalization
#
# Mirrors: code/commands_generator/bulk_expression_commands.ipynb
# SoS notebooks called:
#   - pipeline/RNA_calling.ipynb (fastqc, fastp_trim_adaptor, STAR_align, rnaseqc_call)
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
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        walltime = config["resources"]["rna_calling"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/RNA_calling.ipynb fastqc \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --container {params.container} \
            --numThreads {threads}
        touch {output.done}
        """

# ------------------------------------
# Step 1.2 — STAR Alignment + RNA-SeQC quantification
# ------------------------------------
# Runs STAR alignment (with fastp adapter trimming) and RNA-SeQC gene quantification.
# Produces per-tissue TPM and read count matrices.
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
        star_index   = config["reference"]["star_index"],
        gtf          = config["reference"]["gtf_collapsed"],
        fasta        = config["reference"]["fasta"],
        adapters     = config["reference"]["adapters"],
        outdir       = "{cwd}/{theme}/molecular_phenotypes",
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        walltime = config["resources"]["rna_calling"]["walltime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/RNA_calling.ipynb rnaseqc_call \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --STAR-index {params.star_index} \
            --gtf {params.gtf} \
            --reference-fasta {params.fasta} \
            --fasta-with-adapters-etc {params.adapters} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 1.3 — Multi-sample QC: filter low-expression genes and outlier samples
# ------------------------------------
rule bulk_expression_qc:
    """
    Filter genes with low expression and remove outlier samples
    using RLE and D-statistic methods.
    """
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
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        walltime = config["resources"]["default"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/bulk_expression_QC.ipynb qc \
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
rule bulk_expression_normalization:
    """
    Normalize expression with TMM-CPM-voom and output BED.gz for QTL analysis.
    """
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
        tpm_threshold         = config["normalization"]["tpm_threshold"],
        count_threshold       = config["normalization"]["count_threshold"],
        sample_frac_threshold = config["normalization"]["sample_frac_threshold"],
        quantile_normalize    = lambda _: "--quantile-normalize" if config["normalization"]["quantile_normalize"] else "",
        # The sample-participant lookup maps RNA-seq sample IDs to genotype participant IDs
        sample_lookup         = lambda wc: next(
            t.get("sample_participant_lookup", "")
            for t in config["themes"] if t["name"] == wc.theme
        ),
    wildcard_constraints:
        norm_method = "[a-z_]+",
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb   = config["resources"]["rna_calling"]["mem_mb"],
        walltime = config["resources"]["rna_calling"]["walltime"],
    shell:
        """
        sos run {params.pipeline_dir}/bulk_expression_normalization.ipynb normalize \
            --cwd {params.outdir} \
            --counts-gct {input.count_filt} \
            --tpm-gct {input.tpm_filt} \
            --annotation-gtf {params.gtf} \
            --sample-participant-lookup {params.sample_lookup} \
            --tpm-threshold {params.tpm_threshold} \
            --count-threshold {params.count_threshold} \
            --sample-frac-threshold {params.sample_frac_threshold} \
            --normalization-method {params.norm_method} \
            {params.quantile_normalize} \
            --container {params.container} \
            --numThreads {threads}
        """
