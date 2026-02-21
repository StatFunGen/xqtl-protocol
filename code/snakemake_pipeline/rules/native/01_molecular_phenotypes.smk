# ============================================================
# Rule Module 01: Molecular Phenotype Quantification (native scripts)
# ============================================================
# Calls renovated_code/ shell wrappers instead of SoS notebooks.
# Interface flags are identical to the SoS versions.
# ============================================================

RENOVATED = config["renovated_code_dir"]

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
        container = config["containers"]["rnaquant"],
        data_dir  = lambda wc: next(
            t["data_dir"] for t in config["themes"] if t["name"] == wc.theme
        ),
        outdir    = "{cwd}/{theme}/molecular_phenotypes/fastqc",
        script    = f"{RENOVATED}/molecular_phenotypes/calling/RNA_calling.sh",
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb  = config["resources"]["rna_calling"]["mem_mb"],
        runtime = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} fastqc \
            --container {params.container} \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --numThreads {threads}
        touch {output.done}
        """

# ------------------------------------
# Step 1.2 — STAR alignment + RNA-SeQC quantification
# ------------------------------------
rule rnaseqc_call:
    """Align RNA-seq reads with STAR and quantify with RNA-SeQC."""
    input:
        sample_list = lambda wc: next(
            t["fastq_list"] for t in config["themes"] if t["name"] == wc.theme
        ),
        fastqc_done = "{cwd}/{theme}/molecular_phenotypes/fastqc/.done_fastqc",
    output:
        tpm_gct    = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_tpm.gct.gz",
        counts_gct = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_reads.gct.gz",
        exon_gct   = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.exon_reads.gct.gz",
        metrics    = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.metrics.tsv",
    params:
        container = config["containers"]["rnaquant"],
        data_dir  = lambda wc: next(
            t["data_dir"] for t in config["themes"] if t["name"] == wc.theme
        ),
        gtf    = config["reference"]["gtf_collapsed"],
        fasta  = config["reference"]["fasta"],
        outdir = "{cwd}/{theme}/molecular_phenotypes",
        script = f"{RENOVATED}/molecular_phenotypes/calling/RNA_calling.sh",
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb  = config["resources"]["rna_calling"]["mem_mb"],
        runtime = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} rnaseqc_call \
            --container {params.container} \
            --cwd {params.outdir} \
            --sample-list {input.sample_list} \
            --data-dir {params.data_dir} \
            --gtf {params.gtf} \
            --reference-fasta {params.fasta} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 1.3 — Multi-sample QC
# ------------------------------------
# NOTE: bulk_expression_QC.sh is not yet implemented in renovated_code/.
# This rule will fail until the script is added.
rule bulk_expression_qc:
    """Filter low-expression genes and remove outlier samples."""
    input:
        tpm_gct    = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_tpm.gct.gz",
        counts_gct = "{cwd}/{theme}/molecular_phenotypes/{theme}.rnaseqc.gene_reads.gct.gz",
    output:
        tpm_filt   = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.tpm.gct.gz",
        count_filt = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.geneCount.gct.gz",
    params:
        container            = config["containers"]["rnaquant"],
        outdir               = "{cwd}/{theme}/molecular_phenotypes",
        low_expr_tpm         = config["rnaseq_qc"]["low_expr_tpm"],
        low_expr_tpm_percent = config["rnaseq_qc"]["low_expr_tpm_percent"],
        rle_filter_percent   = config["rnaseq_qc"]["rle_filter_percent"],
        ds_filter_percent    = config["rnaseq_qc"]["ds_filter_percent"],
        script               = f"{RENOVATED}/molecular_phenotypes/qc/bulk_expression_QC.sh",
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        bash {params.script} qc \
            --container {params.container} \
            --cwd {params.outdir} \
            --tpm-gct {input.tpm_gct} \
            --counts-gct {input.counts_gct} \
            --low-expr-TPM {params.low_expr_tpm} \
            --low-expr-TPM-percent {params.low_expr_tpm_percent} \
            --RLEFilterPercent {params.rle_filter_percent} \
            --DSFilterPercent {params.ds_filter_percent} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 1.4 — Normalization: TMM-CPM + quantile normalization
# ------------------------------------
# NOTE: bulk_expression_normalization.sh is not yet implemented in renovated_code/.
# This rule will fail until the script is added.
rule bulk_expression_normalization:
    """Normalize expression with TMM-CPM-voom and output BED.gz for QTL analysis."""
    input:
        tpm_filt   = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.tpm.gct.gz",
        count_filt = "{cwd}/{theme}/molecular_phenotypes/{theme}.low_expression_filtered.outlier_removed.geneCount.gct.gz",
    output:
        phenotype_bed = "{cwd}/{theme}/molecular_phenotypes/{theme}.{norm_method}.expression.bed.gz",
    params:
        container             = config["containers"]["rnaquant"],
        outdir                = "{cwd}/{theme}/molecular_phenotypes",
        gtf                   = config["reference"]["gtf_ercc"],
        norm_method           = config["normalization"]["method"],
        count_threshold       = config["normalization"]["count_threshold"],
        sample_frac_threshold = config["normalization"]["sample_frac_threshold"],
        sample_lookup         = lambda wc: next(
            t.get("sample_participant_lookup", "")
            for t in config["themes"] if t["name"] == wc.theme
        ),
        script = f"{RENOVATED}/molecular_phenotypes/normalization/bulk_expression_normalization.sh",
    wildcard_constraints:
        norm_method = "[a-z_]+",
    threads: config["resources"]["rna_calling"]["threads"]
    resources:
        mem_mb  = config["resources"]["rna_calling"]["mem_mb"],
        runtime = config["resources"]["rna_calling"]["runtime"],
    shell:
        """
        bash {params.script} normalize \
            --container {params.container} \
            --cwd {params.outdir} \
            --counts-gct {input.count_filt} \
            --tpm-gct {input.tpm_filt} \
            --annotation-gtf {params.gtf} \
            --sample_participant_lookup {params.sample_lookup} \
            --count-threshold {params.count_threshold} \
            --sample-frac-threshold {params.sample_frac_threshold} \
            --normalization-method {params.norm_method} \
            --numThreads {threads}
        """
