# ============================================================
# Rule Module 02: Genotype Preprocessing
# ============================================================
# Covers: VCF QC → Plink conversion → GWAS QC (per-sample and per-variant) →
#         Merge plink files → Split by chromosome
#
# Mirrors: eQTL_analysis_commands.ipynb stages:
#   VCF_QC → merge_plink → plink_QC → plink_per_chrom → plink_to_vcf
#
# SoS notebooks called:
#   - pipeline/VCF_QC.ipynb (qc)
#   - pipeline/genotype_formatting.ipynb (vcf_to_plink, merge_plink, genotype_by_chrom)
#   - pipeline/GWAS_QC.ipynb (qc_no_prune)
# ============================================================

# ------------------------------------
# Step 2.1 — VCF QC: normalize, annotate, and filter variants
# ------------------------------------
# Performs left-normalization, chromosome renaming, dbSNP annotation,
# and basic variant-level quality control.
rule vcf_qc:
    """Run VCF quality control: left-normalization, dbSNP annotation, variant filtering."""
    input:
        vcf = lambda wc: config["genotype"]["vcf_files"][int(wc.vcf_idx)],
    output:
        vcf_qc = "{cwd}/data_preprocessing/genotype/vcf_qc_{vcf_idx}.add_chr.leftnorm.filtered.vcf.gz",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/genotype",
        fasta        = config["reference"]["fasta"],
        dbsnp        = config["reference"]["dbsnp"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/VCF_QC.ipynb qc {params.dry_run} \
            --cwd {params.outdir} \
            --genoFile {input.vcf} \
            --dbsnp-variants {params.dbsnp} \
            --reference-genome {params.fasta} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 2.2 — Convert QC'd VCF(s) to plink format
# ------------------------------------
rule vcf_to_plink:
    """Convert QC-filtered VCF files to PLINK binary format."""
    input:
        vcf_qc = expand(
            "{cwd}/data_preprocessing/genotype/vcf_qc_{vcf_idx}.add_chr.leftnorm.filtered.vcf.gz",
            cwd="{cwd}",
            vcf_idx=range(len(config["genotype"]["vcf_files"]))
        ),
    output:
        bed = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.converted.bed",
        bim = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.converted.bim",
        fam = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.converted.fam",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/genotype",
        # Write VCF paths to a temporary list file consumed by the SoS notebook
        vcf_list     = "{cwd}/data_preprocessing/genotype/vcf_qc_files.list",
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        printf '%s\n' {input.vcf_qc} > {params.vcf_list}
        if [ $(wc -l < {params.vcf_list}) -gt 1 ]; then
            # Multiple VCFs: merge then convert
            sos run {params.pipeline_dir}/genotype_formatting.ipynb merge_plink {params.dry_run} \
                --cwd {params.outdir} \
                --genoFile {params.vcf_list} \
                --name xqtl_protocol_data.converted \
                --container {params.container} \
                --numThreads {threads}
        else
            # Single VCF: convert directly
            sos run {params.pipeline_dir}/genotype_formatting.ipynb vcf_to_plink {params.dry_run} \
                --cwd {params.outdir} \
                --genoFile $(cat {params.vcf_list}) \
                --name xqtl_protocol_data.converted \
                --container {params.container} \
                --numThreads {threads}
        fi
        """

# ------------------------------------
# Step 2.3 — Genome-wide GWAS QC
# ------------------------------------
# Applies MAF/MAC, missingness, and HWE filters across all samples.
# This is a pre-filter before tissue-specific kinship analysis.
# Input is either:
#   (a) The converted plink file produced by vcf_to_plink (default), or
#   (b) A pre-existing plink file from config["genotype"]["plink_bed"].
rule plink_qc:
    """Apply genome-wide GWAS QC filters (MAF, MAC, missingness, HWE)."""
    input:
        bed = get_input_plink,
    output:
        bed = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        bim = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bim",
        fam = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.fam",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/genotype",
        mac_filter   = config["genotype_qc"]["mac_filter"],
        maf_filter   = config["genotype_qc"]["maf_filter"],
        geno_filter  = config["genotype_qc"]["geno_filter"],
        mind_filter  = config["genotype_qc"]["mind_filter"],
        hwe_filter   = config["genotype_qc"]["hwe_filter"],
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/GWAS_QC.ipynb qc_no_prune {params.dry_run} \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --name xqtl_protocol_data.plink_qc \
            --mac-filter {params.mac_filter} \
            --maf-filter {params.maf_filter} \
            --geno-filter {params.geno_filter} \
            --mind-filter {params.mind_filter} \
            --hwe-filter {params.hwe_filter} \
            --container {params.container} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 2.4 — Split plink genotype files by chromosome
# ------------------------------------
# Creates one plink set per chromosome and writes a file-list consumed by
# TensorQTL and the fine-mapping steps.
# Actual SoS step name: genotype_by_chrom (not plink_by_chrom).
# Output list file: {name}.genotype_by_chrom_files.txt (SoS naming convention).
rule genotype_by_chrom:
    """Split plink genotype into per-chromosome files for parallel QTL analysis."""
    input:
        bed = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
    output:
        chrom_list = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.genotype_by_chrom_files.txt",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        outdir       = "{cwd}/data_preprocessing/genotype",
        chroms       = " ".join(config["chromosomes"]),
        dry_run     = DRY_RUN_SOS,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/genotype_formatting.ipynb genotype_by_chrom {params.dry_run} \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --name xqtl_protocol_data.plink_qc \
            --chrom {params.chroms} \
            --container {params.container} \
            --numThreads {threads}
        """
