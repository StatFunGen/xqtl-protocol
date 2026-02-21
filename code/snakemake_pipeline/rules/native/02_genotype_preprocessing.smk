# ============================================================
# Rule Module 02: Genotype Preprocessing (native scripts)
# ============================================================
# Calls renovated_code/ shell wrappers instead of SoS notebooks.
# Interface flags are identical to the SoS versions.
# ============================================================

RENOVATED = config["renovated_code_dir"]

# ------------------------------------
# Step 2.1 — VCF QC
# ------------------------------------
rule vcf_qc:
    """Run VCF quality control: left-normalization, dbSNP annotation, variant filtering."""
    input:
        vcf = lambda wc: config["genotype"]["vcf_files"][int(wc.vcf_idx)],
    output:
        vcf_qc = "{cwd}/data_preprocessing/genotype/vcf_qc_{vcf_idx}.add_chr.leftnorm.filtered.vcf.gz",
    params:
        container = config["containers"]["bioinfo"],
        outdir    = "{cwd}/data_preprocessing/genotype",
        fasta     = config["reference"]["fasta"],
        dbsnp     = config["reference"]["dbsnp"],
        script    = f"{RENOVATED}/data_preprocessing/genotype/VCF_QC.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb  = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        bash {params.script} qc {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.vcf} \
            --dbsnp-variants {params.dbsnp} \
            --reference-genome {params.fasta} \
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
        container = config["containers"]["bioinfo"],
        outdir    = "{cwd}/data_preprocessing/genotype",
        vcf_list  = "{cwd}/data_preprocessing/genotype/vcf_qc_files.list",
        script    = f"{RENOVATED}/data_preprocessing/genotype/genotype_formatting.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb  = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        printf '%s\n' {input.vcf_qc} > {params.vcf_list}
        if [ $(wc -l < {params.vcf_list}) -gt 1 ]; then
            bash {params.script} merge_plink {params.dry_run} \
                --container {params.container} \
                --cwd {params.outdir} \
                --genoFile {params.vcf_list} \
                --name xqtl_protocol_data.converted \
                --numThreads {threads}
        else
            bash {params.script} vcf_to_plink {params.dry_run} \
                --container {params.container} \
                --cwd {params.outdir} \
                --genoFile $(cat {params.vcf_list}) \
                --name xqtl_protocol_data.converted \
                --numThreads {threads}
        fi
        """

# ------------------------------------
# Step 2.3 — Genome-wide GWAS QC
# ------------------------------------
rule plink_qc:
    """Apply genome-wide GWAS QC filters (MAF, MAC, missingness, HWE)."""
    input:
        bed = get_input_plink,
    output:
        bed = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
        bim = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bim",
        fam = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.fam",
    params:
        container   = config["containers"]["bioinfo"],
        outdir      = "{cwd}/data_preprocessing/genotype",
        mac_filter  = config["genotype_qc"]["mac_filter"],
        maf_filter  = config["genotype_qc"]["maf_filter"],
        geno_filter = config["genotype_qc"]["geno_filter"],
        mind_filter = config["genotype_qc"]["mind_filter"],
        hwe_filter  = config["genotype_qc"]["hwe_filter"],
        script      = f"{RENOVATED}/data_preprocessing/genotype/GWAS_QC.sh",
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
            --genoFile {input.bed} \
            --name xqtl_protocol_data.plink_qc \
            --mac-filter {params.mac_filter} \
            --maf-filter {params.maf_filter} \
            --geno-filter {params.geno_filter} \
            --mind-filter {params.mind_filter} \
            --hwe-filter {params.hwe_filter} \
            --numThreads {threads}
        """

# ------------------------------------
# Step 2.4 — Split plink genotype files by chromosome
# ------------------------------------
rule genotype_by_chrom:
    """Split plink genotype into per-chromosome files for parallel QTL analysis."""
    input:
        bed = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.bed",
    output:
        chrom_list = "{cwd}/data_preprocessing/genotype/xqtl_protocol_data.plink_qc.genotype_by_chrom_files.txt",
    params:
        container = config["containers"]["bioinfo"],
        outdir    = "{cwd}/data_preprocessing/genotype",
        chroms    = " ".join(config["chromosomes"]),
        script    = f"{RENOVATED}/data_preprocessing/genotype/genotype_formatting.sh",
        dry_run     = DRY_RUN_NATIVE,
    threads: config["resources"]["genotype_qc"]["threads"]
    resources:
        mem_mb  = config["resources"]["genotype_qc"]["mem_mb"],
        runtime = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        bash {params.script} genotype_by_chrom {params.dry_run} \
            --container {params.container} \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --name xqtl_protocol_data.plink_qc \
            --chrom {params.chroms} \
            --numThreads {threads}
        """
