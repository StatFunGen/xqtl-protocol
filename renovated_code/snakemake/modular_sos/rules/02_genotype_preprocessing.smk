# ============================================================
# Rule Module 02: Genotype Preprocessing  (Modular SoS)
# ============================================================
# Covers: VCF QC → Plink conversion → Merge plink → GWAS QC → Split by chromosome
#
# SoS notebooks called (Modular SoS wrappers in modular_sos/notebooks/):
#   - VCF_QC.ipynb              (qc)
#   - genotype_formatting.ipynb (vcf_to_plink, merge_plink, genotype_by_chrom)
#   - GWAS_QC.ipynb             (qc_no_prune)
# ============================================================

VCF_QC_PREFIXES = [
    (
        f"{_vcf_basename(vcf)}.leftnorm"
        f"{'.gt_only' if config.get('genotype_qc', {}).get('gt_only_vcf_qc', False) else ''}"
        ".bcftools_qc"
    )
    for vcf in config["genotype"].get("vcf_files", [])
]
VCF_QC_INPUTS = dict(zip(VCF_QC_PREFIXES, config["genotype"].get("vcf_files", [])))

# ------------------------------------
# Step 2.1 — VCF QC
# ------------------------------------
rule vcf_qc:
    """Run VCF quality control: left-normalization, dbSNP annotation, variant filtering."""
    input:
        vcf = lambda wc: VCF_QC_INPUTS[wc.prefix],
    output:
        vcf_qc = "{cwd}/data_preprocessing/genotype/{prefix}.vcf.gz",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        # container     = config.get("containers", {}).get("bioinfo", ""),  # Disabled - not in config
        outdir        = "{cwd}/data_preprocessing/genotype",
        fasta         = config["reference"]["fasta"],
        dbsnp         = config["reference"]["dbsnp"],
        gt_only_vcf_qc = str(config.get("genotype_qc", {}).get("gt_only_vcf_qc", False)).lower(),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime  = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.notebooks_dir}/VCF_QC.ipynb qc \
            --cwd {params.outdir} \
            --genoFile {input.vcf} \
            --dbsnp-variants {params.dbsnp} \
            --reference-genome {params.fasta} \
            --gt-only-vcf-qc {params.gt_only_vcf_qc} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 2.2 — Convert QC'd VCF(s) to plink format
# ------------------------------------
rule vcf_to_plink:
    """Convert each QC-filtered VCF file to its own PLINK binary bundle."""
    input:
        vcf_qc = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.vcf.gz",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
    output:
        bed = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.bed",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
        bim = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.bim",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
        fam = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.fam",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
        plink_list = "{cwd}/data_preprocessing/genotype/vcf_to_plink_files.list",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        # container     = config.get("containers", {}).get("bioinfo", ""),  # Disabled - not in config
        outdir        = "{cwd}/data_preprocessing/genotype",
        vcf_list      = "{cwd}/data_preprocessing/genotype/vcf_qc_files.list",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime  = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        printf '%s\n' {input.vcf_qc} > {params.vcf_list}
        sos run {params.notebooks_dir}/genotype_formatting.ipynb vcf_to_plink \
            --cwd {params.outdir} \
            --genoFile {params.vcf_list} \
             \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
        printf '%s\n' {output.bed} > {output.plink_list}
        """

# ------------------------------------
# Step 2.3 — Merge per-chromosome plink files
# ------------------------------------
rule merge_plink:
    """Merge per-VCF PLINK bundles into one genome-wide PLINK dataset."""
    input:
        bed = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.bed",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
        bim = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.bim",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
        fam = expand(
            "{cwd}/data_preprocessing/genotype/{prefix}.fam",
            cwd="{cwd}",
            prefix=VCF_QC_PREFIXES,
        ),
        plink_list = "{cwd}/data_preprocessing/genotype/vcf_to_plink_files.list",
    output:
        bed = "{cwd}/data_preprocessing/genotype/" + CONVERTED_PLINK_BASENAME + ".bed",
        bim = "{cwd}/data_preprocessing/genotype/" + CONVERTED_PLINK_BASENAME + ".bim",
        fam = "{cwd}/data_preprocessing/genotype/" + CONVERTED_PLINK_BASENAME + ".fam",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/genotype",
        output_name   = CONVERTED_PLINK_BASENAME,
        merge_mem     = "300G",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = 300000,
        runtime  = config["resources"]["high_mem"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        if [ $(wc -l < {input.plink_list}) -eq 1 ]; then
            input_prefix=$(sed -n '1p' {input.plink_list})
            input_prefix=${{input_prefix%.bed}}
            cp "${{input_prefix}}.bed" {output.bed}
            cp "${{input_prefix}}.bim" {output.bim}
            cp "${{input_prefix}}.fam" {output.fam}
        else
            sos run {params.notebooks_dir}/genotype_formatting.ipynb merge_plink \
                --cwd {params.outdir} \
                --genoFile {input.plink_list} \
                --name {params.output_name} \
                --mem {params.merge_mem} \
                 \
                --renovated-code-dir {params.renovated_dir} \
                --numThreads {threads} {params.dry_run}
        fi
        """

# ------------------------------------
# Step 2.4 — Genome-wide GWAS QC
# ------------------------------------
rule plink_qc:
    """Apply genome-wide GWAS QC filters (MAF, MAC, missingness, HWE)."""
    input:
        bed = get_input_plink,
    output:
        bed = "{cwd}/data_preprocessing/genotype/" + PLINK_QC_BASENAME + ".bed",
        bim = "{cwd}/data_preprocessing/genotype/" + PLINK_QC_BASENAME + ".bim",
        fam = "{cwd}/data_preprocessing/genotype/" + PLINK_QC_BASENAME + ".fam",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        # container     = config.get("containers", {}).get("bioinfo", ""),  # Disabled - not in config
        outdir        = "{cwd}/data_preprocessing/genotype",
        mac_filter    = config["genotype_qc"]["mac_filter"],
        maf_filter    = config["genotype_qc"]["maf_filter"],
        geno_filter   = config["genotype_qc"]["geno_filter"],
        mind_filter   = config["genotype_qc"]["mind_filter"],
        hwe_filter    = config["genotype_qc"]["hwe_filter"],
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime  = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        sos run {params.notebooks_dir}/GWAS_QC.ipynb qc_no_prune \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --mac-filter {params.mac_filter} \
            --maf-filter {params.maf_filter} \
            --geno-filter {params.geno_filter} \
            --mind-filter {params.mind_filter} \
            --hwe-filter {params.hwe_filter} \
             \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
        """

# ------------------------------------
# Step 2.5 — Split plink genotype by chromosome
# ------------------------------------
rule genotype_by_chrom:
    """Split plink genotype into per-chromosome files for parallel QTL analysis."""
    input:
        bed = "{cwd}/data_preprocessing/genotype/" + PLINK_QC_BASENAME + ".bed",
    output:
        chrom_list = "{cwd}/data_preprocessing/genotype/" + PLINK_QC_BASENAME + ".genotype_by_chrom_files.txt",
    params:
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        # container     = config.get("containers", {}).get("bioinfo", ""),  # Disabled - not in config
        outdir        = "{cwd}/data_preprocessing/genotype",
        chroms        = " ".join(
            chrom[3:] if str(chrom).lower().startswith("chr") else str(chrom)
            for chrom in config["chromosomes"]
        ),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime  = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        sos run {params.notebooks_dir}/genotype_formatting.ipynb genotype_by_chrom \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --chrom {params.chroms} \
             \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run}
        """
