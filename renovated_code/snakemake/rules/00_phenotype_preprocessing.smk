# ============================================================
# Rule Module 00: Phenotype Preprocessing (non-RNA-seq)
# ============================================================
# Covers:
#   - Gene coordinate annotation: adds genomic coordinates to a
#     raw molecular phenotype matrix and converts it to BED format
#   - Missing value imputation: fills in missing entries using
#     gEBMF (default) or other methods
#
# Mirrors: pipeline/phenotype_preprocessing.ipynb steps i–ii
#
# SoS notebooks called:
#   - pipeline/gene_annotation.ipynb    (annotate_coord)
#   - pipeline/phenotype_imputation.ipynb (gEBMF, missforest, knn, ...)
#
# Activated when: start_from: "raw_phenotype"
#
# Output naming notes:
#   annotate_coord uses {input_basename}.bed.gz for its output.
#   A no-extension symlink named {theme} is created before the SoS
#   call so the output is deterministically {theme}.bed.gz.
#
#   phenotype_imputation uses {input_stem}.imputed.bed.gz where
#   stem("{theme}.bed.gz") = "{theme}.bed", yielding the output
#   {theme}.bed.imputed.bed.gz.  This naming matches the SoS
#   notebook's built-in convention.
# ============================================================

# ------------------------------------
# Step 0.1 — Gene coordinate annotation
# ------------------------------------
# Converts a raw molecular phenotype matrix (gene expression, protein,
# ATAC, etc.) into a 4-column BED file by looking up genomic
# coordinates from the provided GTF.
#
# Implementation detail: the SoS notebook names its output
#   {cwd}/{input_basename}.bed.gz
# where "basename" strips every extension (e.g. ".csv", ".tsv").
# To produce {theme}.bed.gz deterministically we create a temporary
# no-extension symlink {outdir}/{theme} → raw_phenotype_file, pass
# that symlink to SoS, then remove it after the run.
rule annotate_coord:
    """Annotate genomic coordinates and convert raw phenotype matrix to BED format."""
    input:
        raw_pheno = lambda wc: next(
            t["raw_phenotype_file"] for t in config["themes"] if t["name"] == wc.theme
        ),
        gtf = config["reference"]["gtf_collapsed"],
    output:
        phenotype_bed = "{cwd}/{theme}/phenotype_preprocessing/{theme}.bed.gz",
        region_list   = "{cwd}/{theme}/phenotype_preprocessing/{theme}.region_list.txt",
    params:
        pipeline_dir  = config["pipeline_dir"],
        container     = config["containers"]["rnaquant"],
        outdir        = "{cwd}/{theme}/phenotype_preprocessing",
        pheno_id_col  = lambda wc: next(
            t.get("phenotype_id_column", "gene_id")
            for t in config["themes"] if t["name"] == wc.theme
        ),
        trait_type    = lambda wc: next(
            t.get("molecular_trait_type", "gene")
            for t in config["themes"] if t["name"] == wc.theme
        ),
        sample_lookup = lambda wc: next(
            t.get("sample_participant_lookup", ".")
            for t in config["themes"] if t["name"] == wc.theme
        ),
        dry_run       = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        # Create a no-extension symlink so SoS outputs {theme}.bed.gz
        SYMLINK={params.outdir}/{wildcards.theme}
        ln -sf $(realpath {input.raw_pheno}) "$SYMLINK"
        sos run {params.pipeline_dir}/gene_annotation.ipynb annotate_coord {params.dry_run} \
            --cwd {params.outdir} \
            --phenoFile "$SYMLINK" \
            --coordinate-annotation {input.gtf} \
            --phenotype-id-column {params.pheno_id_col} \
            --molecular-trait-type {params.trait_type} \
            --sample-participant-lookup {params.sample_lookup} \
            --container {params.container} \
            --numThreads {threads}
        rm -f "$SYMLINK"
        """


# ------------------------------------
# Step 0.2 — Missing value imputation (optional)
# ------------------------------------
# Imputes missing entries in the annotated BED file.  The default
# method is gEBMF (grouped Empirical Bayes Matrix Factorization).
# Other options: EBMF, missforest, knn, soft, mean, lod.
#
# Only executed when phenotype_preprocessing.run_imputation: true.
#
# Output naming: the SoS notebook uses {input_stem}.imputed.bed.gz.
# For input {theme}.bed.gz, stem = {theme}.bed (strips last extension),
# so the output is {theme}.bed.imputed.bed.gz.
rule phenotype_impute:
    """Impute missing molecular phenotype values (default: gEBMF)."""
    input:
        phenotype_bed = "{cwd}/{theme}/phenotype_preprocessing/{theme}.bed.gz",
    output:
        imputed_bed = "{cwd}/{theme}/phenotype_preprocessing/{theme}.bed.imputed.bed.gz",
    params:
        pipeline_dir   = config["pipeline_dir"],
        container      = config["containers"]["rnaquant"],
        outdir         = "{cwd}/{theme}/phenotype_preprocessing",
        impute_method  = config["phenotype_preprocessing"]["impute_method"],
        num_factor     = config["phenotype_preprocessing"]["num_factor"],
        qc_flag        = lambda _: (
            "" if config["phenotype_preprocessing"]["qc_prior_to_impute"]
            else "--no-qc-prior-to-impute"
        ),
        dry_run        = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        sos run {params.pipeline_dir}/phenotype_imputation.ipynb {params.impute_method} {params.dry_run} \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            {params.qc_flag} \
            --num-factor {params.num_factor} \
            --container {params.container} \
            --numThreads {threads}
        """
