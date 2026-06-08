# ============================================================
# Rule Module 00: Phenotype Preprocessing (Modular SoS)
# ============================================================
# Mirrors source-Snakemake phenotype preprocessing while using
# Modular SoS SoS wrappers and modular scripts.
#
# Activated when: start_from: "raw_phenotype"
# ============================================================

rule annotate_coord:
    """Annotate genomic coordinates and convert raw phenotype matrix to BED format."""
    input:
        raw_pheno = lambda wc: get_theme_raw_phenotype_file(wc.theme),
        gtf       = config["reference"]["gtf_collapsed"],
    output:
        phenotype_bed = "{cwd}/{theme}/phenotype_preprocessing/{theme}.bed.gz",
        region_list   = "{cwd}/{theme}/phenotype_preprocessing/{theme}.region_list.txt",
    params:
        sos_bin       = SOS_BIN,
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/{theme}/phenotype_preprocessing",
        pheno_id_col  = lambda wc: _theme_cfg(wc.theme).get("phenotype_id_column", "gene_id"),
        trait_type    = lambda wc: _theme_cfg(wc.theme).get("molecular_trait_type", "gene"),
        sample_lookup = lambda wc: _theme_cfg(wc.theme).get("sample_participant_lookup", "."),
        auxiliary_map = lambda wc: _theme_cfg(wc.theme).get("auxiliary_id_mapping", "."),
        strip_id      = lambda wc: "--strip-id" if _theme_cfg(wc.theme).get("strip_id", False) else "",
        symlink_suffix = lambda wc: get_raw_phenotype_symlink_suffix(wc.theme),
        sep_arg       = lambda wc: (
            f"--sep {shlex.quote(str(_theme_cfg(wc.theme).get('phenotype_sep')))}"
            if _theme_cfg(wc.theme).get("phenotype_sep") is not None else ""
        ),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        SYMLINK="{params.outdir}/{wildcards.theme}{params.symlink_suffix}"
        trap 'rm -f "$SYMLINK"' EXIT
        ln -sf "$(realpath "{input.raw_pheno}")" "$SYMLINK"
        {params.sos_bin} run {params.notebooks_dir}/gene_annotation.ipynb annotate_coord \
            --cwd {params.outdir} \
            --phenoFile "$SYMLINK" \
            --coordinate-annotation {input.gtf} \
            --phenotype-id-column {params.pheno_id_col} \
            --molecular-trait-type {params.trait_type} \
            --sample-participant-lookup {params.sample_lookup} \
            --auxiliary-id-mapping {params.auxiliary_map} \
            {params.sep_arg} \
            {params.strip_id} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} || exit $?
        """


rule phenotype_impute:
    """Impute missing molecular phenotype values when configured."""
    input:
        phenotype_bed = "{cwd}/{theme}/phenotype_preprocessing/{theme}.bed.gz",
    output:
        imputed_bed = "{cwd}/{theme}/phenotype_preprocessing/{theme}.bed.imputed.bed.gz",
    params:
        sos_bin       = SOS_BIN,
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/{theme}/phenotype_preprocessing",
        impute_method = config["phenotype_preprocessing"]["impute_method"],
        num_factor    = config["phenotype_preprocessing"]["num_factor"],
        qc_flag       = lambda _: (
            "" if config["phenotype_preprocessing"]["qc_prior_to_impute"]
            else "--no-qc-prior-to-impute"
        ),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/phenotype_imputation.ipynb {params.impute_method} \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype_bed} \
            {params.qc_flag} \
            --num-factor {params.num_factor} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} || exit $?
        """
