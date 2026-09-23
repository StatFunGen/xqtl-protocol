# ============================================================
# Rule Module 09: Functional Fine-mapping (fSuSiE)  (script-backed)
# ============================================================
# Covers: fSuSiE fine-mapping of functional phenotypes over TAD regions
#
# SoS notebooks called (script-backed wrappers in pipeline/):
#   - phenotype_formatting.ipynb (phenotype_annotate_by_tad)
#   - mnm_regression.ipynb       (qtl_dataset_construct, fsusie)
# ============================================================

# ------------------------------------
# Step 9.1 — Build the single-context QtlDataset inputs
# ------------------------------------
rule fsusie_qtl_manifest:
    """Describe one phenotype matrix, its features, and aligned samples for pecotmr."""
    input:
        phenotype      = lambda wc: get_phenotype_bed_by_theme(wc.theme),
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        manifest   = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.phenotype_manifest.tsv",
        region_ids = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.region_ids.txt",
        sample_ids = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.sample_ids.txt",
    params:
        script = MODULAR_SCRIPT_DIR + "/data_preprocessing/phenotype/prepare_qtl_manifest.py",
        chromosomes = " ".join(config["chromosomes"]),
        phenotype_id_column = lambda wc: _theme_cfg(wc.theme).get("phenotype_id_column", "ID"),
    threads: 1
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        python3 {params.script} \
            --bed {input.phenotype} \
            --covariates {input.hidden_factors} \
            --context {wildcards.theme} \
            --chromosomes {params.chromosomes} \
            --phenotype-id-column {params.phenotype_id_column} \
            --phenotype-manifest {output.manifest} \
            --region-ids {output.region_ids} \
            --sample-ids {output.sample_ids}
        """


# ------------------------------------
# Step 9.2 — Construct the QtlDataset once per context
# ------------------------------------
rule fsusie_qtl_dataset:
    """Build the QtlDataset shared by every regional fSuSiE task."""
    input:
        genotype = lambda wc: config["finemapping"].get("genotype_file", get_plink_qc_bed()),
        manifest = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.phenotype_manifest.tsv",
        sample_ids = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.sample_ids.txt",
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        qtl_dataset = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.qtl_dataset.rds",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("qtl_dataset_construct"),
        notebooks_dir = NOTEBOOKS,
        modular_script_dir = MODULAR_SCRIPT_DIR,
        outdir        = CWD + "/finemapping/{theme}/fsusie",
        maf           = config["finemapping"]["maf"],
        mac           = config["association"]["mac_threshold"],
        sos_mem       = sos_mem_arg(config["resources"]["finemapping"]["mem_mb"]),
        sos_walltime  = sos_walltime_arg(config["resources"]["finemapping"]["runtime"], sos_queue("qtl_dataset_construct")),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb  = config["resources"]["finemapping"]["mem_mb"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/mnm_regression.ipynb qtl_dataset_construct \
            --cwd {params.outdir} \
            --name {wildcards.theme} \
            --study {wildcards.theme} \
            --genoFile {input.genotype} \
            --phenoFile {input.manifest} \
            --covFile {input.hidden_factors} \
            --transpose-covariates \
            --maf-cutoff {params.maf} \
            --mac-cutoff {params.mac} \
            --keep-samples {input.sample_ids} \
            --mem {params.sos_mem} \
            --walltime {params.sos_walltime} \
            --modular-script-dir {params.modular_script_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """


# ------------------------------------
# Step 9.3 — TAD regions holding enough phenotype features
# ------------------------------------
rule fsusie_region_list:
    """Keep the TADs that contain at least phenotype_per_tad phenotype features."""
    input:
        phenotype = lambda wc: get_phenotype_bed_by_theme(wc.theme),
        tad_list  = config["finemapping"].get("fsusie", {}).get("tad_list", "") or [],
    output:
        region_list = CWD + "/finemapping/{theme}/fsusie/{region_list_base}.region_list",
    params:
        sos_bin       = SOS_BIN,
        notebooks_dir = NOTEBOOKS,
        modular_script_dir = MODULAR_SCRIPT_DIR,
        outdir        = CWD + "/finemapping/{theme}/fsusie",
        phenotype_per_tad = config["finemapping"].get("fsusie", {}).get("phenotype_per_tad", 2),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/phenotype_formatting.ipynb phenotype_annotate_by_tad \
            --cwd {params.outdir} \
            --phenoFile {input.phenotype} \
            --TAD-list {input.tad_list} \
            --phenotype-per-tad {params.phenotype_per_tad} \
            --modular-script-dir {params.modular_script_dir} \
            --numThreads {threads} {params.dry_run}
        """


# ------------------------------------
# Step 9.4 — fSuSiE per TAD region over the QtlDataset
# ------------------------------------
rule fsusie:
    """Fine-map functional phenotypes across each retained TAD region."""
    input:
        genotype = lambda wc: config["finemapping"].get("genotype_file", get_plink_qc_bed()),
        manifest = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.phenotype_manifest.tsv",
        qtl_dataset = CWD + "/finemapping/{theme}/fsusie/qtl_dataset/{theme}.qtl_dataset.rds",
        region_list = lambda wc: get_fsusie_region_list(wc.theme),
        hidden_factors = lambda wc: get_hidden_factors(wc),
    output:
        done = CWD + "/finemapping/{theme}/fsusie/.done_fsusie",
        table = CWD + "/finemapping/{theme}/fsusie/fsusie/{theme}.exported.bed.gz",
        index = CWD + "/finemapping/{theme}/fsusie/fsusie/{theme}.exported.bed.gz.tbi",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("fsusie"),
        notebooks_dir = NOTEBOOKS,
        modular_script_dir = MODULAR_SCRIPT_DIR,
        outdir        = CWD + "/finemapping/{theme}/fsusie",
        pip_cutoff    = config["finemapping"]["pip_cutoff"],
        cis_window    = config["finemapping"].get("fsusie", {}).get("cis_window", 0),
        post_processing = config["finemapping"].get("fsusie", {}).get("post_processing", "TI"),
        susie_top_pc  = config["finemapping"].get("fsusie", {}).get("susie_top_pc", 0),
        chromosomes   = " ".join(config["chromosomes"]),
        coverage      = " ".join(str(x) for x in config["finemapping"]["coverage"]),
        seed          = config.get("analysis", {}).get("seed", 999),
        sos_mem       = sos_mem_arg(config["resources"]["finemapping"]["mem_mb"]),
        sos_walltime  = sos_walltime_arg(config["resources"]["finemapping"]["runtime"], sos_queue("fsusie")),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb  = config["resources"]["finemapping"]["mem_mb"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/mnm_regression.ipynb fsusie \
            --cwd {params.outdir} \
            --name {wildcards.theme} \
            --genoFile {input.genotype} \
            --phenoFile {input.manifest} \
            --covFile {input.hidden_factors} \
            --customized-association-windows {input.region_list} \
            --cis-window {params.cis_window} \
            --chromosomes {params.chromosomes} \
            --susie-top-pc {params.susie_top_pc} \
            --post-processing {params.post_processing} \
            --pip-cutoff {params.pip_cutoff} \
            --coverage {params.coverage} \
            --seed {params.seed} \
            --mem {params.sos_mem} \
            --walltime {params.sos_walltime} \
            --modular-script-dir {params.modular_script_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        status=$?
        if [ "$status" -ne 0 ]; then exit "$status"; fi
        touch {output.done}
        """
