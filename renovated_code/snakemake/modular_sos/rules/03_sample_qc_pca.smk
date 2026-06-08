# ============================================================
# Rule Module 03: Per-tissue Sample QC, Kinship & PCA  (Modular SoS)
# ============================================================
# Covers: Sample matching → KING kinship → Unrelated/related QC →
#         FlashPCA (unrelated) → Project related samples onto PC space
#
# SoS notebooks called (Modular SoS wrappers in renovated_code/notebook/modular_sos/):
#   - GWAS_QC.ipynb (genotype_phenotype_sample_overlap, king, qc, qc_no_prune)
#   - PCA.ipynb     (flashpca, project_samples)
# ============================================================

# ------------------------------------
# Step 3.0 — Sample matching
# ------------------------------------
rule sample_match:
    """Identify genotype-phenotype sample overlap and write a keep-samples list."""
    input:
        bed           = lambda wc: get_plink_qc_bed(),
        phenotype_bed = lambda wc: get_phenotype_bed(wc),
    output:
        sample_genotypes = "{cwd}/data_preprocessing/{theme}/genotype_data/{pheno_base}.sample_genotypes.txt",
        sample_overlap   = "{cwd}/data_preprocessing/{theme}/genotype_data/{pheno_base}.sample_overlap.txt",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("sample_match"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/genotype_data",
        sample_lookup_arg = lambda wc: (
            f"--sample_participant_lookup {next(t.get('sample_participant_lookup', '') for t in config['themes'] if t['name'] == wc.theme)}"
            if next(t.get("sample_participant_lookup", "") for t in config["themes"] if t["name"] == wc.theme)
            else ""
        ),
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["default"]["mem_mb"],
        runtime  = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        local_pheno="{params.outdir}/{wildcards.pheno_base}.bed.gz"
        ln -sfn {input.phenotype_bed} "$local_pheno"
        local_genofam="$(printf '%s\n' '{input.bed}' | sed 's/\\.bed$/.fam/')"
        {params.sos_bin} run {params.notebooks_dir}/GWAS_QC.ipynb genotype_phenotype_sample_overlap \
            --cwd {params.outdir} \
            --genoFile "$local_genofam" \
            --phenoFile "$local_pheno" \
            {params.sample_lookup_arg} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        actual_sample_genotypes="{params.outdir}/{wildcards.pheno_base}.bed.sample_genotypes.txt"
        actual_sample_overlap="{params.outdir}/{wildcards.pheno_base}.bed.sample_overlap.txt"
        if [ ! -s "$actual_sample_genotypes" ] || [ ! -s "$actual_sample_overlap" ]; then
            echo "ERROR: sample_match did not produce expected SoS outputs:" >&2
            echo "  $actual_sample_genotypes" >&2
            echo "  $actual_sample_overlap" >&2
            exit 1
        fi
        if [ -f "$actual_sample_genotypes" ]; then
            ln -sfn "$actual_sample_genotypes" "{output.sample_genotypes}"
        fi
        if [ -f "$actual_sample_overlap" ]; then
            ln -sfn "$actual_sample_overlap" "{output.sample_overlap}"
        fi
        if [ ! -s "{output.sample_genotypes}" ] || [ ! -s "{output.sample_overlap}" ]; then
            echo "ERROR: sample_match did not materialize Snakemake outputs:" >&2
            echo "  {output.sample_genotypes}" >&2
            echo "  {output.sample_overlap}" >&2
            exit 1
        fi
        """

# ------------------------------------
# Step 3.1 — KING kinship
# ------------------------------------
rule king_kinship:
    """Run KING kinship analysis and split into related/unrelated sample sets."""
    input:
        bed              = lambda wc: get_plink_qc_bed(),
        sample_genotypes = lambda wc: get_sample_genotypes_path(wc.theme),
    output:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.bed",
        unrelated_bim = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.bim",
        unrelated_fam = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.fam",
        related_id    = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related_id",
        related_bed   = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related.bed",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("king_kinship"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/genotype_data",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["high_mem"]["mem_mb"],
        runtime  = config["resources"]["high_mem"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/GWAS_QC.ipynb king \
            --cwd {params.outdir} \
            --genoFile {input.bed} \
            --keep-samples {input.sample_genotypes} \
            --name {wildcards.theme} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}

        base=$(printf '%s\n' "{output.unrelated_bed}" | sed 's/\\.unrelated\\.bed$//')
        for prefix in "$base.unrelated"; do
            for ext in bed bim fam; do
                if [ ! -s "$prefix.$ext" ]; then
                    echo "ERROR: king_kinship produced missing/empty PLINK file: $prefix.$ext" >&2
                    exit 1
                fi
            done
        done
        if [ -s "$base.related_id" ]; then
            for ext in bed bim fam; do
                if [ ! -s "$base.related.$ext" ]; then
                    echo "ERROR: king_kinship produced missing/empty PLINK file: $base.related.$ext" >&2
                    exit 1
                fi
            done
        elif [ ! -e "$base.related.bed" ]; then
            echo "ERROR: king_kinship did not produce no-related placeholder: $base.related.bed" >&2
            exit 1
        fi
        """

# ------------------------------------
# Step 3.2 — QC on unrelated samples + LD pruning
# ------------------------------------
rule unrelated_qc:
    """Apply QC filters and LD pruning to the unrelated sample set."""
    input:
        unrelated_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.bed",
    output:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.bed",
        pruned_bim = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.bim",
        pruned_fam = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.fam",
        pruned_in  = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.in",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("unrelated_qc"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter    = config["genotype_qc"]["mac_filter"],
        ld_window     = config["genotype_qc"]["ld_window"],
        ld_shift      = config["genotype_qc"]["ld_shift"],
        ld_r2         = config["genotype_qc"]["ld_r2"],
        bad_ld_arg    = "--bad-ld" if config["genotype_qc"].get("bad_ld", False) else "",
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime  = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        {params.sos_bin} run {params.notebooks_dir}/GWAS_QC.ipynb qc \
            --cwd {params.outdir} \
            --genoFile {input.unrelated_bed} \
            --mac-filter {params.mac_filter} \
            --window {params.ld_window} \
            --shift {params.ld_shift} \
            --r2 {params.ld_r2} \
            {params.bad_ld_arg} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}

        prefix=$(printf '%s\n' "{output.pruned_bed}" | sed 's/\\.bed$//')
        for ext in bed bim fam; do
            if [ ! -s "$prefix.$ext" ]; then
                echo "ERROR: unrelated_qc produced missing/empty PLINK file: $prefix.$ext" >&2
                exit 1
            fi
        done
        if [ ! -s "{output.pruned_in}" ]; then
            echo "ERROR: unrelated_qc produced missing/empty prune list: {output.pruned_in}" >&2
            exit 1
        fi
        """

# ------------------------------------
# Step 3.3 — QC on related samples
# ------------------------------------
rule related_qc:
    """Filter related samples using the pruned variant list from unrelated QC."""
    input:
        related_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related.bed",
        related_id  = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related_id",
        pruned_in   = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.in",
    output:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related.plink_qc.extracted.bed",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("related_qc"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/genotype_data",
        mac_filter    = config["genotype_qc"]["mac_filter"],
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["genotype_qc"]["mem_mb"],
        runtime  = config["resources"]["genotype_qc"]["runtime"],
    shell:
        """
        if [ ! -s "{input.related_id}" ]; then
            touch "{output.related_qc_bed}"
            exit 0
        fi

        {params.sos_bin} run {params.notebooks_dir}/GWAS_QC.ipynb qc_no_prune \
            --cwd {params.outdir} \
            --genoFile {input.related_bed} \
            --keep-variants {input.pruned_in} \
            --maf-filter 0 \
            --mac-filter 0 \
            --geno-filter 0 \
            --mind-filter 0.1 \
            --hwe-filter 0 \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}

        prefix=$(printf '%s\n' "{output.related_qc_bed}" | sed 's/\\.bed$//')
        for ext in bed bim fam; do
            if [ ! -s "$prefix.$ext" ]; then
                echo "ERROR: related_qc produced missing/empty PLINK file: $prefix.$ext" >&2
                exit 1
            fi
        done
        """

# ------------------------------------
# Step 3.4 — FlashPCA on unrelated LD-pruned samples
# ------------------------------------
rule flashpca:
    """Compute genotype PCs using flashPCA on the unrelated, LD-pruned sample set."""
    input:
        pruned_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.bed",
    output:
        pca_rds = "{cwd}/data_preprocessing/{theme}/pca/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.pca.rds",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("flashpca"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/pca",
        n_pcs         = config["pca"]["n_pcs"],
        maha_k        = config["pca"]["maha_k"],
        maha_prob     = config["pca"]["maha_prob"],
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["pca"]["mem_mb"],
        runtime  = config["resources"]["pca"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        {params.sos_bin} run {params.notebooks_dir}/PCA.ipynb flashpca_core \
            --cwd {params.outdir} \
            --genoFile {input.pruned_bed} \
            --k {params.n_pcs} \
            --maha-k {params.maha_k} \
            --prob {params.maha_prob} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """

# ------------------------------------
# Step 3.5 — Project related samples onto unrelated PCA space
# ------------------------------------
rule project_samples:
    """Project related samples onto the PCA space of unrelated samples."""
    input:
        related_qc_bed = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related.plink_qc.extracted.bed",
        related_id     = "{cwd}/data_preprocessing/{theme}/genotype_data/" + PLINK_QC_BASENAME + ".{theme}.related_id",
        pca_rds        = "{cwd}/data_preprocessing/{theme}/pca/" + PLINK_QC_BASENAME + ".{theme}.unrelated.plink_qc.prune.pca.rds",
    output:
        projected_rds = "{cwd}/data_preprocessing/{theme}/pca/" + PLINK_QC_BASENAME + ".{theme}.related.plink_qc.extracted.pca.projected.rds",
    params:
        sos_bin       = SOS_BIN,
        sos_sched     = sos_sched("project_samples"),
        notebooks_dir = NOTEBOOKS,
        renovated_dir = RENOVATED,
        outdir        = "{cwd}/data_preprocessing/{theme}/pca",
        maha_k        = config["pca"]["maha_k"],
        maha_prob     = config["pca"]["maha_prob"],
        dry_run       = DRY_RUN_SOS,
    threads: 1
    resources:
        mem_mb   = config["resources"]["pca"]["mem_mb"],
        runtime  = config["resources"]["pca"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        if [ ! -s "{input.related_id}" ]; then
            cp -f "{input.pca_rds}" "{output.projected_rds}"
            exit 0
        fi

        {params.sos_bin} run {params.notebooks_dir}/PCA.ipynb project_samples_core \
            --cwd {params.outdir} \
            --genoFile {input.related_qc_bed} \
            --pca-model {input.pca_rds} \
            --maha-k {params.maha_k} \
            --renovated-code-dir {params.renovated_dir} \
            --numThreads {threads} {params.dry_run} {params.sos_sched}
        """
