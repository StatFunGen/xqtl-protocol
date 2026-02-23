# ============================================================
# Rule Module 01: Post-Finemapping Results Export & Visualization
# ============================================================
# Converts univariate SuSiE finemapping RDS results to tabular
# TSV format, exports high-confidence loci, and generates
# credible-set visualizations.
#
# SoS notebooks called:
#   - pipeline/mnm_postprocessing.ipynb (susie_to_tsv_1, susie_to_tsv_2,
#       export_top_loci, susie_pip_landscape_plot, susie_upsetR_cs_plot)
#
# Dependency chain:
#   susie_to_tsv → export_top_loci
#                → pip_landscape_plots
#                → upsetr_cs_plots
# ============================================================


# ------------------------------------
# Step 1.1 — Convert finemapping RDS → TSV
# ------------------------------------
# Globs all *.rds files produced by susie_twas inside the
# per-theme finemapping directory and converts them to long-form TSV.
# Two SoS sub-steps run back-to-back:
#   susie_to_tsv_1: per-region RDS → per-region TSV
#   susie_to_tsv_2: aggregates per-region TSVs
#
# Required config keys (under each theme entry):
#   finemapping_dir / name → resolved path to the susie_twas output folder
# ============================================================
rule susie_to_tsv:
    """Convert finemapping RDS results to aggregated TSV tables."""
    input:
        finemapping_dir = lambda wc: _get_finemapping_dir(wc),
    output:
        done = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        outdir       = "{cwd}/postprocessing/{theme}/susie_tsv",
        name         = "{theme}",
        pip_thres    = config["postprocessing"]["pip_thres"],
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        RDS_FILES=$(ls {input.finemapping_dir}/*.rds 2>/dev/null | tr '\\n' ' ')
        if [ -z "$RDS_FILES" ]; then
            echo "ERROR: No .rds files found in {input.finemapping_dir}" >&2
            exit 1
        fi
        sos run {params.pipeline_dir}/mnm_postprocessing.ipynb \\
            susie_to_tsv_1 susie_to_tsv_2 {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --rds_path $RDS_FILES \\
            --pip_thres {params.pip_thres} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 1.2 — Export top credible-set loci
# ------------------------------------
# Extracts the highest-confidence variants (above PIP / LFSR
# thresholds) from the aggregated TSV into a clean summary table.
rule export_top_loci:
    """Export top fine-mapped loci above PIP / LFSR thresholds."""
    input:
        done = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
    output:
        done = "{cwd}/postprocessing/{theme}/top_loci/.done_top_loci",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        tsv_dir      = "{cwd}/postprocessing/{theme}/susie_tsv",
        outdir       = "{cwd}/postprocessing/{theme}/top_loci",
        name         = "{theme}",
        pip_thres    = config["postprocessing"]["pip_thres"],
        lfsr_thres   = config["postprocessing"]["lfsr_thres"],
        cs_size      = config["postprocessing"]["cs_size"],
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        TSV_FILES=$(ls {params.tsv_dir}/*.tsv.gz 2>/dev/null | tr '\\n' ' ')
        if [ -z "$TSV_FILES" ]; then
            echo "ERROR: No .tsv.gz files found in {params.tsv_dir}" >&2
            exit 1
        fi
        sos run {params.pipeline_dir}/mnm_postprocessing.ipynb \\
            export_top_loci {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --tsv_path $TSV_FILES \\
            --pip_thres {params.pip_thres} \\
            --lfsr_thres {params.lfsr_thres} \\
            --cs_size {params.cs_size} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 1.3 — PIP landscape plots
# ------------------------------------
# Generates lollipop-style PIP plots showing the credible-set
# variant landscape across each fine-mapped region.
rule pip_landscape_plots:
    """Generate PIP landscape plots for fine-mapped credible sets."""
    input:
        done = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
    output:
        done = "{cwd}/postprocessing/{theme}/pip_landscape/.done_pip_landscape",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        tsv_dir      = "{cwd}/postprocessing/{theme}/susie_tsv",
        outdir       = "{cwd}/postprocessing/{theme}/pip_landscape",
        name         = "{theme}",
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        PLOT_FILES=$(ls {params.tsv_dir}/*.tsv.gz 2>/dev/null | tr '\\n' ' ')
        if [ -z "$PLOT_FILES" ]; then
            echo "ERROR: No .tsv.gz files found in {params.tsv_dir}" >&2
            exit 1
        fi
        sos run {params.pipeline_dir}/mnm_postprocessing.ipynb \\
            susie_pip_landscape_plot {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --plot_list $PLOT_FILES \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 1.4 — UpSetR credible-set overlap plots
# ------------------------------------
# Visualises overlaps between credible sets across loci using
# UpSetR, useful for identifying shared fine-mapped variants.
rule upsetr_cs_plots:
    """Generate UpSetR plots showing credible-set overlaps."""
    input:
        done = "{cwd}/postprocessing/{theme}/susie_tsv/.done_tsv",
    output:
        done = "{cwd}/postprocessing/{theme}/upsetr_plots/.done_upsetr",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["susie"],
        tsv_dir      = "{cwd}/postprocessing/{theme}/susie_tsv",
        outdir       = "{cwd}/postprocessing/{theme}/upsetr_plots",
        name         = "{theme}",
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["default"]["threads"]
    resources:
        mem_mb  = config["resources"]["default"]["mem_mb"],
        runtime = config["resources"]["default"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        PLOT_FILES=$(ls {params.tsv_dir}/*.tsv.gz 2>/dev/null | tr '\\n' ' ')
        if [ -z "$PLOT_FILES" ]; then
            echo "ERROR: No .tsv.gz files found in {params.tsv_dir}" >&2
            exit 1
        fi
        sos run {params.pipeline_dir}/mnm_postprocessing.ipynb \\
            susie_upsetR_cs_plot {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --plot_list $PLOT_FILES \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """
