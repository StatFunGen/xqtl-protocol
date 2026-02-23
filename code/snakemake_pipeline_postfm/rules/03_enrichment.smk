# ============================================================
# Rule Module 03: Enrichment Analyses
# ============================================================
# Three complementary enrichment analyses on fine-mapped results:
#
#   3.1  EOO Annotation Enrichment   — tests overlap of credible-set
#        variants with genomic annotations via LOCO block jackknife
#   3.2  Pathway Enrichment (GSEA)   — KEGG pathway enrichment for
#        genes harboring fine-mapped QTLs
#   3.3  Stratified LD Score Regression (S-LDSC) — heritability
#        enrichment of annotations using GWAS summary statistics
#        (three-step: annotation LD scores → per-trait h² → meta)
#
# SoS notebooks called:
#   - pipeline/eoo_enrichment.ipynb   (enrichment)
#   - pipeline/gsea.ipynb             (pathway_analysis)
#   - pipeline/sldsc_enrichment.ipynb (make_annotation_files_ldscore,
#                                      get_heritability, processed_stats)
#
# EOO and GSEA run per-theme, using the top-loci output from
# rule export_top_loci as their variant / gene inputs.
# S-LDSC runs per-annotation (annotation_name wildcard) and is
# GWAS-dependent (requires config["gwas"]["enabled"] = true).
# ============================================================


# ------------------------------------
# Step 3.1 — EOO Annotation Enrichment
# ------------------------------------
# Tests whether fine-mapped credible-set variants are enriched
# in genomic annotations (e.g., open chromatin, histone marks)
# using an excess-of-overlap (EOO) block-jackknife approach.
#
# Inputs consumed from config["enrichment"]["annotation"]:
#   significant_variants_path  — TSV/TXT with chr, pos columns
#                                (defaults to top_loci output if absent)
#   baseline_anno_path         — TSV with CHR, BP, binary annotation cols
rule annotation_enrichment:
    """Test credible-set variant enrichment in genomic annotations (EOO)."""
    input:
        top_loci_done = "{cwd}/postprocessing/{theme}/top_loci/.done_top_loci",
    output:
        done = "{cwd}/enrichment/{theme}/annotation/.done_annotation_enrichment",
    params:
        pipeline_dir    = config["pipeline_dir"],
        container       = config["containers"]["bioinfo"],
        top_loci_dir    = "{cwd}/postprocessing/{theme}/top_loci",
        outdir          = "{cwd}/enrichment/{theme}/annotation",
        name            = "{theme}",
        baseline_anno   = config["enrichment"]["annotation"]["baseline_anno_path"],
        dry_run         = DRY_RUN_SOS,
    threads: config["resources"]["enrichment"]["threads"]
    resources:
        mem_mb  = config["resources"]["enrichment"]["mem_mb"],
        runtime = config["resources"]["enrichment"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        # Use top-loci file as the significant-variants input
        VARIANTS_FILE=$(ls {params.top_loci_dir}/*.tsv.gz 2>/dev/null | head -1)
        if [ -z "$VARIANTS_FILE" ]; then
            echo "ERROR: No top-loci TSV found in {params.top_loci_dir}" >&2
            exit 1
        fi
        sos run {params.pipeline_dir}/eoo_enrichment.ipynb \\
            enrichment {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --significant_variants_path $VARIANTS_FILE \\
            --baseline_anno_path {params.baseline_anno} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 3.2 — Pathway Enrichment (GSEA / KEGG)
# ------------------------------------
# Tests whether genes harbouring fine-mapped QTLs are over-
# represented in KEGG biological pathways.
#
# Input gene list is derived from the top-loci export; a TSV
# with columns [group, gene_id] must be provided via
# config["enrichment"]["pathway"]["genes_file"]
# (or generated from the top-loci output).
rule pathway_enrichment:
    """Run KEGG pathway enrichment on genes with fine-mapped QTLs (GSEA)."""
    input:
        top_loci_done = "{cwd}/postprocessing/{theme}/top_loci/.done_top_loci",
    output:
        done = "{cwd}/enrichment/{theme}/pathway/.done_pathway_enrichment",
    params:
        pipeline_dir = config["pipeline_dir"],
        container    = config["containers"]["bioinfo"],
        top_loci_dir = "{cwd}/postprocessing/{theme}/top_loci",
        outdir       = "{cwd}/enrichment/{theme}/pathway",
        name         = "{theme}",
        # genes_file: user-supplied TSV [group, gene_id]; if empty the rule
        # falls back to deriving it from the top-loci directory.
        genes_file   = lambda wc: (
            config["enrichment"]["pathway"].get("genes_file", "") or
            f"{wc.cwd}/postprocessing/{wc.theme}/top_loci/{wc.theme}.genes.tsv"
        ),
        organism     = config["enrichment"]["pathway"].get("organism", "hsa"),
        dry_run      = DRY_RUN_SOS,
    threads: config["resources"]["enrichment"]["threads"]
    resources:
        mem_mb  = config["resources"]["enrichment"]["mem_mb"],
        runtime = config["resources"]["enrichment"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/gsea.ipynb \\
            pathway_analysis {params.dry_run} \\
            --cwd {params.outdir} \\
            --name {params.name} \\
            --genes_file {params.genes_file} \\
            --organism {params.organism} \\
            --container {params.container} \\
            --numThreads {threads}
        touch {output.done}
        """


# ============================================================
# Steps 3.3a–3.3c — Stratified LD Score Regression (S-LDSC)
# ============================================================
# Three-step workflow:
#   3.3a make_annotation_files_ldscore  — build per-chrom .annot.gz
#        and .ldscore.parquet files for the annotation of interest
#   3.3b get_heritability               — estimate per-GWAS-trait
#        heritability enrichment using the annotation LD scores
#   3.3c sldsc_meta                     — meta-analyse enrichment
#        estimates across all GWAS traits
#
# The {annotation_name} wildcard comes from
# config["enrichment"]["sldsc"]["annotation_names"] (list).
# All S-LDSC steps require config["gwas"]["enabled"] = true.
# ============================================================

# ------------------------------------
# Step 3.3a — Build annotation LD scores
# ------------------------------------
rule sldsc_annotation:
    """Compute per-chromosome annotation files and LD scores for S-LDSC."""
    output:
        done = "{cwd}/enrichment/sldsc/{annotation_name}/.done_annotation",
    params:
        pipeline_dir      = config["pipeline_dir"],
        container         = config["containers"]["bioinfo"],
        outdir            = "{cwd}/enrichment/sldsc/{annotation_name}",
        annotation_name   = "{annotation_name}",
        annotation_file   = config["enrichment"]["sldsc"]["annotation_file"],
        reference_anno    = config["enrichment"]["sldsc"]["reference_anno_file"],
        genome_ref        = config["enrichment"]["sldsc"]["genome_ref_file"],
        polyfun_path      = config["enrichment"]["sldsc"]["polyfun_path"],
        dry_run           = DRY_RUN_SOS,
    threads: config["resources"]["enrichment"]["threads"]
    resources:
        mem_mb  = config["resources"]["enrichment"]["mem_mb"],
        runtime = config["resources"]["enrichment"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/sldsc_enrichment.ipynb \\
            make_annotation_files_ldscore {params.dry_run} \\
            --cwd {params.outdir} \\
            --annotation_name {params.annotation_name} \\
            --annotation_file {params.annotation_file} \\
            --reference_anno_file {params.reference_anno} \\
            --genome_ref_file {params.genome_ref} \\
            --polyfun_path {params.polyfun_path} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 3.3b — Per-trait heritability enrichment
# ------------------------------------
rule sldsc_heritability:
    """Estimate heritability enrichment for each GWAS trait (S-LDSC step 2)."""
    input:
        annot_done = "{cwd}/enrichment/sldsc/{annotation_name}/.done_annotation",
    output:
        done = "{cwd}/enrichment/sldsc/{annotation_name}/heritability/.done_h2",
    params:
        pipeline_dir    = config["pipeline_dir"],
        container       = config["containers"]["bioinfo"],
        target_anno_dir = "{cwd}/enrichment/sldsc/{annotation_name}",
        outdir          = "{cwd}/enrichment/sldsc/{annotation_name}/heritability",
        annotation_name = "{annotation_name}",
        sumstat_dir     = config["enrichment"]["sldsc"]["sumstat_dir"],
        all_traits_file = config["enrichment"]["sldsc"]["all_traits_file"],
        baseline_ld_dir = config["enrichment"]["sldsc"]["baseline_ld_dir"],
        weights_dir     = config["enrichment"]["sldsc"]["weights_dir"],
        polyfun_path    = config["enrichment"]["sldsc"]["polyfun_path"],
        dry_run         = DRY_RUN_SOS,
    threads: config["resources"]["enrichment"]["threads"]
    resources:
        mem_mb  = config["resources"]["enrichment"]["mem_mb"],
        runtime = config["resources"]["enrichment"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/sldsc_enrichment.ipynb \\
            get_heritability {params.dry_run} \\
            --cwd {params.outdir} \\
            --annotation_name {params.annotation_name} \\
            --target_anno_dir {params.target_anno_dir} \\
            --sumstat_dir {params.sumstat_dir} \\
            --all_traits_file {params.all_traits_file} \\
            --baseline_ld_dir {params.baseline_ld_dir} \\
            --weights_dir {params.weights_dir} \\
            --polyfun_path {params.polyfun_path} \\
            --numThreads {threads}
        touch {output.done}
        """


# ------------------------------------
# Step 3.3c — Meta-analyse S-LDSC results
# ------------------------------------
rule sldsc_meta:
    """Meta-analyse S-LDSC heritability enrichment across GWAS traits."""
    input:
        h2_done = "{cwd}/enrichment/sldsc/{annotation_name}/heritability/.done_h2",
    output:
        done = "{cwd}/enrichment/sldsc/{annotation_name}/meta/.done_sldsc_meta",
    params:
        pipeline_dir    = config["pipeline_dir"],
        container       = config["containers"]["bioinfo"],
        h2_dir          = "{cwd}/enrichment/sldsc/{annotation_name}/heritability",
        outdir          = "{cwd}/enrichment/sldsc/{annotation_name}/meta",
        annotation_name = "{annotation_name}",
        polyfun_path    = config["enrichment"]["sldsc"]["polyfun_path"],
        dry_run         = DRY_RUN_SOS,
    threads: config["resources"]["enrichment"]["threads"]
    resources:
        mem_mb  = config["resources"]["enrichment"]["mem_mb"],
        runtime = config["resources"]["enrichment"]["runtime"],
    shell:
        """
        mkdir -p {params.outdir}
        sos run {params.pipeline_dir}/sldsc_enrichment.ipynb \\
            processed_stats {params.dry_run} \\
            --cwd {params.outdir} \\
            --annotation_name {params.annotation_name} \\
            --target_anno_dir {params.h2_dir} \\
            --polyfun_path {params.polyfun_path} \\
            --numThreads {threads}
        touch {output.done}
        """
