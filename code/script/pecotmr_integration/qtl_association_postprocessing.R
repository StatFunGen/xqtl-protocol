#!/usr/bin/env Rscript
# qtl_association_postprocessing.R
#
# Hierarchical multiple-testing correction of cis-QTL association results
# (TensorQTL), via pecotmr::qtlAssociationPostprocess. This is the thin wrapper
# that replaces the old notebook's `source(pecotmr/inst/code/tensorqtl_postprocessor.R)`
# pattern: it does the file I/O (read the per-gene regional + per-variant pairs,
# p-value pre-filter, join the filtered variant counts), assembles a per-gene
# QtlSumStats (one ROW per gene; each ENTRY the gene's variants), calls the
# pecotmr correction engine, and writes the consolidated RDS + the per-method
# regional / significant-QTL / significant-event / summary tables. All the
# multiple-testing statistics live in pecotmr (p.adjust / qvalue / qbeta).
#
# Inputs (one chromosome / batch per call):
#   --regional <tsv.gz>   per-gene TensorQTL regional summary. Needs
#                         molecular_trait_object_id, n_variants, beta_shape1,
#                         beta_shape2, p_beta (genetic-effect flavour).
#   --pairs <tsv.gz>      per-variant TensorQTL cis pairs. Needs
#                         molecular_trait_object_id, variant_id, chrom, pos,
#                         <pvalue-col>, <af-col>, tss_distance, tes_distance,
#                         and (optionally) qvalue.
#   --n-variants-stats <tsv.gz>  optional per-gene n_variants_filtered (else the
#                         filtered Bonferroni flavour is skipped).
# Params mirror the legacy CLI: --maf-cutoff / --cis-window / --fdr-threshold /
#   --pvalue-cutoff / --pvalue-col / --af-col / --study / --context / --genome.
# Outputs:
#   --output <RDS>        the enriched QtlSumStats (consolidated correction object).
#   --output-dir <dir>    optional: flat per-method tables (regional / significant
#                         QTL / significant events / summary).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(readr)
  library(GenomicRanges)
  library(IRanges)
  library(S4Vectors)
})

p <- arg_parser("cis-QTL association postprocessing via pecotmr::qtlAssociationPostprocess")
p <- add_argument(p, "--regional", help = "per-gene regional tsv(.gz)", type = "character", default = NA)
p <- add_argument(p, "--pairs", help = "per-variant pairs tsv(.gz)", type = "character", default = NA)
p <- add_argument(p, "--n-variants-stats", help = "per-gene n_variants_filtered tsv(.gz)",
                  type = "character", default = NA)
# Legacy CLI: glob the inputs from a work dir by (regex) pattern instead of
# passing explicit paths.
p <- add_argument(p, "--cwd", help = "work dir to glob inputs from", type = "character", default = NA)
p <- add_argument(p, "--regional-pattern", help = "regex for the regional file in --cwd", type = "character", default = NA)
p <- add_argument(p, "--qtl-pattern", help = "regex for the pairs file in --cwd", type = "character", default = NA)
p <- add_argument(p, "--n-variants-suffix", help = "regex for the n_variants stats file in --cwd", type = "character", default = NA)
p <- add_argument(p, "--maf-cutoff", help = "MAF cutoff (filtered Bonferroni)", type = "numeric", default = 0)
p <- add_argument(p, "--cis-window", help = "cis-window bp (filtered Bonferroni)", type = "numeric", default = 0)
p <- add_argument(p, "--fdr-threshold", help = "FDR threshold", type = "numeric", default = 0.05)
p <- add_argument(p, "--pvalue-cutoff", help = "pre-filter pairs to p < cutoff", type = "numeric", default = 1)
p <- add_argument(p, "--pvalue-col", help = "pairs p-value column", type = "character", default = "pvalue")
p <- add_argument(p, "--af-col", help = "pairs allele-frequency column", type = "character", default = "af")
p <- add_argument(p, "--study", help = "study label", type = "character", default = "study")
p <- add_argument(p, "--context", help = "context label", type = "character", default = "context")
p <- add_argument(p, "--genome", help = "genome build", type = "character", default = "hg38")
p <- add_argument(p, "--output", help = "output QtlSumStats RDS", type = "character")
p <- add_argument(p, "--output-dir", help = "optional dir for flat per-method tables",
                  type = "character", default = NA)
argv <- parse_args(p)

idCol <- "molecular_trait_object_id"
# Resolve each input: an explicit --path, or the first match of a regex pattern
# in --cwd (the legacy glob CLI).
resolveOne <- function(explicit, cwd, pattern, what) {
  if (!is.na(explicit)) return(explicit)
  if (is.na(cwd) || is.na(pattern)) return(NA_character_)
  hits <- list.files(cwd, pattern = pattern, full.names = TRUE)
  if (length(hits) == 0L) stop("No ", what, " file matching '", pattern, "' in ", cwd)
  hits[1L]
}
regionalFile <- resolveOne(argv$regional, argv$cwd, argv$regional_pattern, "regional")
pairsFile    <- resolveOne(argv$pairs, argv$cwd, argv$qtl_pattern, "pairs")
nvFile       <- resolveOne(argv$n_variants_stats, argv$cwd, argv$n_variants_suffix, "n_variants")
if (is.na(regionalFile) || is.na(pairsFile))
  stop("Provide --regional/--pairs, or --cwd with --regional-pattern/--qtl-pattern.")
regional <- readr::read_tsv(regionalFile, show_col_types = FALSE, progress = FALSE)
pairs    <- readr::read_tsv(pairsFile, show_col_types = FALSE, progress = FALSE)
if (!(idCol %in% names(regional)) || !("p_beta" %in% names(regional)))
  stop("--regional must carry ", idCol, " and p_beta.")
if (!(idCol %in% names(pairs)) || !(argv$pvalue_col %in% names(pairs)))
  stop("--pairs must carry ", idCol, " and the p-value column '", argv$pvalue_col, "'.")

# p-value pre-filter (keeps the object small; the per-gene min variant is always
# retained so the Bonferroni min is exact).
if (argv$pvalue_cutoff < 1)
  pairs <- pairs[!is.na(pairs[[argv$pvalue_col]]) & pairs[[argv$pvalue_col]] < argv$pvalue_cutoff, ]

nvFilt <- NULL
if (!is.na(nvFile)) {
  nvs <- readr::read_tsv(nvFile, show_col_types = FALSE, progress = FALSE)
  nvFilt <- setNames(nvs$n_variants_filtered, nvs[[idCol]])
}

genes <- as.character(regional[[idCol]])
pairsByGene <- split(pairs, factor(pairs[[idCol]], levels = genes))

mkEntry <- function(g) {
  d <- pairsByGene[[g]]
  if (is.null(d) || nrow(d) == 0L) return(GenomicRanges::GRanges())
  chrom <- paste0("chr", sub("^chr", "", as.character(d$chrom)))
  gr <- GenomicRanges::GRanges(chrom, IRanges::IRanges(as.integer(d$pos), width = 1L))
  mc <- S4Vectors::DataFrame(
    SNP = as.character(d$variant_id),
    P   = as.numeric(d[[argv$pvalue_col]]),
    af  = as.numeric(d[[argv$af_col]]),
    tss_distance = as.numeric(d$tss_distance),
    tes_distance = as.numeric(d$tes_distance))
  if ("qvalue" %in% names(d)) mc$qvalue <- as.numeric(d$qvalue)
  S4Vectors::mcols(gr) <- mc
  gr
}
entries <- lapply(genes, mkEntry)

nVar     <- as.numeric(regional$n_variants)
nVarFcol <- if (!is.null(nvFilt)) as.numeric(nvFilt[genes]) else NULL

qssArgs <- list(
  study = rep(argv$study, length(genes)), context = rep(argv$context, length(genes)),
  trait = genes, entry = entries, genome = argv$genome,
  n_variants = nVar, p_beta = as.numeric(regional$p_beta),
  beta_shape1 = as.numeric(regional$beta_shape1),
  beta_shape2 = as.numeric(regional$beta_shape2))
if (!is.null(nVarFcol)) qssArgs$n_variants_filtered <- nVarFcol
qss <- do.call(QtlSumStats, qssArgs)

filtering <- (argv$maf_cutoff > 0 || argv$cis_window > 0) && !is.null(nVarFcol)
r <- qtlAssociationPostprocess(
  qss, fdrThreshold = argv$fdr_threshold,
  mafCutoff = if (filtering) argv$maf_cutoff else 0,
  cisWindow = if (filtering) argv$cis_window else 0,
  methods = c("permutation", "bonferroni"))

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(r, argv$output)
cat(sprintf("Wrote enriched QtlSumStats (%d genes) to %s\n", nrow(r), argv$output))

# ---- optional flat per-method exports ----
if (!is.na(argv$output_dir)) {
  od <- argv$output_dir
  dir.create(od, showWarnings = FALSE, recursive = TRUE)
  base <- sub("\\.[^.]*$", "", basename(argv$output))

  # Enriched regional = input regional + the correction columns (row order kept).
  newCols <- setdiff(names(r), c(names(regional), "study", "context", "trait",
                                 "entry", "varY", "traitPos"))
  regionalOut <- cbind(as.data.frame(regional),
                       as.data.frame(S4Vectors::as.data.frame(r[, newCols, drop = FALSE])))
  readr::write_tsv(regionalOut, file.path(od, paste0(base, ".cis_regional.fdr.tsv.gz")))

  # Per-method EVENT-significance column (which genes pass): permutation and the
  # q-value SNP method gate events on the Storey q of the beta-approx permutation
  # p (q_beta); Bonferroni gates on its BH-FDR of the per-gene min. This matches
  # the variant-level rules getSignificantQtls() applies.
  flavours <- c(permutation = "q_beta",
                bonferroni_original = "fdr_bonferroni_min_original",
                bonferroni_filtered = "fdr_bonferroni_min_filtered",
                qvalue = "q_beta")
  summary_rows <- list()
  for (m in names(flavours)) {
    fcol <- flavours[[m]]
    if (is.null(r[[fcol]])) next
    # significant events (per-gene) at the FDR threshold
    keep <- !is.na(r[[fcol]]) & as.numeric(r[[fcol]]) < argv$fdr_threshold
    events <- regionalOut[keep, , drop = FALSE]
    if (nrow(events) > 0)
      readr::write_tsv(events, file.path(od, paste0(base, ".significant_events.", m, ".tsv.gz")))
    # significant variant-level QTL (derived, never stored)
    sig <- tryCatch(getSignificantQtls(r, m, threshold = argv$fdr_threshold),
                    error = function(e) NULL)
    nqtl <- 0L
    if (!is.null(sig) && length(sig) > 0) {
      sdf <- as.data.frame(sig)
      readr::write_tsv(sdf, file.path(od, paste0(base, ".significant_qtl.", m, ".tsv.gz")))
      nqtl <- length(sig)
    }
    summary_rows[[m]] <- data.frame(method = m, significant_events = sum(keep),
                                    significant_qtl = nqtl)
  }
  if (length(summary_rows) > 0)
    readr::write_tsv(do.call(rbind, summary_rows),
                     file.path(od, paste0(base, ".summary.tsv")))
  cat(sprintf("Wrote flat per-method tables to %s\n", od))
}
