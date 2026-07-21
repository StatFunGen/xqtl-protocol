#!/usr/bin/env Rscript
# legacy_twas_weights_to_s4.R
#
# One-shot converter: legacy `*.univariate_twas_weights.rds` (top-level
# gene IDs -> context names -> `twas_weights` + `twas_cv_result` etc.)
# into the new S4 `pecotmr::TwasWeights` collection.
#
# Usage:
#   Rscript legacy_twas_weights_to_s4.R \
#     --legacy <legacy.rds> \
#     --study <study_label> \
#     --ld-meta <ld_meta.tsv> \
#     --ld-block <chr:start-end>  (or --ld-prefix <prefix>) \
#     --output <out.rds>
#
# The LD-block / LD-prefix is required because TwasWeights carries an
# `ldSketch` slot that must match the GwasSumStats' LD sketch downstream.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Convert legacy TwasWeights RDS to S4 TwasWeights")
parser <- add_argument(parser, "--legacy",
                       help = "Path to legacy *.univariate_twas_weights.rds",
                       type = "character")
parser <- add_argument(parser, "--study",
                       help = "Study label to stamp on every entry",
                       type = "character")
parser <- add_argument(parser, "--ld-meta",
                       help = "Path to LD-meta TSV (used when --ld-block is given)",
                       type = "character", default = "")
parser <- add_argument(parser, "--ld-block",
                       help = "LD block as chr:start-end (resolved against --ld-meta)",
                       type = "character", default = "")
parser <- add_argument(parser, "--ld-prefix",
                       help = "Explicit PLINK2/PLINK1/GDS/VCF prefix; bypasses LD-meta lookup",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# --- Build an LD GenotypeHandle (same logic as gwas_sumstats_construct.R) ----
resolve_ld_prefix <- function(meta_path, block_str) {
  m <- regmatches(block_str, regexec("^([^:]+):([0-9]+)-([0-9]+)$", block_str))[[1L]]
  if (length(m) != 4L)
    stop("--ld-block must be chr:start-end (got: ", block_str, ")")
  block <- list(chr = m[[2L]], start = as.integer(m[[3L]]),
                end = as.integer(m[[4L]]))
  meta_dir <- dirname(normalizePath(meta_path))
  meta <- read.table(meta_path, header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE, check.names = FALSE,
                     comment.char = "")
  chr_col <- intersect(c("#chr", "#chrom", "chr", "chrom"), names(meta))[1L]
  if (is.na(chr_col))
    stop("Could not find a chromosome column in ", meta_path)
  chr_norm    <- sub("^chr", "", as.character(meta[[chr_col]]), ignore.case = TRUE)
  block_norm  <- sub("^chr", "", block$chr, ignore.case = TRUE)
  s <- suppressWarnings(as.integer(meta$start))
  e <- suppressWarnings(as.integer(meta$end))
  whole <- !is.na(s) & !is.na(e) & s == 0L & e == 0L
  hit <- which(chr_norm == block_norm &
               (whole | (s <= block$start & e >= block$end)))
  if (length(hit) != 1L)
    stop("LD-meta lookup for ", block_str, " returned ", length(hit), " rows.")
  pfx <- meta$path[hit]
  if (!startsWith(pfx, "/")) pfx <- file.path(meta_dir, pfx)
  pfx
}

open_handle <- function(prefix) {
  if (file.exists(paste0(prefix, ".pgen"))) GenotypeHandle(plink2Prefix = prefix)
  else if (file.exists(paste0(prefix, ".bed"))) GenotypeHandle(plink1Prefix = prefix)
  else if (file.exists(paste0(prefix, ".gds"))) GenotypeHandle(path = paste0(prefix, ".gds"))
  else if (file.exists(paste0(prefix, ".vcf.gz"))) GenotypeHandle(path = paste0(prefix, ".vcf.gz"))
  else stop("No recognised genotype payload at LD prefix: ", prefix)
}

ld_handle <- if (nzchar(argv$ld_prefix)) {
  open_handle(argv$ld_prefix)
} else if (nzchar(argv$ld_meta) && nzchar(argv$ld_block)) {
  open_handle(resolve_ld_prefix(argv$ld_meta, argv$ld_block))
} else {
  stop("Provide either --ld-prefix or (--ld-meta AND --ld-block).")
}

# --- Walk the legacy structure and build TwasWeightsEntry per (gene, context) ----
legacy <- readRDS(argv$legacy)
if (!is.list(legacy) || length(legacy) == 0L)
  stop("Legacy RDS is empty or not a list: ", argv$legacy)

studies  <- character(0)
contexts <- character(0)
traits   <- character(0)
methods  <- character(0)
entries  <- list()

for (gene in names(legacy)) {
  gene_obj <- legacy[[gene]]
  if (!is.list(gene_obj)) next
  for (ctx_name in names(gene_obj)) {
    ctx_obj <- gene_obj[[ctx_name]]
    if (!is.list(ctx_obj) ||
        is.null(ctx_obj$twas_weights) ||
        is.null(ctx_obj$twas_cv_result)) next
    wts_list <- ctx_obj$twas_weights
    if (length(wts_list) == 0L) next

    # Variant ids: rownames of the first matrix-shaped weight column. Some
    # legacy fields are length-N numeric vectors (e.g. `ensemble_weights`);
    # coerce them to single-column matrices below.
    first_mat <- Find(is.matrix, wts_list)
    if (is.null(first_mat))
      stop("Legacy weights for ", gene, "/", ctx_name,
           " contain no matrix-shaped entry; cannot recover variant IDs.")
    vids <- rownames(first_mat)
    method_names <- sub("_weights$", "", names(wts_list))

    # CV performance per method (subset of methods that actually had CV).
    perf_list <- ctx_obj$twas_cv_result$performance
    perf_method_names <- sub("_performance$", "", names(perf_list))
    cv_lookup <- setNames(perf_list, perf_method_names)

    # context_name follows the legacy convention "<context>_<trait>" — strip
    # the trailing trait suffix so the new schema's context is a clean label.
    context_label <- sub(paste0("_", gene, "$"), "", ctx_name)

    # Legacy susie_weights_intermediate carries the SuSiE posterior
    # tensor (mu, lbf_variable, X_column_scale_factors, etc.) that
    # ctwasPipeline's alpha renormalization needs when variants get
    # dropped downstream of the original fit. Attach it on susie-method
    # entries via the TwasWeightsEntry@fits slot.
    legacy_intermediate <- ctx_obj$susie_weights_intermediate

    # One TwasWeightsEntry per method (TwasWeights schema requires one row
    # per (study, context, trait, method) tuple).
    for (mi in seq_along(method_names)) {
      m  <- method_names[[mi]]
      wm <- wts_list[[mi]]
      wv <- if (is.matrix(wm)) as.numeric(wm[, 1L]) else as.numeric(wm)
      # Skip a method when every weight is zero or NA — it would never
      # contribute a TWAS-Z and downstream would silently drop it anyway.
      if (all(is.na(wv)) || all(wv == 0, na.rm = TRUE)) next

      weights_mat <- matrix(wv, ncol = 1L,
                            dimnames = list(vids, m))

      perf_row <- cv_lookup[[m]]
      cv_df <- if (!is.null(perf_row)) {
        df <- as.data.frame(perf_row, stringsAsFactors = FALSE)
        names(df)[names(df) == "adj_rsq"] <- "adjusted_rsq"
        df$method <- m
        df <- df[, intersect(c("method", "rsq", "adjusted_rsq", "pval",
                                "corr", "RMSE", "MAE"),
                              names(df)), drop = FALSE]
        rownames(df) <- NULL
        df
      } else {
        data.frame(method = m, rsq = NA_real_, adjusted_rsq = NA_real_,
                   pval = NA_real_, stringsAsFactors = FALSE)
      }

      # Attach SuSiE intermediate on susie-family methods so
      # ctwasPipeline's renormalization branch can recompute weights
      # over a filtered variant set.
      entry_fits <- if (m %in% c("susie", "susie_inf", "susie_ash") &&
                       !is.null(legacy_intermediate)) {
        legacy_intermediate
      } else {
        NULL
      }

      entry <- TwasWeightsEntry(
        variantIds    = vids,
        weights       = weights_mat,
        fits          = entry_fits,
        cvPerformance = cv_df,
        standardized  = FALSE)

      studies  <- c(studies, argv$study)
      contexts <- c(contexts, context_label)
      traits   <- c(traits, gene)
      methods  <- c(methods, m)
      entries  <- c(entries, list(entry))
    }
  }
}

if (length(entries) == 0L)
  stop("Converter produced 0 entries from ", argv$legacy)

tw <- TwasWeights(
  study    = studies,
  context  = contexts,
  trait    = traits,
  method   = methods,
  entry    = entries,
  ldSketch = ld_handle)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(tw, argv$output)
cat(sprintf("Wrote S4 TwasWeights (%d entries; %d unique traits) to %s\n",
            length(entries),
            length(unique(traits)),
            argv$output))
