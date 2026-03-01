#!/usr/bin/env Rscript
# ============================================================
# RNA_calling.R
# Mirrors: pipeline/RNA_calling.ipynb R-specific steps
#
# Steps (selected via --step):
#   aggregate_picard_qc — aggregate per-sample Picard alignment,
#                         RNA, and duplicate metrics into one TSV
#
# Usage:
#   Rscript RNA_calling.R --step aggregate_picard_qc \
#       --input-dir  path/to/picard_metrics/ \
#       --output     name.picard.aggregated_quality.metrics.tsv \
#       --is-paired-end 1 \
#       [--wasp]
# ============================================================

suppressPackageStartupMessages(library(optparse))

opt_list <- list(
  make_option("--step",          type = "character", default = NULL,
              help = "Step to run: aggregate_picard_qc"),
  make_option("--input-dir",     type = "character", default = NULL,
              help = "Directory containing per-sample Picard metrics files"),
  make_option("--output",        type = "character", default = NULL,
              help = "Output TSV file path"),
  make_option("--is-paired-end", type = "integer",   default = 1,
              help = "1 for paired-end (reads 2 rows), 0 for single-end (reads 1 row)"),
  make_option("--wasp",          action = "store_true", default = FALSE,
              help = "Flag: WASP mode was used (affects pattern for duplicate-metrics files)"),
  make_option("--numThreads",    type = "integer",   default = 8)
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))        stop("--step is required")
if (is.null(opt[["input-dir"]])) stop("--input-dir is required")
if (is.null(opt$output))      stop("--output is required")

source_dir   <- opt[["input-dir"]]
is_paired    <- opt[["is-paired-end"]]
wasp_mode    <- opt$wasp
wasp_suffix  <- if (wasp_mode) "_wasp" else "_nowasp"
qc_suffix    <- if (wasp_mode) "_qc"   else "_noqc"


# ── Helper readers ────────────────────────────────────────────────────────────

readPicard.alignment_summary_metrics <- function(source_path) {
  stopifnot(length(source_path) == 1, file.exists(source_path))
  is_dir <- file.info(source_path)$isdir

  if (is_dir) {
    files   <- system(paste("find -L", source_path,
                            "-name '*.alignment_summary_metrics'"), intern = TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".alignment_summary_metrics", "", basename(files), fixed = TRUE)
  } else {
    files   <- source_path
    samples <- gsub(".alignment_summary_metrics", "", basename(files), fixed = TRUE)
  }

  metrics <- list()
  for (i in seq_along(files)) {
    m <- read.table(files[i], header = TRUE, sep = "\t", comment.char = "#",
                    stringsAsFactors = FALSE, nrows = is_paired + 1)
    metrics[[i]] <- data.frame(
      Sample               = samples[i],
      File                 = files[i],
      PF_READS             = sum(m$PF_READS[1:2]),
      PF_READS_ALIGNED     = sum(m$PF_READS_ALIGNED[1:2]),
      PCT_PF_READS_ALIGNED = sum(m$PF_READS_ALIGNED[1:2]) / sum(m$PF_READS[1:2]),
      stringsAsFactors     = FALSE
    )
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  metrics
}


readPicard.rna_metrics <- function(source_path) {
  stopifnot(length(source_path) == 1, file.exists(source_path))
  is_dir <- file.info(source_path)$isdir

  if (is_dir) {
    files   <- system(paste("find -L", source_path, "-name '*.rna_metrics'"), intern = TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".rna_metrics", "", basename(files), fixed = TRUE)
  } else {
    files   <- source_path
    samples <- gsub(".rna_metrics", "", basename(files), fixed = TRUE)
  }

  metrics <- list()
  for (i in seq_along(files)) {
    m <- read.table(files[i], header = TRUE, sep = "\t", comment.char = "#",
                    stringsAsFactors = FALSE, nrows = 1)
    metrics[[i]] <- data.frame(
      Sample                    = samples[i],
      File                      = files[i],
      PCT_RIBOSOMAL_BASES       = m$PCT_RIBOSOMAL_BASES,
      PCT_CODING_BASES          = m$PCT_CODING_BASES,
      PCT_UTR_BASES             = m$PCT_UTR_BASES,
      PCT_INTRONIC_BASES        = m$PCT_INTRONIC_BASES,
      PCT_INTERGENIC_BASES      = m$PCT_INTERGENIC_BASES,
      PCT_MRNA_BASES            = m$PCT_MRNA_BASES,
      PCT_USABLE_BASES          = m$PCT_USABLE_BASES,
      MEDIAN_CV_COVERAGE        = m$MEDIAN_CV_COVERAGE,
      MEDIAN_5PRIME_BIAS        = m$MEDIAN_5PRIME_BIAS,
      MEDIAN_3PRIME_BIAS        = m$MEDIAN_3PRIME_BIAS,
      MEDIAN_5PRIME_TO_3PRIME_BIAS = m$MEDIAN_5PRIME_TO_3PRIME_BIAS,
      stringsAsFactors          = FALSE
    )
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  metrics
}


readPicard.duplicate_metrics <- function(source_path, wasp_sfx, qc_sfx) {
  stopifnot(length(source_path) == 1, file.exists(source_path))

  pattern   <- paste0("*.Aligned.sortedByCoord.out", wasp_sfx, qc_sfx, ".md.metrics")
  substitute_str <- paste0(".Aligned.sortedByCoord.out", wasp_sfx, qc_sfx, ".md.metrics")
  is_dir    <- file.info(source_path)$isdir

  if (is_dir) {
    files   <- system(paste("find -L", source_path, "-name", pattern), intern = TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(substitute_str, "", basename(files), fixed = TRUE)
  } else {
    files   <- source_path
    samples <- gsub(substitute_str, "", basename(files), fixed = TRUE)
  }

  metrics <- list()
  for (i in seq_along(files)) {
    m <- read.table(files[i], header = TRUE, sep = "\t", comment.char = "#",
                    stringsAsFactors = FALSE, nrows = 1)
    metrics[[i]] <- data.frame(
      Sample                   = samples[i],
      File                     = files[i],
      PERCENT_DUPLICATION      = m$PERCENT_DUPLICATION,
      ESTIMATED_LIBRARY_SIZE   = m$ESTIMATED_LIBRARY_SIZE,
      stringsAsFactors         = FALSE
    )
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  metrics
}


readPicard <- function(source_path, wasp_sfx, qc_sfx) {
  metrics_aln <- readPicard.alignment_summary_metrics(source_path)
  metrics_rna <- readPicard.rna_metrics(source_path)
  metrics_dup <- readPicard.duplicate_metrics(source_path, wasp_sfx, qc_sfx)

  stopifnot(
    all(row.names(metrics_aln) %in% row.names(metrics_rna)),
    all(row.names(metrics_rna) %in% row.names(metrics_dup)),
    all(row.names(metrics_dup) %in% row.names(metrics_aln))
  )

  metrics_aln$File <- NULL
  metrics_rna$File <- NULL
  metrics_dup$File <- NULL
  metrics_rna$Sample <- NULL
  metrics_dup$Sample <- NULL

  metrics <- cbind(metrics_aln, metrics_rna[row.names(metrics_aln), ])
  metrics <- cbind(metrics,     metrics_dup[row.names(metrics_aln), ])
  metrics
}


# ── Step: aggregate_picard_qc ─────────────────────────────────────────────────

if (opt$step == "aggregate_picard_qc") {
  picard_metrics <- readPicard(source_dir, wasp_suffix, qc_suffix)
  write.table(picard_metrics, file = opt$output,
              col.names = TRUE, row.names = FALSE, quote = FALSE)
} else {
  stop(paste0("Unknown step: '", opt$step, "'. Valid steps: aggregate_picard_qc"))
}
