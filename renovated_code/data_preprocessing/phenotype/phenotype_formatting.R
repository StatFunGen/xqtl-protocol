#!/usr/bin/env Rscript
# ============================================================
# phenotype_formatting.R
# Mirrors: code/data_preprocessing/phenotype/phenotype_formatting.ipynb
#
# Steps (selected via --step):
#   gct_extract_samples — filter samples from a GCT file by a keep-list
#
# Flags are kept identical to the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(readr)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
opt_list <- list(
  make_option("--step",         type = "character", default = NULL,
              help = "Step to run: gct_extract_samples"),
  make_option("--phenoFile",    type = "character", default = NULL,
              help = "Input GCT.gz file"),
  make_option("--cwd",          type = "character", default = "output",
              help = "Output directory [default: output]"),
  make_option("--name",         type = "character", default = "",
              help = "Output name prefix (default: derived from phenoFile)"),
  make_option("--keep-samples", type = "character", default = NULL,
              help = "[gct_extract_samples] File with one sample ID per line to keep"),
  make_option("--numThreads",   type = "integer",   default = 1,
              help = "Number of threads [default: 1]")
)

opt <- parse_args(OptionParser(option_list = opt_list))

if (is.null(opt$step)) stop("--step is required")
if (is.null(opt$phenoFile)) stop("--phenoFile is required")

# Derive output name from phenoFile if not provided
if (nchar(opt$name) == 0) {
  bname <- basename(opt$phenoFile)
  opt$name <- sub("\\.gct(\\.gz)?$", "", bname)
}

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Step: gct_extract_samples
# ---------------------------------------------------------------------------
gct_extract_samples <- function(opt) {
  if (is.null(opt[["keep-samples"]])) stop("--keep-samples is required")

  keep_ids <- readLines(opt[["keep-samples"]])
  keep_ids <- trimws(keep_ids[nchar(trimws(keep_ids)) > 0])
  cat(sprintf("Keeping %d samples\n", length(keep_ids)))

  # Read GCT header (first 2 lines)
  con <- if (grepl("\\.gz$", opt$phenoFile)) gzcon(file(opt$phenoFile, "rb")) else file(opt$phenoFile, "r")
  header_lines <- readLines(con, n = 2)
  close(con)

  # Read data (skip 2 header lines)
  dat <- read_tsv(opt$phenoFile, skip = 2, col_types = cols(.default = "c"), show_col_types = FALSE)

  # Identify fixed columns (Name, Description) and sample columns
  fixed_cols <- colnames(dat)[1:2]
  sample_cols <- setdiff(colnames(dat), fixed_cols)

  # Filter to requested samples
  available <- intersect(keep_ids, sample_cols)
  missing   <- setdiff(keep_ids, sample_cols)
  if (length(missing) > 0) {
    warning(sprintf("%d requested samples not found in GCT: %s",
                    length(missing), paste(head(missing, 5), collapse = ", ")))
  }
  cat(sprintf("Retaining %d of %d samples\n", length(available), length(sample_cols)))

  dat_filt <- dat[, c(fixed_cols, available)]

  # Reconstruct GCT format
  out_file <- file.path(opt$cwd, paste0(opt$name, ".filtered.gct.gz"))
  con_out  <- gzcon(file(out_file, "wb"))
  writeLines(header_lines[1], con_out)
  writeLines(sprintf("%d\t%d", nrow(dat_filt), length(available)), con_out)
  write_tsv(dat_filt, con_out)
  close(con_out)

  cat(sprintf("Written: %s\n", out_file))
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch(opt$step,
  gct_extract_samples = gct_extract_samples(opt),
  stop(sprintf("Unknown step: '%s'. Available: gct_extract_samples", opt$step))
)
