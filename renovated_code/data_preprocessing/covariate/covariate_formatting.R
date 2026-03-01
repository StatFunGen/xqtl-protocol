#!/usr/bin/env Rscript
# ============================================================
# covariate_formatting.R
# Mirrors: code/data_preprocessing/covariate/covariate_formatting.ipynb
#
# Steps (selected via --step):
#   merge_genotype_pc — merge projected PCA scores with fixed covariates
#
# Flags are kept identical to the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(readr)
})

opt_list <- list(
  make_option("--step",         type = "character", default = NULL),
  make_option("--cwd",          type = "character", default = "output"),
  make_option("--pcaFile",      type = "character", default = NULL,
              help = "RDS file from PCA.R project_samples (contains $pcs data frame)"),
  make_option("--covFile",      type = "character", default = NULL,
              help = "Fixed covariate file (gzip or plain TSV, samples as columns)"),
  make_option("--k",            type = "integer",   default = 20,
              help = "Number of genotype PCs to include"),
  make_option("--tol-cov",      type = "double",    default = 0.3,
              help = "Collinearity tolerance: max fraction of missing covariates per sample"),
  make_option("--mean-impute",  action = "store_true", default = FALSE,
              help = "Mean-impute missing covariate values"),
  make_option("--numThreads",   type = "integer",   default = 1),
  make_option("--dry-run",      action = "store_true", default = FALSE,
              help = "Print full command + validate inputs; do not run.")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))    stop("--step is required")
if (is.null(opt$pcaFile)) stop("--pcaFile is required")
if (is.null(opt$covFile)) stop("--covFile is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

# ── Step: merge_genotype_pc ──────────────────────────────────────────────────
merge_genotype_pc <- function(opt) {
  # ── Dry-run: print full command and validate inputs ──────────────────────
  if (isTRUE(opt$`dry-run`)) {
    script_path <- tryCatch(normalizePath(sys.frame(0)$filename), error = function(e) "covariate_formatting.R")
    cat("[DRY-RUN] covariate_formatting.R merge_genotype_pc — would execute:\n")
    cat(sprintf("  Rscript %s \\\n",              script_path))
    cat(sprintf("    --step %s \\\n",              opt$step))
    cat(sprintf("    --pcaFile %s \\\n",           opt$pcaFile))
    cat(sprintf("    --covFile %s \\\n",           opt$covFile))
    cat(sprintf("    --k %d \\\n",                 opt$k))
    cat(sprintf("    --tol-cov %.2f \\\n",         opt$`tol-cov`))
    cat(sprintf("    --cwd %s\n",                    opt$cwd))
    cat("\n[DRY-RUN] Input file check:\n")
    for (f in c(opt$pcaFile, opt$covFile)) {
      if (is.null(f) || is.na(f)) next
      status <- if (file.exists(f)) "\u2713" else "\u2717 NOT FOUND"
      cat(sprintf("  %s  %s\n", status, f))
    }
    quit(status = 0)
  }

  cat("=== merge_genotype_pc ===\n")

  # Load PCA output
  pca_obj <- readRDS(opt$pcaFile)
  pcs <- if (is.list(pca_obj) && "pcs" %in% names(pca_obj)) pca_obj$pcs else pca_obj
  k   <- min(opt$k, sum(grepl("^PC", colnames(pcs))))
  cat(sprintf("Using %d genotype PCs from: %s\n", k, opt$pcaFile))

  # Keep only top-k PCs, pivot to covariate matrix format:
  #   rows = covariates, cols = samples
  pc_cols <- paste0("PC", seq_len(k))
  id_col  <- if ("sample_id" %in% colnames(pcs)) "sample_id" else colnames(pcs)[1]
  pcs_sub <- pcs[, c(id_col, pc_cols), drop = FALSE]

  # Transpose: covariates × samples
  pc_t            <- t(as.matrix(pcs_sub[, pc_cols]))
  colnames(pc_t)  <- pcs_sub[[id_col]]
  rownames(pc_t)  <- pc_cols
  pc_df           <- as.data.frame(pc_t)
  pc_df           <- cbind(ID = rownames(pc_df), pc_df)

  # Load fixed covariates
  cov_df <- if (grepl("\\.gz$", opt$covFile)) {
    read_tsv(opt$covFile, col_types = cols(.default = "c"), show_col_types = FALSE)
  } else {
    read_tsv(opt$covFile, col_types = cols(.default = "c"), show_col_types = FALSE)
  }
  cat(sprintf("Fixed covariates: %d rows × %d cols\n", nrow(cov_df), ncol(cov_df)))

  # Align samples: intersect of PC samples and covariate samples
  # Assume first column of covFile is the covariate/row ID
  id_col_cov <- colnames(cov_df)[1]
  sample_cols_cov <- setdiff(colnames(cov_df), id_col_cov)
  sample_cols_pc  <- setdiff(colnames(pc_df), "ID")
  shared_samples  <- intersect(sample_cols_cov, sample_cols_pc)
  cat(sprintf("Shared samples: %d\n", length(shared_samples)))

  # Filter samples with too many missing covariates
  cov_mat <- cov_df[, shared_samples, drop = FALSE]
  miss_rate <- colMeans(is.na(cov_mat))
  keep_samples <- names(miss_rate[miss_rate <= opt$`tol-cov`])
  n_removed <- length(shared_samples) - length(keep_samples)
  if (n_removed > 0)
    cat(sprintf("Removing %d samples exceeding missing covariate tolerance (%.0f%%)\n",
                n_removed, opt$`tol-cov` * 100))

  # Mean imputation
  cov_sub <- cov_df[, c(id_col_cov, keep_samples), drop = FALSE]
  if (isTRUE(opt$`mean-impute`)) {
    cov_sub[, -1] <- lapply(cov_sub[, -1], function(x) {
      x <- as.numeric(x)
      x[is.na(x)] <- mean(x, na.rm = TRUE)
      x
    })
  }

  # Merge PCs on top of fixed covariates
  pc_sub  <- pc_df[, c("ID", keep_samples), drop = FALSE]
  colnames(pc_sub)[1]  <- id_col_cov
  merged  <- bind_rows(cov_sub, pc_sub)

  # Output filename derived from pcaFile basename
  bname    <- sub("\\.pca\\.projected\\.rds$", "", basename(opt$pcaFile))
  out_file <- file.path(opt$cwd, paste0(bname, ".pca.gz"))
  write_tsv(merged, out_file)   # readr detects .gz and compresses automatically
  cat(sprintf("Output: %s (%d covariates × %d samples)\n",
              out_file, nrow(merged), length(keep_samples)))
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  merge_genotype_pc = merge_genotype_pc(opt),
  stop(sprintf("Unknown step '%s'. Available: merge_genotype_pc", opt$step))
)
