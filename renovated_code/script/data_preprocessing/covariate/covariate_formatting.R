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
  library(data.table)
})

opt_list <- list(
  make_option("--step",         type = "character", default = NULL),
  make_option("--cwd",          type = "character", default = "output"),
  make_option("--pcaFile",      type = "character", default = NULL,
              help = "RDS file from PCA.R project_samples (contains $pcs data frame)"),
  make_option("--covFile",      type = "character", default = NULL,
              help = "Fixed covariate file (gzip or plain TSV, samples as columns)"),
  make_option("--name",         type = "character", default = NULL,
              help = "Output basename without the trailing .gz"),
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

strip_last_ext <- function(path) {
  sub("\\.[^.]+$", "", basename(path))
}

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
    if (!is.null(opt$name) && nzchar(opt$name)) {
      cat(sprintf("    --name %s \\\n",            opt$name))
    }
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

  compute_missing <- function(mtx) {
    sum(is.na(mtx)) / length(mtx)
  }

  mean_impute_mtx <- function(mtx) {
    f <- apply(mtx, 2, function(x) mean(x, na.rm = TRUE))
    for (i in seq_along(f)) {
      mtx[, i][which(is.na(mtx[, i]))] <- f[i]
    }
    mtx
  }

  filter_mtx <- function(X, missing_rate_thresh) {
    rm_col <- which(apply(X, 2, compute_missing) > missing_rate_thresh)
    if (length(rm_col)) {
      X <- X[, -rm_col, drop = FALSE]
    }
    if (isTRUE(opt$`mean-impute`)) {
      return(mean_impute_mtx(X))
    }
    X
  }

  pca_obj <- readRDS(opt$pcaFile)
  pca_output <- if (is.list(pca_obj) && "pc_scores" %in% names(pca_obj)) {
    pca_obj$pc_scores
  } else if (is.list(pca_obj) && "pcs" %in% names(pca_obj)) {
    pca_obj$pcs
  } else {
    pca_obj
  }

  pc_cols <- grep("^PC", colnames(pca_output), value = TRUE)
  k <- min(opt$k, length(pc_cols))
  mtx <- pca_output %>% select(all_of(pc_cols[seq_len(k)])) %>% t()
  colnames(mtx) <- if ("IID" %in% colnames(pca_output)) pca_output$IID else pca_output[[1]]
  mtx <- mtx[seq_len(k), , drop = FALSE]
  mtx <- mtx %>% as_tibble(rownames = "#id")
  pca_samples <- colnames(mtx)

  raw_cov <- fread(opt$covFile, head = TRUE, data.table = FALSE, check.names = FALSE)
  header_sample_hits <- sum(colnames(raw_cov)[-1] %in% colnames(mtx))
  row_sample_hits <- sum(raw_cov[[1]] %in% pca_samples)

  if (header_sample_hits >= row_sample_hits) {
    covt <- as_tibble(raw_cov)
    colnames(covt)[1] <- "#id"
  } else {
    covt <- transpose(as.data.table(raw_cov), keep.names = "#id", make.names = 1) %>% as_tibble()
  }

  overlap <- intersect(colnames(covt), colnames(mtx))
  print(paste(ncol(covt) - 1, "samples are in the covariate file", sep = " "))
  print(paste(ncol(mtx), "samples are in the PCA file", sep = " "))
  print(paste(length(overlap) - 1, "samples overlap between covariate & PCA files and are included in the analysis:", sep = " "))
  print(overlap[!overlap == "#id"])
  if (length(overlap) <= 1) {
    stop("No overlapping samples between covariate and PCA inputs")
  }

  cov_missing <- covt %>% select(-all_of(overlap))
  print(paste(ncol(cov_missing), "samples in the covariate file are missing from the PCA file:", sep = " "))
  print(colnames(cov_missing))

  pca_missing <- mtx %>% select(-all_of(overlap))
  print(paste(ncol(pca_missing), "samples in the PCA file are missing from the covariate file:", sep = " "))
  print(colnames(pca_missing))

  covt <- covt %>% select(all_of(overlap))
  mtx <- mtx %>% select(all_of(overlap))
  output <- bind_rows(covt, mtx)

  if (opt$`tol-cov` == -1 && sum(is.na(output)) > 0) {
    stop("NA in covariates input: Check input file or set parameter tol_cov to allow for removing missing values; mean_impute to allow for imputing missing values")
  }

  output <- output %>% as.data.frame
  rownames(output) <- output$`#id`
  output <- filter_mtx(output[, 2:ncol(output), drop = FALSE], opt$`tol-cov`) %>% as_tibble(rownames = "#id")
  output_name <- if (!is.null(opt$name) && nzchar(opt$name)) {
    opt$name
  } else {
    paste0(strip_last_ext(opt$covFile), ".", strip_last_ext(opt$pcaFile))
  }
  out_file <- file.path(opt$cwd, paste0(output_name, ".gz"))
  output %>% write_delim(out_file, "\t")
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  merge_genotype_pc = merge_genotype_pc(opt),
  stop(sprintf("Unknown step '%s'. Available: merge_genotype_pc", opt$step))
)
