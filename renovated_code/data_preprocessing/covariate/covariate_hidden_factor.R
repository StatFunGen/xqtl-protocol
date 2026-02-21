#!/usr/bin/env Rscript
# ============================================================
# covariate_hidden_factor.R
# Mirrors: code/data_preprocessing/covariate/covariate_hidden_factor.ipynb
#
# Steps (selected via --step):
#   Marchenko_PC — residualize phenotype, then apply Marchenko-Pastur PCA
#   PEER         — residualize phenotype, then run PEER factor analysis
#
# Internal sub-steps (run automatically):
#   *_1: compute residual phenotype by regressing out merged covariates
#   Marchenko_PC_2: apply Marchenko-Pastur law to choose # factors, run PCA
#   PEER_2: run PEER model fitting
#   PEER_3: extract PEER factors
#
# Flags are kept identical to the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(readr)
})

opt_list <- list(
  make_option("--step",                  type = "character", default = NULL),
  make_option("--cwd",                   type = "character", default = "output"),
  make_option("--phenoFile",             type = "character", default = NULL,
              help = "Input phenotype BED.gz file"),
  make_option("--covFile",               type = "character", default = NULL,
              help = "Merged covariate file (output of covariate_formatting.R)"),
  make_option("--N",                     type = "integer",   default = 0,
              help = "Number of hidden factors (0 = auto-determine)"),
  make_option("--mean-impute-missing",   action = "store_true", default = FALSE,
              help = "Mean-impute missing phenotype values before residualization"),
  # PEER-specific
  make_option("--iteration",             type = "integer",   default = 1000),
  make_option("--convergence-mode",      type = "character", default = "fast",
              help = "PEER convergence mode: fast, medium, slow"),
  make_option("--numThreads",            type = "integer",   default = 8)
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))     stop("--step is required")
if (is.null(opt$phenoFile)) stop("--phenoFile is required")
if (is.null(opt$covFile))   stop("--covFile is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

bname <- sub("\\.bed\\.gz$", "", basename(opt$phenoFile))

# ── Shared: compute residual phenotype ───────────────────────────────────────
compute_residuals <- function(opt) {
  cat("=== Sub-step 1: compute residuals ===\n")

  # Read phenotype BED
  pheno <- read_tsv(opt$phenoFile, col_types = cols(.default = "c"), show_col_types = FALSE)
  coord_cols <- colnames(pheno)[1:4]
  sample_cols <- colnames(pheno)[-(1:4)]
  mat <- as.matrix(pheno[, sample_cols])
  class(mat) <- "numeric"
  rownames(mat) <- pheno[[1]]   # use first column (chr:start:end:id) as rowID

  # Mean-impute if requested
  if (isTRUE(opt$`mean-impute-missing`)) {
    row_means <- rowMeans(mat, na.rm = TRUE)
    for (i in seq_len(nrow(mat))) {
      na_idx <- is.na(mat[i, ])
      if (any(na_idx)) mat[i, na_idx] <- row_means[i]
    }
  }

  # Read covariates (rows = covariates, cols = samples)
  cov_df <- read_tsv(opt$covFile, col_types = cols(.default = "d"),
                     show_col_types = FALSE)
  id_col <- colnames(cov_df)[1]
  cov_samples <- setdiff(colnames(cov_df), id_col)
  shared <- intersect(sample_cols, cov_samples)
  cat(sprintf("Shared samples for residualization: %d\n", length(shared)))

  mat_sub <- mat[, shared, drop = FALSE]
  cov_mat <- as.matrix(t(cov_df[, shared, drop = FALSE]))
  colnames(cov_mat) <- cov_df[[id_col]]

  # Regress out covariates row-wise
  cat("Regressing out covariates...\n")
  residuals <- t(apply(mat_sub, 1, function(y) {
    if (all(is.na(y))) return(y)
    fit <- tryCatch(lm.fit(cbind(1, cov_mat), y), error = function(e) NULL)
    if (is.null(fit)) return(y)
    resid(fit)
  }))
  colnames(residuals) <- shared

  # Reconstruct BED with residuals
  resid_df <- bind_cols(pheno[, coord_cols], as.data.frame(residuals)[, shared])
  list(residuals = residuals, resid_df = resid_df,
       coord = pheno[, coord_cols], shared = shared)
}

# ── Step: Marchenko_PC ────────────────────────────────────────────────────────
run_marchenko <- function(opt) {
  res <- compute_residuals(opt)
  cat("=== Sub-step 2: Marchenko-Pastur PCA ===\n")

  suppressPackageStartupMessages(library(RMTstat))

  mat <- res$residuals
  n   <- ncol(mat)
  p   <- nrow(mat)

  # SVD
  sv  <- svd(mat / sqrt(n - 1), nu = 0)
  eig <- sv$d^2

  # Marchenko-Pastur upper edge
  gamma <- p / n
  lambda_plus <- (1 + sqrt(gamma))^2

  if (opt$N == 0) {
    # Auto-determine: keep components above the MP upper edge
    n_factors <- sum(eig > lambda_plus)
    cat(sprintf("Marchenko-Pastur threshold = %.4f, selecting %d factors\n",
                lambda_plus, n_factors))
    if (n_factors == 0) {
      cat("WARNING: No factors above MP threshold. Using 1.\n")
      n_factors <- 1L
    }
  } else {
    n_factors <- opt$N
    cat(sprintf("Using user-specified N = %d factors\n", n_factors))
  }

  # PCA with n_factors
  pca_res <- prcomp(t(mat), center = TRUE, scale. = FALSE, rank. = n_factors)
  factors <- as.data.frame(t(pca_res$x))  # factors × samples
  factors <- cbind(ID = rownames(factors), factors)

  # Write output
  out_file <- file.path(opt$cwd, paste0(bname, ".Marchenko_PC.gz"))
  write_tsv(factors, gzfile(out_file))
  cat(sprintf("Output: %s (%d factors × %d samples)\n",
              out_file, n_factors, ncol(mat)))
}

# ── Step: PEER ────────────────────────────────────────────────────────────────
run_peer <- function(opt) {
  res <- compute_residuals(opt)
  cat("=== Sub-step 2: PEER factor analysis ===\n")

  suppressPackageStartupMessages(library(peer))

  mat <- res$residuals
  n_samples  <- ncol(mat)
  n_features <- nrow(mat)

  n_factors <- if (opt$N == 0) {
    min(n_samples, 25L)   # PEER default heuristic
  } else {
    opt$N
  }
  cat(sprintf("Running PEER with %d factors, %d iterations\n",
              n_factors, opt$iteration))

  model <- PEER()
  PEER_setPhenoMean(model, t(mat))
  PEER_setNk(model, n_factors)
  PEER_setMaxIter(model, opt$iteration)
  PEER_update(model)

  # Save PEER model
  model_file <- file.path(opt$cwd, paste0(bname, ".PEER_MODEL.hd5"))
  # Note: PEER's HDF5 save is model-specific; save as RDS instead
  saveRDS(model, file.path(opt$cwd, paste0(bname, ".PEER_MODEL.rds")))

  cat("=== Sub-step 3: Extract PEER factors ===\n")
  factors_mat <- t(PEER_getX(model))   # factors × samples
  rownames(factors_mat) <- paste0("peer_factor_", seq_len(nrow(factors_mat)))
  colnames(factors_mat) <- colnames(mat)

  factors_df <- cbind(ID = rownames(factors_mat), as.data.frame(factors_mat))
  out_file   <- file.path(opt$cwd, paste0(bname, ".PEER.gz"))
  write_tsv(factors_df, gzfile(out_file))
  cat(sprintf("Output: %s (%d factors × %d samples)\n",
              out_file, nrow(factors_mat), ncol(mat)))
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  Marchenko_PC = run_marchenko(opt),
  PEER         = run_peer(opt),
  stop(sprintf("Unknown step '%s'. Available: Marchenko_PC, PEER", opt$step))
)
