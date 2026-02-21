#!/usr/bin/env Rscript
# ============================================================
# PCA.R
# Mirrors: code/data_preprocessing/genotype/PCA.ipynb
#
# Steps (selected via --step):
#   flashpca        — compute PCs from LD-pruned unrelated genotypes
#   project_samples — project related samples onto unrelated PCA space
#
# Flags are kept identical to the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
})

opt_list <- list(
  make_option("--step",        type = "character", default = NULL),
  make_option("--genoFile",    type = "character", default = NULL,
              help = "Input PLINK .bed file"),
  make_option("--cwd",         type = "character", default = "output"),
  make_option("--k",           type = "integer",   default = 20,
              help = "Number of PCs to compute"),
  make_option("--maha-k",      type = "integer",   default = 5,
              help = "Number of PCs for Mahalanobis outlier detection"),
  make_option("--prob",        type = "double",    default = 0.997,
              help = "Probability threshold for Mahalanobis outlier removal"),
  make_option("--pca-model",   type = "character", default = NULL,
              help = "[project_samples] RDS file from flashpca step"),
  make_option("--numThreads",  type = "integer",   default = 8)
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))     stop("--step is required")
if (is.null(opt$genoFile)) stop("--genoFile is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

bed_prefix <- sub("\\.bed$", "", opt$genoFile)
out_prefix <- file.path(opt$cwd,
    sub("\\.bed$", "", basename(opt$genoFile)))

# ── Mahalanobis-based outlier detection ──────────────────────────────────────
remove_outliers <- function(pcs, k, prob) {
  cat(sprintf("Outlier detection: using top %d PCs, prob = %g\n", k, prob))
  pc_mat <- as.matrix(pcs[, paste0("PC", seq_len(k))])
  cov_mat <- cov(pc_mat)
  center  <- colMeans(pc_mat)
  maha    <- mahalanobis(pc_mat, center, cov_mat)
  cutoff  <- qchisq(prob, df = k)
  keep    <- maha <= cutoff
  n_removed <- sum(!keep)
  if (n_removed > 0) {
    cat(sprintf("Removing %d outlier samples (Mahalanobis > %.1f)\n",
                n_removed, cutoff))
  } else {
    cat("No outliers detected\n")
  }
  pcs[keep, ]
}

# ── Step: flashpca ───────────────────────────────────────────────────────────
run_flashpca <- function(opt) {
  suppressPackageStartupMessages({
    library(flashpcaR)
    library(data.table)
  })

  cat(sprintf("Running flashPCA on: %s\n", opt$genoFile))

  # Run flashPCA
  result <- flashpca(opt$genoFile,
                     ndim   = opt$k,
                     nextra = 10L,
                     stand  = "binom2",
                     numthreads = opt$numThreads,
                     do_loadings = TRUE,
                     verbose = TRUE)

  # Assemble PC data frame
  pcs <- as.data.frame(result$vectors)
  colnames(pcs) <- paste0("PC", seq_len(ncol(pcs)))
  pcs$sample_id <- result$samples
  pcs <- pcs[, c("sample_id", paste0("PC", seq_len(opt$k)))]

  # Outlier removal
  pcs_clean <- remove_outliers(pcs, opt$`maha-k`, opt$prob)

  # Proportion of variance explained
  pve <- result$pve * 100
  pve_df <- data.frame(PC = paste0("PC", seq_along(pve)), pve = pve)

  # Save outputs
  rds_out <- paste0(out_prefix, ".pca.rds")
  saveRDS(list(
    pcs      = pcs_clean,
    loadings = result$loadings,
    pve      = pve_df,
    mean     = result$center,
    sd       = result$scale,
    result   = result
  ), rds_out)

  write.table(pcs_clean, paste0(out_prefix, ".pca.pcs.txt"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(pve_df, paste0(out_prefix, ".pca.pve.txt"),
              sep = "\t", quote = FALSE, row.names = FALSE)

  cat(sprintf("PCA RDS saved: %s\n", rds_out))
  cat("Top PVE:\n"); print(head(pve_df))
}

# ── Step: project_samples ────────────────────────────────────────────────────
run_project_samples <- function(opt) {
  suppressPackageStartupMessages({
    library(flashpcaR)
    library(data.table)
  })

  if (is.null(opt$`pca-model`))
    stop("--pca-model is required for project_samples")

  cat(sprintf("Projecting: %s\n", opt$genoFile))
  cat(sprintf("Using PCA model: %s\n", opt$`pca-model`))

  pca_obj <- readRDS(opt$`pca-model`)

  # Project related samples onto unrelated PC space
  projected <- project(pca_obj$result,
                       bfile = opt$genoFile,
                       numthreads = opt$numThreads)

  pcs <- as.data.frame(projected$scores)
  colnames(pcs) <- paste0("PC", seq_len(ncol(pcs)))
  pcs$sample_id <- projected$samples
  pcs <- pcs[, c("sample_id", paste0("PC", seq_len(ncol(pcs) - 1)))]

  # Outlier removal
  k_use <- min(opt$`maha-k`, ncol(pcs) - 1)
  pcs_clean <- remove_outliers(pcs, k_use, opt$prob)

  # Combine with original unrelated PCs
  pcs_all <- rbind(pca_obj$pcs, pcs_clean)

  rds_out <- file.path(opt$cwd,
    sub("\\.pca\\.rds$", "", basename(opt$`pca-model`)))
  rds_out <- paste0(gsub("\\.unrelated\\.plink_qc\\.prune", "", rds_out),
                    ".pca.projected.rds")

  saveRDS(list(
    pcs = pcs_all,
    pve = pca_obj$pve
  ), rds_out)

  write.table(pcs_all, sub("\\.rds$", ".pcs.txt", rds_out),
              sep = "\t", quote = FALSE, row.names = FALSE)

  cat(sprintf("Projected PCA RDS saved: %s\n", rds_out))
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  flashpca        = run_flashpca(opt),
  project_samples = run_project_samples(opt),
  stop(sprintf("Unknown step '%s'. Available: flashpca, project_samples", opt$step))
)
