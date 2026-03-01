#!/usr/bin/env Rscript
# ============================================================
# phenotype_imputation.R
# Mirrors: code/data_preprocessing/phenotype/phenotype_imputation.ipynb
#
# Steps (selected via --step):
#   EBMF        — Empirical Bayes Matrix Factorization (flashier)
#   gEBMF       — Grouped EBMF
#   missforest  — Random forest imputation
#   knn         — K-nearest neighbors imputation
#   soft        — Soft Impute
#   mean        — Mean imputation
#   lod         — Limit of Detection imputation
#   bed_filter_na — Filter NAs, then apply soft or mean imputation
#
# All steps share the same QC filters applied before imputation:
#   - Remove rows with > qc_missing_rate fraction of missing values
#   - Remove rows with > qc_zero_rate fraction of zeros (or zero+NA)
#
# Flags are kept identical to the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages(library(optparse))

opt_list <- list(
  # ── Universal parameters ──────────────────────────────────────────────────
  make_option("--step",              type = "character", default = NULL),
  make_option("--phenoFile",         type = "character", default = NULL,
              help = "Input BED.gz phenotype file"),
  make_option("--cwd",               type = "character", default = "output"),
  make_option("--numThreads",        type = "integer",   default = 20),
  make_option("--qc-prior-to-impute",type = "logical",   default = TRUE,
              help = "Apply QC filters before imputation"),
  make_option("--qc-missing-rate",   type = "double",    default = 0.4,
              help = "Max fraction of missing values per row"),
  make_option("--qc-zero-rate",      type = "double",    default = 0.95,
              help = "Max fraction of zeros (or zero+NA) per row"),

  # ── EBMF-specific ─────────────────────────────────────────────────────────
  make_option("--prior",             type = "character", default = "ebnm_point_laplace"),
  make_option("--varType",           type = "character", default = "1"),
  make_option("--num-factor",        type = "integer",   default = 60),

  # ── gEBMF-specific ────────────────────────────────────────────────────────
  make_option("--nCores",            type = "integer",   default = 1),
  make_option("--backfit-iter",      type = "integer",   default = 3),
  make_option("--save-flash",        type = "logical",   default = TRUE,
              help = "Save FLASH object to .rds"),
  make_option("--null-check",        type = "logical",   default = FALSE),

  # ── bed_filter_na-specific ────────────────────────────────────────────────
  make_option("--rank-max",          type = "integer",   default = 50),
  make_option("--lambda-hyp",        type = "integer",   default = 30),
  make_option("--impute-method",     type = "character", default = "soft",
              help = "Imputation method for bed_filter_na: soft or mean"),
  make_option("--tol-missing",       type = "double",    default = 0.05,
              help = "Missing rate tolerance for bed_filter_na")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))     stop("--step is required")
if (is.null(opt$phenoFile)) stop("--phenoFile is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

read_bed <- function(path) {
  cat(sprintf("Reading: %s\n", path))
  dat <- read.table(gzfile(path), header = TRUE, sep = "\t",
                    stringsAsFactors = FALSE, check.names = FALSE)
  dat
}

write_bed <- function(dat, path) {
  cat(sprintf("Writing: %s\n", path))
  gz <- gzfile(path, "w")
  write.table(dat, gz, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz)
  system(paste("tabix -p bed", path))
}

qc_filter <- function(mat, missing_rate = 0.4, zero_rate = 0.95) {
  # mat: numeric matrix, rows = features, cols = samples
  n   <- ncol(mat)
  mis <- rowMeans(is.na(mat))
  zer <- rowMeans(mat == 0 | is.na(mat), na.rm = FALSE)
  keep <- mis <= missing_rate & zer <= zero_rate
  cat(sprintf("QC: retaining %d / %d rows (removed %d high-missing, %d high-zero)\n",
              sum(keep), nrow(mat),
              sum(mis > missing_rate), sum(zer > zero_rate & mis <= missing_rate)))
  mat[keep, , drop = FALSE]
}

get_outpath <- function(opt, suffix) {
  bname <- sub("\\.bed\\.gz$", "", basename(opt$phenoFile))
  file.path(opt$cwd, paste0(bname, suffix))
}

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

run_EBMF <- function(opt) {
  suppressPackageStartupMessages(library(flashier))
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  if (isTRUE(opt$`qc-prior-to-impute`))
    mat <- qc_filter(mat, opt$`qc-missing-rate`, opt$`qc-zero-rate`)
  prior_fn <- switch(opt$prior,
    ebnm_point_laplace = ebnm::ebnm_point_laplace,
    ebnm_point_normal  = ebnm::ebnm_point_normal,
    stop("Unknown prior: ", opt$prior))
  fl <- flash(mat, ebnm_fn = prior_fn, var_type = as.integer(opt$varType),
              n_threads = opt$numThreads, greedy_Kmax = opt[["num-factor"]])
  imputed <- fitted(fl)
  mat[is.na(mat)] <- imputed[is.na(mat)]
  out <- cbind(coord[rownames(mat), ], as.data.frame(mat))
  write_bed(out, get_outpath(opt, ".EBMF.imputed.bed.gz"))
}

run_gEBMF <- function(opt) {
  suppressPackageStartupMessages(library(flashier))
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  if (isTRUE(opt$`qc-prior-to-impute`))
    mat <- qc_filter(mat, opt$`qc-missing-rate`, opt$`qc-zero-rate`)
  fl <- flash(mat, var_type = 2L, n_threads = opt$nCores,
              greedy_Kmax = opt[["num-factor"]], backfit = opt[["backfit-iter"]])
  imputed <- fitted(fl)
  mat[is.na(mat)] <- imputed[is.na(mat)]
  out <- cbind(coord[rownames(mat), ], as.data.frame(mat))
  write_bed(out, get_outpath(opt, ".gEBMF.imputed.bed.gz"))
  if (isTRUE(opt$`save-flash`))
    saveRDS(fl, get_outpath(opt, ".gEBMF_factors.rds"))
}

run_missforest <- function(opt) {
  suppressPackageStartupMessages(library(missForest))
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  if (isTRUE(opt$`qc-prior-to-impute`))
    mat <- qc_filter(mat, opt$`qc-missing-rate`, opt$`qc-zero-rate`)
  res <- missForest(t(mat), ntree = 100, parallelize = "forests")
  imputed_t <- res$ximp
  out <- cbind(coord[rownames(mat), ], as.data.frame(t(imputed_t)))
  write_bed(out, get_outpath(opt, ".missForest.imputed.bed.gz"))
}

run_knn <- function(opt) {
  suppressPackageStartupMessages(library(impute))
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  if (isTRUE(opt$`qc-prior-to-impute`))
    mat <- qc_filter(mat, opt$`qc-missing-rate`, opt$`qc-zero-rate`)
  res <- impute.knn(mat)
  out <- cbind(coord[rownames(mat), ], as.data.frame(res$data))
  write_bed(out, get_outpath(opt, ".knn.imputed.bed.gz"))
}

run_soft <- function(opt) {
  suppressPackageStartupMessages(library(softImpute))
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  if (isTRUE(opt$`qc-prior-to-impute`))
    mat <- qc_filter(mat, opt$`qc-missing-rate`, opt$`qc-zero-rate`)
  fit <- softImpute(mat, rank.max = opt[["rank-max"]], lambda = opt[["lambda-hyp"]])
  imputed <- complete(mat, fit)
  out <- cbind(coord[rownames(mat), ], as.data.frame(imputed))
  write_bed(out, get_outpath(opt, ".soft.imputed.bed.gz"))
}

run_mean <- function(opt) {
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  if (isTRUE(opt$`qc-prior-to-impute`))
    mat <- qc_filter(mat, opt$`qc-missing-rate`, opt$`qc-zero-rate`)
  row_means <- rowMeans(mat, na.rm = TRUE)
  for (i in seq_len(nrow(mat))) {
    na_idx <- is.na(mat[i, ])
    if (any(na_idx)) mat[i, na_idx] <- row_means[i]
  }
  out <- cbind(coord[rownames(mat), ], as.data.frame(mat))
  write_bed(out, get_outpath(opt, ".mean.imputed.bed.gz"))
}

run_lod <- function(opt) {
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])
  # LOD: replace NA with half the minimum observed value per row
  for (i in seq_len(nrow(mat))) {
    na_idx <- is.na(mat[i, ])
    if (any(na_idx)) {
      lod_val <- min(mat[i, !na_idx], na.rm = TRUE) / 2
      mat[i, na_idx] <- lod_val
    }
  }
  out <- cbind(coord, as.data.frame(mat))
  write_bed(out, get_outpath(opt, ".lod.imputed.bed.gz"))
}

run_bed_filter_na <- function(opt) {
  suppressPackageStartupMessages(library(softImpute))
  dat <- read_bed(opt$phenoFile)
  coord <- dat[, 1:4]; mat <- as.matrix(dat[, -(1:4)])

  # Filter rows exceeding tolerance
  mis <- rowMeans(is.na(mat))
  keep <- mis <= opt[["tol-missing"]]
  cat(sprintf("bed_filter_na: retaining %d / %d rows\n", sum(keep), nrow(mat)))
  mat  <- mat[keep, , drop = FALSE]
  coord <- coord[keep, , drop = FALSE]

  # Impute remaining NAs
  if (any(is.na(mat))) {
    if (opt[["impute-method"]] == "soft") {
      fit <- softImpute(mat, rank.max = opt[["rank-max"]], lambda = opt[["lambda-hyp"]])
      mat <- complete(mat, fit)
    } else {
      row_means <- rowMeans(mat, na.rm = TRUE)
      for (i in seq_len(nrow(mat))) {
        na_idx <- is.na(mat[i, ])
        if (any(na_idx)) mat[i, na_idx] <- row_means[i]
      }
    }
  }

  out <- cbind(coord, as.data.frame(mat))
  write_bed(out, get_outpath(opt, ".filtered.imputed.bed.gz"))
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch(opt$step,
  EBMF         = run_EBMF(opt),
  gEBMF        = run_gEBMF(opt),
  missforest   = run_missforest(opt),
  knn          = run_knn(opt),
  soft         = run_soft(opt),
  mean         = run_mean(opt),
  lod          = run_lod(opt),
  bed_filter_na = run_bed_filter_na(opt),
  stop(sprintf("Unknown step '%s'. Available: EBMF, gEBMF, missforest, knn, soft, mean, lod, bed_filter_na",
               opt$step))
)
