#!/usr/bin/env Rscript
# ============================================================
# covariate_hidden_factor.R
# Mirrors: code/data_preprocessing/covariate/covariate_hidden_factor.ipynb
#
# Steps (selected via --step):
#   compute_residual — regress covariates out of phenotype (notebook [*_1])
#   Marchenko_PC     — Marchenko-Pastur PCA on residual file (notebook [Marchenko_PC_2])
#
# Legacy combined steps (kept for backward compatibility):
#   Marchenko_PC_full — compute_residual + Marchenko_PC in one call
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
  # Inputs for compute_residual / legacy combined steps
  make_option("--phenoFile",             type = "character", default = NULL,
              help = "Input phenotype BED.gz file"),
  make_option("--covFile",               type = "character", default = NULL,
              help = "Merged covariate file (output of covariate_formatting.R)"),
  # Input for Marchenko_PC sub-step
  make_option("--residFile",             type = "character", default = NULL,
              help = "[Marchenko_PC] Residual phenotype .bed.gz from compute_residual"),
  make_option("--output",                type = "character", default = NULL,
              help = "Optional explicit output path"),
  make_option("--choose-k-method",       type = "character", default = "Marchenko",
              help = "[Marchenko_PC] Method to choose PCA factor count: Marchenko or Buja_Eyuboglu"),
  make_option("--N",                     type = "integer",   default = 0,
              help = "Number of hidden factors (0 = auto-determine)"),
  make_option("--mean-impute-missing",   action = "store_true", default = FALSE,
              help = "Mean-impute missing phenotype values before residualization"),
  make_option("--numThreads",            type = "integer",   default = 8,
              help = "Thread count (accepted for notebook interface parity; single-threaded steps ignore it)"),
  make_option("--seed",                  type = "integer",   default = NA,
              help = "[Marchenko_PC] Integer RNG seed set before jackstraw::permutationPA (Buja_Eyuboglu route) for reproducibility; unset = no seeding"),
  make_option("--dry-run",               action = "store_true", default = FALSE,
              help = "Print full command + validate inputs; do not run.")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step)) stop("--step is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

strip_last_ext <- function(path) {
  sub("\\.[^.]+$", "", basename(path))
}

hidden_factor_prefix <- function(pheno_file, cov_file) {
  paste0(sub("\\.bed\\.gz$", "", basename(pheno_file)), ".", strip_last_ext(cov_file))
}

residual_prefix <- function(resid_file) {
  sub("\\.residual\\.bed\\.gz$", "", basename(resid_file))
}

explicit_or_default_output <- function(opt, default_file) {
  if (!is.null(opt$output) && nzchar(opt$output)) {
    opt$output
  } else {
    default_file
  }
}

bed_sort_index <- function(df) {
  chrom <- as.character(df[[1]])
  start <- suppressWarnings(as.numeric(df[[2]]))
  end <- suppressWarnings(as.numeric(df[[3]]))

  chrom_key <- sub("^chr", "", chrom, ignore.case = TRUE)
  chrom_rank <- suppressWarnings(as.numeric(chrom_key))
  chrom_upper <- toupper(chrom_key)
  chrom_rank[chrom_upper == "X"] <- 23
  chrom_rank[chrom_upper == "Y"] <- 24
  chrom_rank[chrom_upper %in% c("M", "MT")] <- 25

  unresolved <- is.na(chrom_rank)
  if (any(unresolved)) {
    chrom_rank[unresolved] <- 1000 + match(chrom[unresolved], unique(chrom[unresolved]))
  }

  start[is.na(start)] <- Inf
  end[is.na(end)] <- Inf
  order(chrom_rank, start, end, chrom, method = "radix")
}

write_bgzip_bed <- function(df, out_file) {
  plain_file <- sub("\\.gz$", "", out_file)
  df <- df[bed_sort_index(df), , drop = FALSE]
  readr::write_delim(df, plain_file, delim = "\t")
  Rsamtools::bgzip(plain_file, dest = out_file, overwrite = TRUE)
  unlink(plain_file)
  Rsamtools::indexTabix(out_file, format = "bed")
}

mean_impute_old <- function(d) {
  f <- apply(d, 2, function(x) mean(x, na.rm = TRUE))
  for (i in seq_along(f)) {
    d[, i][which(is.na(d[, i]))] <- f[i]
  }
  d
}

# ── Shared: compute residual phenotype ───────────────────────────────────────
compute_residuals <- function(opt) {
  cat("=== Sub-step 1: compute residuals ===\n")
  pheno <- read_delim(opt$phenoFile, delim = "\t", show_col_types = FALSE)
  covariate <- read_delim(opt$covFile, delim = "\t", show_col_types = FALSE)
  extraction_sample_list <- intersect(colnames(pheno), colnames(covariate))

  if (length(extraction_sample_list) == 0) {
    stop("No samples are overlapped in two files!")
  }

  cat(sprintf("%d samples are in the phenotype file\n", ncol(pheno) - 4))
  cat(sprintf("%d samples are in the covariate file\n", ncol(covariate) - 1))
  cat(sprintf("%d samples overlap between phenotype & covariate files and are included in the analysis\n",
              length(extraction_sample_list)))

  covariate <- covariate[, extraction_sample_list, drop = FALSE] %>% as.matrix() %>% t()
  pheno_id <- pheno %>% select(1:4)
  pheno_mat <- pheno %>% select(all_of(rownames(covariate))) %>% as.matrix() %>% t()

  if (isTRUE(opt$`mean-impute-missing`)) {
    pheno_mat <- mean_impute_old(pheno_mat)
  } else if (sum(is.na(pheno_mat)) > 0) {
    stop("NA in phenotype input is not allowed!")
  }

  pheno_resid <- .lm.fit(x = cbind(1, covariate), y = pheno_mat)$residuals
  pheno_output <- cbind(pheno_id, pheno_resid %>% t())

  list(
    residuals = pheno_resid,
    resid_df = pheno_output,
    cov_df = read_delim(opt$covFile, delim = "\t", show_col_types = FALSE),
    shared = extraction_sample_list
  )
}

# ── Step: Marchenko_PC ────────────────────────────────────────────────────────
run_marchenko <- function(opt) {
  # ── Dry-run ─────────────────────────────────────────────────────────────────
  if (isTRUE(opt$`dry-run`)) {
    script_path <- tryCatch(normalizePath(sys.frame(0)$filename), error = function(e) "covariate_hidden_factor.R")
    cat("[DRY-RUN] covariate_hidden_factor.R Marchenko_PC — would execute:\n")
    cat(sprintf("  Rscript %s \\\n",    script_path))
    cat(sprintf("    --step Marchenko_PC \\\n"))
    cat(sprintf("    --phenoFile %s \\\n", opt$phenoFile))
    cat(sprintf("    --covFile %s \\\n",   opt$covFile))
    cat(sprintf("    --N %d \\\n",         opt$N))
    cat(sprintf("    --cwd %s\n",            opt$cwd))
    cat("\n[DRY-RUN] Input file check:\n")
    for (f in c(opt$phenoFile, opt$covFile)) {
      if (is.null(f) || is.na(f)) next
      status <- if (file.exists(f)) "✓" else "✗ NOT FOUND"
      cat(sprintf("  %s  %s\n", status, f))
    }
    quit(status = 0)
  }

  res <- compute_residuals(opt)
  cat("=== Sub-step 2: Marchenko-Pastur PCA ===\n")

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
  bname    <- sub("\\.bed\\.gz$", "", basename(opt$phenoFile))
  out_file <- file.path(opt$cwd, paste0(bname, ".Marchenko_PC.gz"))
  write_tsv(factors, out_file)   # readr detects .gz and compresses automatically
  cat(sprintf("Output: %s (%d factors × %d samples)\n",
              out_file, n_factors, ncol(mat)))
}

# ── Sub-step helpers ──────────────────────────────────────────────────────────

# compute_residual: standalone step matching notebook [*_1]
run_compute_residual <- function(opt) {
  if (is.null(opt$phenoFile)) stop("--phenoFile is required for compute_residual")
  if (is.null(opt$covFile))   stop("--covFile is required for compute_residual")
  res <- compute_residuals(opt)
  bname <- hidden_factor_prefix(opt$phenoFile, opt$covFile)
  out_file <- explicit_or_default_output(
    opt,
    file.path(opt$cwd, paste0(bname, ".residual.bed.gz"))
  )
  dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
  write_bgzip_bed(res$resid_df, out_file)
  cat(sprintf("compute_residual output: %s (%d genes × %d samples)\n",
              out_file, ncol(res$residuals), nrow(res$residuals)))
}

# Marchenko_PC sub-step: takes residFile, matching notebook [Marchenko_PC_2]
run_marchenko_from_resid <- function(opt) {
  if (is.null(opt$residFile)) stop("--residFile is required for Marchenko_PC")
  if (is.null(opt$covFile)) stop("--covFile is required for Marchenko_PC")
  choose_k_method <- opt$`choose-k-method`
  allowed_methods <- c("Marchenko", "Buja_Eyuboglu")
  if (!choose_k_method %in% allowed_methods) {
    stop(sprintf(
      "Invalid choice of methods to choose K for PCA: %s. Available: %s",
      choose_k_method,
      paste(allowed_methods, collapse = ", ")
    ))
  }
  cat("=== Marchenko_PC (from residual file) ===\n")
  suppressPackageStartupMessages(library(PCAtools))
  suppressPackageStartupMessages(library(BiocSingular))
  resid_df <- read_delim(opt$residFile, delim = "\t", show_col_types = FALSE)
  cov_df <- read_delim(opt$covFile, delim = "\t", show_col_types = FALSE)
  common_samples <- intersect(colnames(cov_df), colnames(resid_df))
  if (length(common_samples) == 0L) {
    stop("No overlapping samples between residual phenotype and covariate inputs")
  }
  cov_df_common <- cbind(cov_df[, 1, drop = FALSE], cov_df[, common_samples, drop = FALSE])
  bname <- residual_prefix(opt$residFile)
  resid_pc <- pca(
    resid_df[, common_samples, drop = FALSE],
    scale = TRUE,
    center = TRUE,
    BSPARAM = ExactParam()
  )

  if (opt$N == 0L) {
    if (choose_k_method == "Marchenko") {
      M <- apply(resid_df[, common_samples, drop = FALSE], 1, function(x) {
        (x - mean(x)) / sqrt(var(x))
      })
      resid_sigma2 <- var(as.vector(M))
      n_factors <- chooseMarchenkoPastur(
        .dim = dim(resid_df[, common_samples, drop = FALSE]),
        var.explained = resid_pc$sdev^2,
        noise = resid_sigma2
      )
    } else if (choose_k_method == "Buja_Eyuboglu") {
      if (!requireNamespace("jackstraw", quietly = TRUE)) {
        stop("Package 'jackstraw' is required for choose-k-method Buja_Eyuboglu")
      }
      # permutationPA permutes the data B times via sample() (pure-R RNG); seed for reproducibility.
      if (!is.null(opt$seed) && !is.na(opt$seed)) set.seed(as.integer(opt$seed))
      n_factors <- jackstraw::permutationPA(
        data.matrix(resid_df[, common_samples, drop = FALSE]),
        B = 100,
        threshold = 0.05,
        verbose = FALSE
      )$r
    }
  } else {
    n_factors <- opt$N
  }

  if (n_factors == 0L) {
    stop(sprintf(
      "Invalid choice of methods to choose K for PCA: %s (returned %d)",
      choose_k_method,
      n_factors
    ))
  }

  factors <- as.data.frame(resid_pc$rotated[, seq_len(n_factors), drop = FALSE])
  colnames(factors) <- paste0("Hidden_Factor_PC", seq_len(n_factors))
  factors <- as.data.frame(t(factors))
  factors$id <- rownames(factors)
  factors <- factors %>% select(id, everything()) %>% rename("#id" = "id")
  out_file <- explicit_or_default_output(
    opt,
    file.path(opt$cwd, paste0(bname, ".", choose_k_method, "_PC.gz"))
  )
  dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
  write_delim(rbind(cov_df_common, factors), out_file, "\t")
  cat(sprintf("Output: %s (%d factors × %d samples)\n",
              out_file, n_factors, length(common_samples)))
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  # Fine-grained sub-steps (matching notebook structure)
  compute_residual = run_compute_residual(opt),
  Marchenko_PC     = run_marchenko_from_resid(opt),
  # Legacy combined step (backward compatibility)
  Marchenko_PC_full = {
    if (is.null(opt$phenoFile)) stop("--phenoFile is required")
    if (is.null(opt$covFile))   stop("--covFile is required")
    run_marchenko(opt)
  },
  stop(sprintf(
    "Unknown step '%s'. Available: compute_residual, Marchenko_PC, Marchenko_PC_full",
    opt$step))
)
