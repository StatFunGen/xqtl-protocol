#!/usr/bin/env Rscript
# ============================================================
# covariate_hidden_factor.R
# Mirrors: code/data_preprocessing/covariate/covariate_hidden_factor.ipynb
#
# Steps (selected via --step):
#   compute_residual — regress covariates out of phenotype (notebook [*_1])
#   Marchenko_PC     — Marchenko-Pastur PCA on residual file (notebook [Marchenko_PC_2])
#   PEER_fit         — PEER factor model fitting (notebook [PEER_2])
#   PEER_extract     — extract PEER factors from model (notebook [PEER_3])
#   BiCV_2           — build a single-site fake VCF from phenotype BED (notebook [BiCV_2])
#   BiCV_3           — run APEX factor on residual BED + fake VCF (notebook [BiCV_3])
#
# Legacy combined steps (kept for backward compatibility):
#   Marchenko_PC_full — compute_residual + Marchenko_PC in one call
#   PEER              — compute_residual + PEER_fit + PEER_extract in one call
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
  # Input for Marchenko_PC / PEER_fit sub-steps
  make_option("--residFile",             type = "character", default = NULL,
              help = "[Marchenko_PC / PEER_fit] Residual phenotype .bed.gz from compute_residual"),
  # Input for PEER_extract
  make_option("--modelFile",             type = "character", default = NULL,
              help = "[PEER_extract] PEER model .rds file from PEER_fit"),
  make_option("--vcfFile",               type = "character", default = NULL,
              help = "[BiCV_3] Fake VCF generated from residual BED"),
  make_option("--output",                type = "character", default = NULL,
              help = "Optional explicit output path"),
  make_option("--choose-k-method",       type = "character", default = "Marchenko",
              help = "[Marchenko_PC] Method to choose PCA factor count: Marchenko or Buja_Eyuboglu"),
  make_option("--N",                     type = "integer",   default = 0,
              help = "Number of hidden factors (0 = auto-determine)"),
  make_option("--mean-impute-missing",   action = "store_true", default = FALSE,
              help = "Mean-impute missing phenotype values before residualization"),
  # PEER-specific
  make_option("--iteration",             type = "integer",   default = 1000),
  make_option("--convergence-mode",      type = "character", default = "fast",
              help = "PEER convergence mode: fast, medium, slow"),
  make_option("--tol",                   type = "double",    default = 0.001,
              help = "[PEER_fit] PEER/MOFA convergence tolerance"),
  make_option("--r2-tol",                type = "character", default = "False",
              help = "[PEER_fit] Optional PEER/MOFA dropR2 setting; False disables it"),
  make_option("--numThreads",            type = "integer",   default = 8),
  make_option("--dry-run",               action = "store_true", default = FALSE,
              help = "Print full command + validate inputs; do not run.")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step)) stop("--step is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

strip_last_ext <- function(path) {
  sub("\\.[^.]+$", "", basename(path))
}

# Directory holding this script (so we can locate covariate_hidden_factor_peer.py)
script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  getwd()
}

hidden_factor_prefix <- function(pheno_file, cov_file) {
  paste0(sub("\\.bed\\.gz$", "", basename(pheno_file)), ".", strip_last_ext(cov_file))
}

residual_prefix <- function(resid_file) {
  sub("\\.residual\\.bed\\.gz$", "", basename(resid_file))
}

peer_model_prefix <- function(model_file) {
  sub("\\.PEER_MODEL\\.[^.]+$", "", basename(model_file))
}

explicit_or_default_output <- function(opt, default_file) {
  if (!is.null(opt$output) && nzchar(opt$output)) {
    opt$output
  } else {
    default_file
  }
}

fake_vcf_prefix <- function(resid_file) {
  sub("\\.gz$", "", basename(resid_file))
}

phenotype_coord_cols <- function(df) {
  if ("Description" %in% colnames(df)[seq_len(min(5, ncol(df)))]) {
    colnames(df)[1:5]
  } else {
    colnames(df)[1:4]
  }
}

run_command <- function(command, args = character()) {
  status <- system2(command, args = args)
  if (!identical(status, 0L)) {
    stop(sprintf("Command failed: %s %s", command, paste(args, collapse = " ")))
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
  run_command("bgzip", c("-f", plain_file))
  run_command("tabix", c("-f", "-p", "bed", out_file))
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
      status <- if (file.exists(f)) "\u2713" else "\u2717 NOT FOUND"
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

# # ── Step: PEER ────────────────────────────────────────────────────────────────
# run_peer <- function(opt) {
#   # ── Dry-run ─────────────────────────────────────────────────────────────────
#   if (isTRUE(opt$`dry-run`)) {
#     script_path <- tryCatch(normalizePath(sys.frame(0)$filename), error = function(e) "covariate_hidden_factor.R")
#     cat("[DRY-RUN] covariate_hidden_factor.R PEER — would execute:\n")
#     cat(sprintf("  Rscript %s \\\n",    script_path))
#     cat(sprintf("    --step PEER \\\n"))
#     cat(sprintf("    --phenoFile %s \\\n",  opt$phenoFile))
#     cat(sprintf("    --covFile %s \\\n",    opt$covFile))
#     cat(sprintf("    --N %d \\\n",          opt$N))
#     cat(sprintf("    --iteration %d \\\n",  opt$iteration))
#     cat(sprintf("    --convergence-mode %s \\\n", opt$`convergence-mode`))
#     cat(sprintf("    --cwd %s\n",             opt$cwd))
#     cat("\n[DRY-RUN] Input file check:\n")
#     for (f in c(opt$phenoFile, opt$covFile)) {
#       if (is.null(f) || is.na(f)) next
#       status <- if (file.exists(f)) "\u2713" else "\u2717 NOT FOUND"
#       cat(sprintf("  %s  %s\n", status, f))
#     }
#     quit(status = 0)
#   }
# 
#   res <- compute_residuals(opt)
#   cat("=== Sub-step 2: PEER factor analysis ===\n")
# 
#   suppressPackageStartupMessages(library(peer))
# 
#   mat <- res$residuals
#   n_samples  <- ncol(mat)
#   n_features <- nrow(mat)
# 
#   n_factors <- if (opt$N == 0) {
#     min(n_samples, 25L)   # PEER default heuristic
#   } else {
#     opt$N
#   }
#   cat(sprintf("Running PEER with %d factors, %d iterations\n",
#               n_factors, opt$iteration))
# 
#   model <- PEER()
#   PEER_setPhenoMean(model, t(mat))
#   PEER_setNk(model, n_factors)
#   PEER_setMaxIter(model, opt$iteration)
#   PEER_update(model)
# 
#   # Save PEER model
#   bname      <- sub("\\.bed\\.gz$", "", basename(opt$phenoFile))
#   saveRDS(model, file.path(opt$cwd, paste0(bname, ".PEER_MODEL.rds")))
# 
#   cat("=== Sub-step 3: Extract PEER factors ===\n")
#   factors_mat <- t(PEER_getX(model))   # factors × samples
#   # Use Hidden_Factor_PC prefix to match the SoS notebook (mofapy2/MOFA2) naming
#   rownames(factors_mat) <- paste0("Hidden_Factor_PC", seq_len(nrow(factors_mat)))
#   colnames(factors_mat) <- colnames(mat)
# 
#   factors_df <- cbind(ID = rownames(factors_mat), as.data.frame(factors_mat))
#   out_file   <- file.path(opt$cwd, paste0(bname, ".PEER.gz"))
#   write_tsv(factors_df, out_file)
#   cat(sprintf("Output: %s (%d factors × %d samples)\n",
#               out_file, nrow(factors_mat), ncol(mat)))
# }

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

# PEER_fit sub-step: takes residFile, matching notebook [PEER_2]
# Delegates to the standalone covariate_hidden_factor_peer.py, which fits the
# mofapy2 MOFA model, saves the HDF5 model, and writes the factor/weight/variance
# TSV sidecars that PEER_extract reads (no MOFA2 Bioconductor package / reticulate).
run_peer_fit <- function(opt) {
  if (is.null(opt$residFile)) stop("--residFile is required for PEER_fit")
  cat("=== PEER_fit (mofapy2) ===\n")
  bname <- sub("\\.bed\\.gz$", "", basename(opt$residFile))
  model_file    <- file.path(opt$cwd, paste0(bname, ".PEER_MODEL.hd5"))
  factors_file  <- file.path(opt$cwd, paste0(bname, ".PEER.factors.tsv"))
  weights_file  <- file.path(opt$cwd, paste0(bname, ".PEER.weights.tsv"))
  variance_file <- file.path(opt$cwd, paste0(bname, ".PEER.variance.tsv"))
  py_script <- file.path(script_dir(), "covariate_hidden_factor_peer.py")
  if (!file.exists(py_script)) stop(sprintf("PEER Python worker not found: %s", py_script))
  run_command(
    Sys.which("python"),
    c(py_script,
      "--resid-file",       opt$residFile,
      "--model-file",       model_file,
      "--factors-out",      factors_file,
      "--weights-out",      weights_file,
      "--variance-out",     variance_file,
      "--num-factor",       as.character(opt$N),
      "--iteration",        as.character(opt$iteration),
      "--convergence-mode", opt$`convergence-mode`,
      "--num-threads",      as.character(opt$numThreads),
      "--tol",              as.character(opt$tol),
      "--r2-tol",           as.character(opt$`r2-tol`))
  )
  cat(sprintf("PEER model saved: %s\n", model_file))
}

# Diagnostic PDF from the PEER TSV sidecars (replaces the MOFA2 plot_* family).
# factors_df: '#id' (Factor*) + sample columns; weights_df: 'feature' + Factor*;
# variance_df: factor, r2 (per-factor rows + a 'Total' row).
make_peer_diag_pdf <- function(factors_df, weights_df, variance_df, diag_file) {
  suppressPackageStartupMessages({
    library(ggplot2)
    library(tidyr)
  })
  factor_names <- factors_df$`#id`
  sample_cols <- setdiff(colnames(factors_df), "#id")
  fac_mat <- as.matrix(factors_df[, sample_cols, drop = FALSE])   # K x N
  rownames(fac_mat) <- factor_names
  fac_lvls <- factor(factor_names, levels = factor_names)

  pdf(diag_file, width = 7, height = 5)

  # (1) Variance explained per factor
  vperf <- variance_df[variance_df$factor != "Total", , drop = FALSE]
  vperf$factor <- factor(vperf$factor, levels = vperf$factor)
  total_r2 <- variance_df$r2[variance_df$factor == "Total"]
  subtitle <- if (length(total_r2) == 1) sprintf("Total variance explained: %.2f%%", total_r2) else NULL
  print(
    ggplot(vperf, aes(x = factor, y = r2)) +
      geom_col(fill = "steelblue") +
      labs(title = "Variance explained per factor", subtitle = subtitle,
           x = NULL, y = "Variance explained (%)") +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )

  # (2) Factor value distributions across samples
  long <- pivot_longer(factors_df, all_of(sample_cols), names_to = "sample", values_to = "value")
  long$`#id` <- factor(long$`#id`, levels = factor_names)
  print(
    ggplot(long, aes(x = `#id`, y = value)) +
      geom_boxplot(outlier.size = 0.6, fill = "grey85") +
      labs(title = "Factor value distributions", x = NULL, y = "Factor value") +
      theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )

  # (3) Scatter of the first two factors + (4) factor correlation heatmap
  if (length(factor_names) >= 2L) {
    scat <- data.frame(sample = sample_cols, F1 = fac_mat[1, ], F2 = fac_mat[2, ])
    print(
      ggplot(scat, aes(x = F1, y = F2)) +
        geom_point(color = "steelblue", alpha = 0.8) +
        labs(title = "Samples in factor space",
             x = factor_names[1], y = factor_names[2]) +
        theme_bw()
    )
    cor_mat <- cor(t(fac_mat))
    cor_long <- as.data.frame(as.table(cor_mat))
    colnames(cor_long) <- c("Var1", "Var2", "cor")
    cor_long$Var1 <- factor(cor_long$Var1, levels = factor_names)
    cor_long$Var2 <- factor(cor_long$Var2, levels = rev(factor_names))
    print(
      ggplot(cor_long, aes(Var1, Var2, fill = cor)) +
        geom_tile() +
        scale_fill_gradient2(limits = c(-1, 1), low = "blue", mid = "white", high = "red") +
        labs(title = "Factor correlation", x = NULL, y = NULL) +
        theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    )
  }

  # (5) Top features by |weight| for Factor1
  f1 <- factor_names[1]
  top <- weights_df[order(-abs(weights_df[[f1]])), c("feature", f1)]
  top <- head(top, 10)
  top$feature <- factor(top$feature, levels = rev(top$feature))
  print(
    ggplot(top, aes(x = .data[[f1]], y = feature)) +
      geom_col(fill = "darkorange") +
      labs(title = sprintf("Top 10 features by weight (%s)", f1),
           x = "Weight", y = NULL) +
      theme_bw()
  )

  invisible(dev.off())
}

# PEER_extract sub-step: takes modelFile, matching notebook [PEER_3].
# Reads the TSV sidecars written by PEER_fit (no MOFA2 package / reticulate).
run_peer_extract <- function(opt) {
  if (is.null(opt$modelFile)) stop("--modelFile is required for PEER_extract")
  if (is.null(opt$covFile)) stop("--covFile is required for PEER_extract")
  cat("=== PEER_extract (from PEER TSV sidecars) ===\n")
  bname <- peer_model_prefix(opt$modelFile)
  base_dir <- dirname(opt$modelFile)
  factors_file  <- file.path(base_dir, paste0(bname, ".PEER.factors.tsv"))
  weights_file  <- file.path(base_dir, paste0(bname, ".PEER.weights.tsv"))
  variance_file <- file.path(base_dir, paste0(bname, ".PEER.variance.tsv"))
  for (f in c(factors_file, weights_file, variance_file)) {
    if (!file.exists(f)) {
      stop(sprintf("Expected PEER sidecar not found: %s (run PEER_fit first)", f))
    }
  }
  factors_df  <- read_delim(factors_file,  delim = "\t", show_col_types = FALSE)
  weights_df  <- read_delim(weights_file,  delim = "\t", show_col_types = FALSE)
  variance_df <- read_delim(variance_file, delim = "\t", show_col_types = FALSE)

  cov_df <- read_delim(opt$covFile, delim = "\t", show_col_types = FALSE)
  common_samples <- intersect(colnames(cov_df), colnames(factors_df))
  out_file <- file.path(opt$cwd, paste0(bname, ".PEER.gz"))
  (rbind(cov_df[, common_samples, drop = FALSE], factors_df[, common_samples, drop = FALSE]) %>%
      write_delim(out_file, "\t"))

  diag_file <- file.path(opt$cwd, paste0(bname, ".PEER.diag.pdf"))
  make_peer_diag_pdf(factors_df, weights_df, variance_df, diag_file)
  cat(sprintf("Output: %s (%d factors × %d samples)\n",
              out_file, nrow(factors_df), length(common_samples) - 1L))
}

# BiCV_2 sub-step: create a one-site fake VCF from the residual phenotype BED
run_bicv_fake_vcf <- function(opt) {
  if (is.null(opt$residFile)) stop("--residFile is required for BiCV_2")
  cat("=== BiCV_2 (fake VCF from residual file) ===\n")
  out_file <- file.path(opt$cwd, paste0(fake_vcf_prefix(opt$residFile), ".fake.vcf.gz"))
  plain_file <- sub("\\.gz$", "", out_file)

  pheno <- read_delim(opt$residFile, delim = "\t", n_max = 1, show_col_types = FALSE)
  if (nrow(pheno) == 0L) {
    stop(sprintf("No phenotype rows found in %s", opt$residFile))
  }
  if (ncol(pheno) < 5L) {
    stop(sprintf("Expected at least 5 columns in %s", opt$residFile))
  }

  colnames(pheno)[1:3] <- c("#CHROM", "POS", "ID")
  vcf_row <- cbind(
    pheno[, 1:3, drop = FALSE] %>%
      mutate(REF = "A", ALT = "C", QUAL = ".", FILTER = ".", INFO = "PR", FORMAT = "GT"),
    pheno[, 5:ncol(pheno), drop = FALSE]
  )

  writeLines(
    c(
      "##fileformat=VCFv4.2",
      sprintf("##fileDate=%s", format(Sys.Date(), "%Y%m%d")),
      "##source=FAKE"
    ),
    plain_file
  )
  write_delim(vcf_row, plain_file, delim = "\t", col_names = TRUE, append = TRUE)
  run_command("bgzip", c("-f", plain_file))
  run_command("tabix", c("-f", "-p", "vcf", out_file))
  cat(sprintf("Output: %s\n", out_file))
}

run_bicv_factor <- function(opt) {
  if (is.null(opt$residFile) || !file.exists(opt$residFile)) {
    stop("--residFile is required for BiCV_3")
  }
  if (is.null(opt$vcfFile) || !file.exists(opt$vcfFile)) {
    stop("--vcfFile is required for BiCV_3")
  }

  resid_df <- read_delim(opt$residFile, delim = "\t", show_col_types = FALSE)
  coord_cols <- phenotype_coord_cols(resid_df)
  sample_count <- ncol(resid_df) - length(coord_cols)
  n_factors <- opt$N
  if (n_factors == 0L) {
    if (sample_count < 150) {
      n_factors <- 15L
    } else if (sample_count < 250) {
      n_factors <- 30L
    } else if (sample_count < 350) {
      n_factors <- 45L
    } else {
      n_factors <- 60L
    }
  }

  out_file <- opt$output
  if (is.null(out_file) || !nzchar(out_file)) {
    out_file <- file.path(opt$cwd, paste0(fake_vcf_prefix(opt$residFile), ".BiCV.gz"))
  }
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  out_prefix <- sub("\\.BiCV\\.gz$", "", out_file)
  if (identical(out_prefix, out_file)) {
    out_prefix <- sub("\\.gz$", "", out_file)
  }

  apex_args <- c(
    "factor",
    "--out", out_prefix,
    "--iter", as.character(opt$iteration),
    "--factors", as.character(n_factors),
    "--bed", opt$residFile,
    "--vcf", opt$vcfFile,
    "--threads", as.character(opt$numThreads)
  )
  if (!is.null(opt$covFile) && nzchar(opt$covFile) && file.exists(opt$covFile)) {
    apex_args <- c(apex_args, "--cov", opt$covFile)
  }
  run_command("apex", apex_args)
  # apex writes "<out_prefix>.cov.gz" (the inferred factors formatted as covariates);
  # rename it to the declared BiCV output so the SoS output contract is satisfied.
  apex_out <- paste0(out_prefix, ".cov.gz")
  if (!identical(apex_out, out_file)) {
    if (!file.exists(apex_out)) {
      stop(sprintf("apex did not produce expected output: %s", apex_out))
    }
    if (file.exists(out_file)) unlink(out_file)
    file.rename(apex_out, out_file)
  }
  cat(sprintf("Output: %s\n", out_file))
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  # Fine-grained sub-steps (matching notebook structure)
  compute_residual = run_compute_residual(opt),
  Marchenko_PC     = run_marchenko_from_resid(opt),
  PEER_fit         = run_peer_fit(opt),
  PEER_extract     = run_peer_extract(opt),
  BiCV_2           = run_bicv_fake_vcf(opt),
  BiCV_3           = run_bicv_factor(opt),
  # Legacy combined steps (backward compatibility)
  Marchenko_PC_full = {
    if (is.null(opt$phenoFile)) stop("--phenoFile is required")
    if (is.null(opt$covFile))   stop("--covFile is required")
    run_marchenko(opt)
  },
  # Legacy R-PEER combined step (library(peer)) — commented out: the notebook uses
  # the mofapy2 PEER_fit/PEER_extract path, so this route (and r-peer) is unused.
  # PEER = {
  #   if (is.null(opt$phenoFile)) stop("--phenoFile is required")
  #   if (is.null(opt$covFile))   stop("--covFile is required")
  #   run_peer(opt)
  # },
  stop(sprintf(
    "Unknown step '%s'. Available: compute_residual, Marchenko_PC, PEER_fit, PEER_extract, BiCV_2, BiCV_3",
    opt$step))
)
