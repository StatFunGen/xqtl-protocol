#!/usr/bin/env Rscript
# ============================================================
# bulk_expression_QC.R
# Mirrors: code/molecular_phenotypes/QC/bulk_expression_QC.ipynb
#
# Steps (selected via --step):
#   qc_1 — basic integrity check and low-expression gene filtering on TPM
#   qc_2 — outlier sample removal via RLE + Mahalanobis distance
#   qc_3 — subset raw counts to match QC-filtered TPM (genes and samples)
#
# Flags are kept identical to the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
})

opt_list <- list(
  make_option("--step",                  type = "character", default = NULL,
              help = "Step to run: qc_1, qc_2, qc_3"),
  make_option("--cwd",                   type = "character", default = "output",
              help = "Output directory [default: output]"),
  make_option("--tpm-gct",               type = "character", default = NULL,
              help = "Input TPM GCT.gz file"),
  make_option("--counts-gct",            type = "character", default = NULL,
              help = "[qc_3] Raw counts GCT.gz file"),
  # qc_1 parameters
  make_option("--low-expr-TPM",          type = "double",    default = 0.1,
              help = "[qc_1] Minimum TPM threshold for expression [default: 0.1]"),
  make_option("--low-expr-TPM-percent",  type = "double",    default = 0.2,
              help = "[qc_1] Minimum fraction of samples above threshold [default: 0.2]"),
  # qc_2 parameters
  make_option("--RLEFilterPercent",      type = "double",    default = 0.05,
              help = "[qc_2] RLE IQR outlier cutoff fraction [default: 0.05]"),
  make_option("--DSFilterPercent",       type = "double",    default = 0.05,
              help = "[qc_2] Distance-sum outlier cutoff fraction [default: 0.05]"),
  make_option("--topk-genes",            type = "integer",   default = 100,
              help = "[qc_2] Top-k genes for hierarchical clustering [default: 100]"),
  make_option("--cluster-percent",       type = "double",    default = 0.6,
              help = "[qc_2] Minimum cluster size as fraction of samples [default: 0.6]"),
  make_option("--pvalue-cutoff",         type = "double",    default = 0.05,
              help = "[qc_2] P-value cutoff for AU test [default: 0.05]"),
  make_option("--cluster-level",         type = "integer",   default = 5,
              help = "[qc_2] Number of cluster tree levels to examine [default: 5]"),
  make_option("--numThreads",            type = "integer",   default = 8,
              help = "Number of threads [default: 8]")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))       stop("--step is required")
if (is.null(opt[["tpm-gct"]])) stop("--tpm-gct is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

# ── Step: qc_1 ───────────────────────────────────────────────────────────────
run_qc_1 <- function(opt) {
  suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
    library(tibble)
    library(purrr)
  })

  low_expr_TPM         <- opt[["low-expr-TPM"]]
  low_expr_TPM_percent <- opt[["low-expr-TPM-percent"]]
  tpm_file             <- opt[["tpm-gct"]]

  TPM_data <- read_tsv(tpm_file, col_names = TRUE, comment = "#") %>% as.data.frame()
  names(TPM_data)[1] <- "feature"

  if (sum(duplicated(TPM_data$feature)) > 0) {
    message("feature (e.g. gene names) should be in the first column. Remove duplicates, Exit!")
    quit(save = "no", status = 1, runLast = FALSE)
  }
  rownames(TPM_data) <- TPM_data$feature
  TPM_data <- TPM_data[, -1]

  if (sum(is.na(TPM_data)) > 0) {
    message(paste0("NA is not allowed in the data, there are ", sum(is.na(TPM_data)), " NAs, Exit!"))
    quit(save = "no", status = 1, runLast = FALSE)
  }

  matrix_check <- map(TPM_data, is.numeric) %>% unlist()
  if (sum(!matrix_check) > 0) {
    message("Non-numeric columns detected — excluding them:")
    message(paste(names(matrix_check)[!matrix_check], collapse = "; "))
    TPM_data <- TPM_data[, matrix_check]
  }

  message(paste(nrow(TPM_data), "genes and", ncol(TPM_data), "samples loaded from", tpm_file))

  keep_genes_idx <- (rowMeans(TPM_data > low_expr_TPM) > low_expr_TPM_percent)
  TPM_data       <- TPM_data[keep_genes_idx, ]
  message(paste(sum(!keep_genes_idx), "genes filtered; low expression threshold: >",
                low_expr_TPM_percent * 100, "% samples with TPM <", low_expr_TPM))
  message(paste(sum(keep_genes_idx), "genes retained."))

  bname    <- sub("\\.(gct|GCT)(\\.gz)?$", "", basename(tpm_file))
  out_file <- file.path(opt$cwd, paste0(bname, ".low_expression_filtered.tpm.gct.gz"))

  # Write with GCT-style comment header
  cat(paste0("#1.2\n# ", nrow(TPM_data), " ", ncol(TPM_data), "\n"),
      file = sub("\\.gz$", "", out_file), append = FALSE)
  TPM_data %>%
    as_tibble(rownames = "gene_ID") %>%
    write_delim(sub("\\.gz$", "", out_file), delim = "\t", col_names = TRUE, append = TRUE)
  system2("gzip", c("-f", "--best", shQuote(sub("\\.gz$", "", out_file))))

  message(paste("Output:", out_file))
}

# ── Step: qc_2 ───────────────────────────────────────────────────────────────
run_qc_2 <- function(opt) {
  suppressPackageStartupMessages({
    library(RColorBrewer)
    library(ape)
    library(reshape2)
    library(dplyr)
    library(readr)
    library(MASS)
  })

  RLEFilterPercent <- opt[["RLEFilterPercent"]]
  DSFilterPercent  <- opt[["DSFilterPercent"]]
  pvalues.cut      <- opt[["pvalue-cutoff"]]
  topk_genes       <- opt[["topk-genes"]]
  cluster_percent  <- opt[["cluster-percent"]]
  treesNum         <- opt[["cluster-level"]]
  tpm_file         <- opt[["tpm-gct"]]

  # Robust Mahalanobis (handles rank-deficient covariance)
  mahalanobis_robust <- function(x, center, cov, ...) {
    x <- if (is.vector(x)) matrix(x, ncol = length(x)) else as.matrix(x)
    if (!isFALSE(center)) x <- sweep(x, 2L, center)
    cov_inv <- MASS::ginv(cov, ...)
    setNames(rowSums(x %*% cov_inv * x), rownames(x))
  }

  TPM_data <- read_tsv(tpm_file, col_names = TRUE, comment = "#") %>% as.data.frame()
  rownames(TPM_data) <- TPM_data$gene_ID
  TPM_data <- TPM_data[, -1]

  RLEFilterLength <- RLEFilterPercent * ncol(TPM_data)
  DSFilter        <- DSFilterPercent  * ncol(TPM_data)

  # RLE outlier detection
  log_offset <- 0.0001
  logtpm     <- log10(as.matrix(TPM_data) + log_offset)
  rle        <- logtpm - apply(logtpm, 1, median)
  rle_long   <- melt(rle, variable.name = "Sample", value.name = "TPM", id = "feature")
  names(rle_long) <- c("feature", "Sample", "TPM")
  rle_IQR    <- rle_long %>% group_by(Sample) %>% summarise(IQR = IQR(TPM))
  RLE_outliers <- as.character(
    rle_IQR$Sample[rle_IQR$IQR > quantile(rle_IQR$IQR, 1 - RLEFilterPercent)]
  )
  message(paste("RLE outliers detected:", length(RLE_outliers)))

  # Hierarchical clustering outlier detection
  log_tpm    <- log10(as.matrix(TPM_data) + log_offset)
  gene_var   <- apply(log_tpm, 1, var)
  top_genes  <- names(sort(gene_var, decreasing = TRUE))[seq_len(min(topk_genes, nrow(log_tpm)))]
  dist_mat   <- dist(t(log_tpm[top_genes, ]))
  hc         <- hclust(dist_mat, method = "average")

  DS <- rowSums(as.matrix(dist_mat))
  DS_outliers <- as.character(names(DS)[DS > quantile(DS, 1 - DSFilterPercent)])
  message(paste("Distance-sum outliers detected:", length(DS_outliers)))

  # Mahalanobis outlier detection
  pca_res    <- prcomp(t(log_tpm[top_genes, ]), center = TRUE, scale. = FALSE)
  pc_scores  <- pca_res$x[, seq_len(min(10, ncol(pca_res$x))), drop = FALSE]
  mah_dist   <- tryCatch(
    mahalanobis_robust(pc_scores, colMeans(pc_scores), cov(pc_scores)),
    error = function(e) rep(NA_real_, nrow(pc_scores))
  )
  mah_pval   <- pchisq(mah_dist, df = ncol(pc_scores), lower.tail = FALSE)
  mah_outliers <- names(mah_pval[!is.na(mah_pval) & mah_pval < pvalues.cut])
  message(paste("Mahalanobis outliers detected:", length(mah_outliers)))

  all_outliers <- unique(c(RLE_outliers, DS_outliers, mah_outliers))
  message(paste("Total outliers removed:", length(all_outliers)))
  if (length(all_outliers) > 0) message(paste(" ", all_outliers, collapse = "\n"))

  keep_samples <- setdiff(colnames(TPM_data), all_outliers)
  TPM_clean    <- TPM_data[, keep_samples, drop = FALSE]
  message(paste(ncol(TPM_clean), "samples retained after outlier removal."))

  bname    <- sub("\\.(gct|GCT)(\\.gz)?$", "", basename(tpm_file))
  out_file <- file.path(opt$cwd, paste0(bname, ".outlier_removed.tpm.gct.gz"))

  cat(paste0("#1.2\n# ", nrow(TPM_clean), " ", ncol(TPM_clean), "\n"),
      file = sub("\\.gz$", "", out_file), append = FALSE)
  TPM_clean %>%
    as.data.frame() %>%
    tibble::rownames_to_column("gene_ID") %>%
    write_delim(sub("\\.gz$", "", out_file), delim = "\t", col_names = TRUE, append = TRUE)
  system2("gzip", c("-f", "--best", shQuote(sub("\\.gz$", "", out_file))))

  message(paste("Output:", out_file))
}

# ── Step: qc_3 ───────────────────────────────────────────────────────────────
run_qc_3 <- function(opt) {
  suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
  })

  if (is.null(opt[["counts-gct"]])) stop("--counts-gct is required for qc_3")

  tpm_file    <- opt[["tpm-gct"]]
  counts_file <- opt[["counts-gct"]]

  tpm      <- read_delim(tpm_file,    "\t", col_names = TRUE, comment = "#")
  geneCount <- read_delim(counts_file, "\t", col_names = TRUE, comment = "#")

  # Match genes and samples to QC-filtered TPM
  geneCount <- geneCount %>%
    filter(gene_ID %in% tpm$gene_ID) %>%
    select(colnames(tpm))

  bname    <- sub("\\.(gct|GCT)(\\.gz)?$", "", basename(tpm_file))
  out_file <- file.path(opt$cwd, paste0(bname, ".geneCount.gct.gz"))
  out_tmp  <- sub("\\.gz$", "", out_file)

  cat(paste("#1.2\n#", nrow(geneCount), ncol(geneCount) - 1, "\n"),
      file = out_tmp, append = FALSE)
  geneCount %>%
    write_delim(out_tmp, delim = "\t", col_names = TRUE, append = TRUE)
  system2("gzip", c("-f", "--best", shQuote(out_tmp)))

  message(paste("Output:", out_file))
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  qc_1 = run_qc_1(opt),
  qc_2 = run_qc_2(opt),
  qc_3 = run_qc_3(opt),
  stop(sprintf("Unknown step '%s'. Available: qc_1, qc_2, qc_3", opt$step))
)
