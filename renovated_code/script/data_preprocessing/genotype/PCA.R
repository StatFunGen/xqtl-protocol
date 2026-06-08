#!/usr/bin/env Rscript
# ============================================================
# PCA.R
# Mirrors: code/data_preprocessing/genotype/PCA.ipynb
#
# Steps (selected via --step):
#   flashpca        — compute PCs from LD-pruned unrelated genotypes
#   project_samples — project related samples onto unrelated PCA space
#   plot_pca        — render PCA scatter and scree plots
#   detect_outliers — compute Mahalanobis outliers and summaries
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
  make_option("--phenoFile",   type = "character", default = NULL,
              help = "Phenotype/FAM/PSAM metadata file"),
  make_option("--cwd",         type = "character", default = "output"),
  make_option("--output",      type = "character", default = "",
              help = "[flashpca/project_samples] RDS output path"),
  make_option("--k",           type = "integer",   default = 20,
              help = "Number of PCs to compute"),
  make_option("--maha-k",      type = "integer",   default = 5,
              help = "Number of PCs for Mahalanobis outlier detection"),
  make_option("--stand",       type = "character", default = "binom2",
              help = "[flashpca] Genotype standardization passed to flashpca"),
  make_option("--min-pop-size", type = "integer",  default = 2,
              help = "[flashpca] Minimum population size to retain"),
  make_option("--homogeneous", type = "character", default = "FALSE",
              help = "[flashpca] Whether to compute one shared PC summary"),
  make_option("--prob",        type = "double",    default = 0.997,
              help = "Probability threshold for Mahalanobis outlier removal"),
  make_option("--pval",        type = "double",    default = 0.05,
              help = "[detect_outliers] Mahalanobis p-value cutoff"),
  make_option("--pca-model",   type = "character", default = NULL,
              help = "[project_samples] RDS file from flashpca step"),
  make_option("--plot-data",   type = "character", default = NULL,
              help = "[plot_pca] PCA RDS file to plot"),
  make_option("--pca-result",  type = "character", default = NULL,
              help = "[detect_outliers] PCA RDS file to analyze"),
  make_option("--outlier-file", type = "character", default = "",
              help = "[plot_pca] Optional outlier file"),
  make_option("--min-axis",    type = "character", default = "",
              help = "[plot_pca] Optional lower axis bounds"),
  make_option("--max-axis",    type = "character", default = "",
              help = "[plot_pca] Optional upper axis bounds"),
  make_option("--pop-col",     type = "character", default = "",
              help = "Population column in pc_scores"),
  make_option("--label-col",   type = "character", default = "",
              help = "Label column for coloring PCA plots"),
  make_option("--pops",        type = "character", default = "",
              help = "[plot_pca] Comma-separated populations to plot"),
  make_option("--robust",      type = "character", default = "TRUE",
              help = "[detect_outliers] Whether to use median center"),
  make_option("--output-pc-plot", type = "character", default = "",
              help = "[plot_pca] PCA scatter PNG output"),
  make_option("--output-scree-plot", type = "character", default = "",
              help = "[plot_pca] Scree PNG output"),
  make_option("--output-scree-text", type = "character", default = "",
              help = "[plot_pca] Scree TSV output"),
  make_option("--distance-output", type = "character", default = "",
              help = "[detect_outliers] Mahalanobis RDS output"),
  make_option("--identified-outliers-output", type = "character", default = "",
              help = "[detect_outliers] Outlier list output"),
  make_option("--analysis-summary-output", type = "character", default = "",
              help = "[detect_outliers] Markdown summary output"),
  make_option("--qqplot-output", type = "character", default = "",
              help = "[detect_outliers] QQ plot PNG output"),
  make_option("--hist-output", type = "character", default = "",
              help = "[detect_outliers] Histogram PNG output"),
  make_option("--numThreads",  type = "integer",   default = 8)
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))     stop("--step is required")

dir.create(opt$cwd, showWarnings = FALSE, recursive = TRUE)

resolve_bed_file <- function(geno_file) {
  if (grepl("\\.bed$", geno_file)) return(geno_file)

  candidate <- paste0(geno_file, ".bed")
  if (file.exists(candidate)) return(candidate)

  geno_file
}

resolve_pheno_file <- function(opt, geno_file) {
  if (!is.null(opt$phenoFile) && nzchar(opt$phenoFile)) return(opt$phenoFile)
  sub("\\.bed$", ".fam", resolve_bed_file(geno_file))
}

read_pheno_metadata <- function(pheno_file) {
  if (!file.exists(pheno_file)) stop("Missing phenoFile: ", pheno_file)

  if (grepl("\\.(fam|psam)$", pheno_file)) {
    pheno <- read.table(pheno_file, header = FALSE, stringsAsFactors = FALSE)
    if (ncol(pheno) < 6) stop("Unexpected phenoFile schema: ", pheno_file)
    colnames(pheno)[1:6] <- c("FID", "IID", "MID", "PID", "SEX", "STATUS")
  } else {
    pheno <- read.table(pheno_file, header = TRUE, stringsAsFactors = FALSE)
    if (!"IID" %in% colnames(pheno)) {
      stop("No IID column in the phenoFile. Please rename the header of the phenoFile")
    }
    if (!"FID" %in% colnames(pheno)) pheno$FID <- pheno$IID
  }

  pheno$ID <- paste(pheno$FID, pheno$IID, sep = ":")
  if (length(unique(pheno$ID)) != length(pheno$ID)) {
    stop("There are duplicated names in IID column of phenoFile")
  }
  pheno
}

read_bim_reference <- function(geno_file) {
  bim_file <- sub("\\.bed$", ".bim", resolve_bed_file(geno_file))
  if (!file.exists(bim_file)) stop("Missing BIM file: ", bim_file)

  bim <- read.table(bim_file, header = FALSE, stringsAsFactors = FALSE)
  list(
    snp_ids = as.character(bim[[2]]),
    ref_alleles = stats::setNames(as.character(bim[[5]]), as.character(bim[[2]]))
  )
}

compute_pc_summary <- function(pc_scores, k) {
  pc_cols <- grep("^PC", colnames(pc_scores), value = TRUE)
  pc_cols <- pc_cols[seq_len(min(k, length(pc_cols)))]
  pc_mat <- as.matrix(pc_scores[, pc_cols, drop = FALSE])
  dimnames(pc_mat) <- NULL
  pc_cov <- stats::cov(pc_mat)
  dimnames(pc_cov) <- NULL

  list(
    pc_cov = pc_cov,
    pc_mean = unname(colMeans(pc_mat)),
    pc_median = unname(apply(pc_mat, 2, median))
  )
}

parse_bool <- function(x) {
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y")
}

parse_list_arg <- function(x) {
  x <- trimws(as.character(x))
  if (!nzchar(x)) return(character())
  parts <- unlist(strsplit(gsub("[\\[\\]\\(\\)]", "", x), "[,[:space:]]+"))
  parts[nzchar(parts)]
}

path_without_rds_ext <- function(path_value) {
  if (grepl("\\.rds$", path_value)) return(sub("\\.rds$", "", path_value))
  path_value
}

rds_txt_path <- function(path_value) {
  paste0(path_without_rds_ext(path_value), ".txt")
}

file_base_no_ext <- function(path_value) {
  sub("\\.[^.]*$", "", basename(path_value))
}

resolve_output_path <- function(output, fallback) {
  out <- if (!is.null(output) && nzchar(output)) output else fallback
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  file.path(normalizePath(dirname(out), mustWork = TRUE), basename(out))
}

require_column <- function(dat, col, context) {
  if (!col %in% colnames(dat)) stop(sprintf("%s column '%s' is missing from phenoFile", context, col))
}

filter_pheno_by_pops <- function(pheno, pop_col, label_col, pops) {
  if (length(pops) == 0L) return(pheno)

  pop_col <- if (nzchar(pop_col)) pop_col else "pop"
  label_col <- if (nzchar(label_col)) label_col else pop_col
  require_column(pheno, pop_col, "Population")
  require_column(pheno, label_col, "Label")
  pheno[pheno[[pop_col]] %in% pops | pheno[[label_col]] %in% pops, , drop = FALSE]
}

filter_small_populations <- function(pca, pop_col, min_pop_size) {
  require_column(pca, pop_col, "Population")
  pop_counts <- table(pca[[pop_col]])
  pop_filter <- names(pop_counts)[pop_counts < min_pop_size]
  if (length(pop_filter) > 0L) {
    warning(paste(pop_filter, collapse = ";"), " these ", length(pop_filter),
            " population will be removed due to having less than ", min_pop_size,
            " samples in data.")
    pca <- pca[!pca[[pop_col]] %in% pop_filter, , drop = FALSE]
  }
  pca
}

compute_group_pc_summary <- function(pc_scores, pop_col, k) {
  require_column(pc_scores, pop_col, "Population")
  pc_cols <- grep("^PC", colnames(pc_scores), value = TRUE)
  pc_cols <- pc_cols[seq_len(min(k, length(pc_cols)))]
  pop_group <- split(pc_scores[, pc_cols, drop = FALSE], pc_scores[[pop_col]])
  list(
    pc_cov = lapply(pop_group, stats::cov),
    pc_mean = lapply(pop_group, function(x) sapply(x, mean)),
    pc_median = lapply(pop_group, function(x) sapply(x, median))
  )
}

first_existing_path <- function(path_value) {
  if (is.null(path_value)) return(NULL)
  path_value <- as.character(path_value)
  if (!nzchar(path_value) || identical(path_value, "NA")) return(NULL)
  path_value
}

default_output <- function(input_file, cwd, suffix) {
  file.path(cwd, paste0(sub("\\.rds$", "", basename(input_file)), suffix))
}

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
    library(dplyr)
    library(tibble)
  })

  if (is.null(opt$genoFile)) stop("--genoFile is required for flashpca")

  bed_file <- normalizePath(resolve_bed_file(opt$genoFile), mustWork = TRUE)
  pheno_file <- normalizePath(resolve_pheno_file(opt, bed_file), mustWork = TRUE)
  pops <- parse_list_arg(opt$pops)
  out_prefix <- file.path(opt$cwd, file_base_no_ext(pheno_file))
  rds_out <- resolve_output_path(opt$output, paste0(out_prefix, ".pca.rds"))
  homogeneous <- parse_bool(opt$homogeneous) || !nzchar(opt$`pop-col`)

  cat(sprintf("Running flashPCA on: %s\n", bed_file))
  cat(sprintf("Using phenoFile: %s\n", pheno_file))
  cat(sprintf("Using stand: %s\n", opt$stand))
  cat(sprintf("Writing flashPCA RDS: %s\n", rds_out))

  # Workaround for flashpcaR path bug - copy files with .bed.fam name
  bed_dir <- dirname(bed_file)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(bed_dir)
  bed_base <- basename(bed_file)
  fam_src <- sub("\\.bed$", ".fam", bed_file)
  bim_src <- sub("\\.bed$", ".bim", bed_file)
  fam_dst <- file.path(bed_dir, paste0(bed_base, ".fam"))
  bim_dst <- file.path(bed_dir, paste0(bed_base, ".bim"))
  if (file.access(bed_dir, 2) == 0) {
    if (file.exists(fam_src) && !file.exists(fam_dst)) {
      file.copy(fam_src, fam_dst, overwrite = TRUE)
    }
    if (file.exists(bim_src) && !file.exists(bim_dst)) {
      file.copy(bim_src, bim_dst, overwrite = TRUE)
    }
  }
  cat(sprintf("Running flashPCA on: %s\n", bed_base))
    # flashpcaR adds .bed extension automatically, so pass name without extension
  bed_base_noext <- sub("\\.bed$", "", bed_base)
  result <- flashpca(bed_base_noext,
                     ndim   = opt$k,
                     stand  = opt$stand,
                     do_loadings = TRUE,
                     check_geno = TRUE)
  bim_reference <- read_bim_reference(bed_file)
  if (nrow(result$loadings) == length(bim_reference$snp_ids)) {
    rownames(result$loadings) <- bim_reference$snp_ids
  }

  pca <- as.data.frame(result$projection, check.names = FALSE)
  pca <- tibble::rownames_to_column(pca, "ID")
  colnames(pca) <- c("ID", paste0("PC", seq_len(ncol(pca) - 1L)))

  pheno <- read_pheno_metadata(pheno_file)
  pheno <- filter_pheno_by_pops(pheno, opt$`pop-col`, opt$`label-col`, pops)

  pca <- merge(pheno, pca, by = "ID", all = FALSE)
  if (nzchar(opt$`pop-col`)) {
    pca <- filter_small_populations(pca, opt$`pop-col`, opt$`min-pop-size`)
  } else {
    pca$pop <- 1
  }

  if (homogeneous) {
    summary_mat <- result$projection[, seq_len(min(opt$`maha-k`, ncol(result$projection))), drop = FALSE]
    pc_cov <- stats::cov(summary_mat)
    dimnames(pc_cov) <- NULL
    pc_summary <- list(
      pc_cov = pc_cov,
      pc_mean = apply(summary_mat, 2, mean),
      pc_median = apply(summary_mat, 2, median)
    )
  } else {
    pc_summary <- compute_group_pc_summary(pca, opt$`pop-col`, opt$`maha-k`)
  }

  saveRDS(list(
    pca_model = result,
    pc_scores = pca,
    meta      = paste0(file_base_no_ext(pheno_file), " ", paste(pops, collapse = "_")),
    pc_cov    = pc_summary$pc_cov,
    pc_mean   = pc_summary$pc_mean,
    pc_median = pc_summary$pc_median
  ), rds_out)

  write.table(pca, rds_txt_path(rds_out),
              sep = "\t", quote = FALSE, row.names = FALSE)

  cat(sprintf("PCA RDS saved: %s\n", rds_out))
}

# ── Step: project_samples ────────────────────────────────────────────────────
run_project_samples <- function(opt) {
  suppressPackageStartupMessages({
    library(flashpcaR)
    library(dplyr)
    library(tibble)
  })

  if (is.null(opt$genoFile))
    stop("--genoFile is required for project_samples")
  if (is.null(opt$`pca-model`))
    stop("--pca-model is required for project_samples")

  cat(sprintf("Projecting: %s\n", opt$genoFile))
  cat(sprintf("Using PCA model: %s\n", opt$`pca-model`))
  if (nzchar(opt$stand)) {
    cat(sprintf("Projection standardization inherited from PCA model; --stand received: %s\n", opt$stand))
  }

  pca_obj <- readRDS(opt$`pca-model`)
  bed_file <- normalizePath(resolve_bed_file(opt$genoFile), mustWork = TRUE)
  bed_prefix <- sub("\\.bed$", "", bed_file)
  pheno_file <- normalizePath(resolve_pheno_file(opt, bed_file), mustWork = TRUE)
  rds_out <- resolve_output_path(
    opt$output,
    file.path(opt$cwd, paste0(file_base_no_ext(pheno_file), ".pca.projected.rds"))
  )
  cat(sprintf("Using phenoFile: %s\n", pheno_file))
  cat(sprintf("Writing projected PCA RDS: %s\n", rds_out))
  f <- pca_obj$pca_model
  bim_reference <- read_bim_reference(bed_file)
  overlapped_variants <- match(bim_reference$snp_ids, rownames(f$loadings))
  if (anyNA(overlapped_variants)) {
    stop("Input variants do not fully overlap with PCA loadings")
  }

  projected <- project(bed_prefix,
                       loadings = f$loadings[overlapped_variants, , drop = FALSE],
                       orig_mean = f$center[overlapped_variants],
                       orig_sd = f$scale[overlapped_variants],
                       ref_allele = bim_reference$ref_alleles)

  pca <- as.data.frame(projected$projection, check.names = FALSE)
  pca <- tibble::rownames_to_column(pca, "ID")
  colnames(pca) <- c("ID", paste0("PC", seq_len(ncol(pca) - 1L)))

  pheno <- read_pheno_metadata(pheno_file)

  pca <- merge(pheno, pca, by = "ID", all = FALSE)
  pca_obj$pc_scores <- bind_rows(pca_obj$pc_scores, pca)

  saveRDS(pca_obj, rds_out)

  write.table(pca_obj$pc_scores, rds_txt_path(rds_out),
              sep = "\t", quote = FALSE, row.names = FALSE)

  cat(sprintf("Projected PCA RDS saved: %s\n", rds_out))
}

run_plot_pca <- function(opt) {
  suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(gridExtra)
    library(matrixStats)
    library(readr)
  })

  plot_data <- if (!is.null(opt$`plot-data`)) opt$`plot-data` else opt$`pca-result`
  if (is.null(plot_data)) stop("--plot-data is required for plot_pca")

  dat <- readRDS(plot_data)
  f <- dat$pca_model
  pca_final <- dat$pc_scores
  pop_col <- if (nzchar(opt$`pop-col`)) opt$`pop-col` else "pop"
  label_col <- if (nzchar(opt$`label-col`)) opt$`label-col` else pop_col
  pops <- parse_list_arg(opt$pops)
  if (length(pops) > 1 && pop_col %in% colnames(pca_final) && label_col %in% colnames(pca_final)) {
    pca_final <- pca_final %>%
      filter(.data[[pop_col]] %in% pops | .data[[label_col]] %in% pops)
  }
  if (label_col %in% colnames(pca_final)) {
    pca_final[[label_col]] <- as.character(pca_final[[label_col]])
  }

  k <- min(opt$k, length(grep("^PC", colnames(pca_final))))
  if (k < 2L) stop("plot_pca requires at least 2 PC columns")

  colors_40 <- c("#a9a9a9", "#2f4f4f", "#556b2f", "#a0522d", "#7f0000", "#006400", "#808000", "#483d8b", "#3cb371", "#bdb76b", "#4682b4", "#9acd32",
                 "#20b2aa", "#00008b", "#32cd32", "#daa520", "#7f007f", "#b03060", "#ff0000", "#ff8c00", "#ffff00", "#0000cd", "#00ff00", "#9400d3",
                 "#00fa9a", "#00ffff", "#00bfff", "#f4a460", "#f08080", "#adff2f", "#ff6347", "#ff00ff", "#1e90ff", "#dda0dd", "#7b68ee", "#afeeee",
                 "#ee82ee", "#ff69b4", "#ffe4c4", "#ffc0cb")
  colors_20 <- c("#2f4f4f", "#2e8b57", "#8b0000", "#808000", "#00008b", "#ff0000", "#ff8c00", "#00ff00", "#4169e1", "#00ffff", "#00bfff", "#0000ff",
                 "#da70d6", "#d8bfd8", "#ff00ff", "#eee8aa", "#ffff54", "#ff1493", "#ffa07a", "#98fb98")
  set.seed(999)
  num_col <- if (label_col %in% colnames(pca_final)) length(unique(pca_final[[label_col]])) else 1L
  color_list <- sample(if (num_col <= 20) colors_20 else colors_40)[seq_len(num_col)]

  min_axis <- parse_list_arg(opt$`min-axis`)
  max_axis <- parse_list_arg(opt$`max-axis`)
  if (length(min_axis) == 0L || length(max_axis) == 0L) {
    proj_df <- as.data.frame(f$projection)
    num_proj <- proj_df[vapply(proj_df, is.numeric, logical(1))]
    min_axis <- round(colMins(as.matrix(num_proj)), 1)
    max_axis <- round(colMaxs(as.matrix(num_proj)), 1)
  } else {
    min_axis <- as.double(min_axis)
    max_axis <- as.double(max_axis)
  }
  axis_min <- min(min_axis, na.rm = TRUE)
  axis_max <- max(max_axis, na.rm = TRUE)

  outliers <- NULL
  outlier_file <- first_existing_path(opt$`outlier-file`)
  if (!is.null(outlier_file) && file.exists(outlier_file)) {
    outliers <- read.table(outlier_file, col.names = c("FID", "IID"), stringsAsFactors = FALSE)
  }

  plot_pcs <- function(df, x, y, title = "") {
    base <- ggplot(df, aes_string(x = x, y = y)) +
      labs(title = title, x = x, y = y) +
      scale_y_continuous(limits = c(axis_min, axis_max)) +
      scale_x_continuous(limits = c(axis_min, axis_max)) +
      theme_classic()
    if (label_col %in% colnames(df)) {
      base <- base +
        geom_point(aes_string(color = label_col)) +
        scale_color_manual(values = color_list)
    } else {
      base <- base + geom_point()
    }
    if (!is.null(outliers)) {
      overlay <- dplyr::filter(df, IID %in% outliers$IID, FID %in% outliers$FID)
      if (nrow(overlay) > 0) {
        base <- base +
          geom_point(data = overlay, shape = 21, size = 1.5, color = "red", stroke = 0.9)
        if (label_col %in% colnames(df)) {
          base <- base + geom_point(data = overlay, aes_string(color = label_col), shape = 16, size = 1)
        } else {
          base <- base + geom_point(data = overlay, shape = 16, size = 1)
        }
      }
    }
    base
  }

  pc_plot_out <- if (nzchar(opt$`output-pc-plot`)) opt$`output-pc-plot` else default_output(plot_data, opt$cwd, ".pc.png")
  scree_plot_out <- if (nzchar(opt$`output-scree-plot`)) opt$`output-scree-plot` else default_output(plot_data, opt$cwd, ".scree.png")
  scree_text_out <- if (nzchar(opt$`output-scree-text`)) opt$`output-scree-text` else default_output(plot_data, opt$cwd, ".scree.txt")
  for (out_path in c(pc_plot_out, scree_plot_out, scree_text_out)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  }

  unit <- 4
  n_col <- min(4, k)
  n_row <- ceiling((k - 1) / n_col)
  plots <- lapply(seq_len(k - 1L), function(i) {
    plot_pcs(pca_final, paste0("PC", i), paste0("PC", i + 1L), dat$meta)
  })
  png(pc_plot_out, width = unit * n_col, height = unit * n_row, unit = "in", res = 300)
  do.call(gridExtra::grid.arrange, c(plots, list(ncol = n_col, nrow = n_row)))
  invisible(dev.off())

  pve <- round(f$values / sum(f$values), 2)
  pve_cum <- cumsum(pve) / sum(pve)
  pve_df <- data.frame(PCs = seq_along(pve), PVE = pve, PVE_cum = pve_cum)
  pve_plot <- ggplot(pve_df, aes(x = PCs, y = PVE)) +
    geom_line() + xlab("Principal Component") + ylab("PVE") + ggtitle("Scree Plot") +
    ylim(0, 1) + theme_classic()
  cum_plot <- ggplot(pve_df, aes(x = PCs, y = cumsum(PVE))) +
    geom_line() + xlab("Principal Component") + ylab("PVE") + ggtitle("Cumulative PVE Plot") +
    ylim(0, 1) + theme_classic()
  png(scree_plot_out, width = 8, height = 4, unit = "in", res = 300)
  gridExtra::grid.arrange(pve_plot, cum_plot, nrow = 1)
  invisible(dev.off())
  readr::write_delim(pve_df, scree_text_out, "\t")
}

run_detect_outliers <- function(opt) {
  suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
  })

  pca_result <- if (!is.null(opt$`pca-result`)) opt$`pca-result` else opt$`plot-data`
  if (is.null(pca_result)) stop("--pca-result is required for detect_outliers")

  robust <- parse_bool(opt$robust)
  pop_col <- if (nzchar(opt$`pop-col`)) opt$`pop-col` else "pop"
  dat <- readRDS(pca_result)

  robust_inv <- function(s) {
    tryCatch(solve(s), error = function(cond) MASS::ginv(s))
  }

  calc_mahalanobis_dist <- function(x, m, s, name = "", prob = opt$prob) {
    pc <- x %>% select("IID", "FID", all_of(pop_col), starts_with("PC"))
    mu_pc <- pc[, 4:(4 + length(m) - 1), drop = FALSE]
    pc$mahal <- mahalanobis(mu_pc, m, robust_inv(s), inverted = TRUE)
    pc$p <- pchisq(pc$mahal, df = nrow(s), lower.tail = FALSE)
    cutoff <- quantile(pc$mahal, probs = prob)
    outliers <- pc[(pc$mahal > cutoff & pc$p < opt$pval), , drop = FALSE]
    d_summary <- paste0(capture.output(summary(pc$mahal)), collapse = "\n")
    msg <- paste(
      "#", name, "result summary\n## Mahalanobis distance summary:\n```\n",
      d_summary, "\n```\n",
      paste("The cut-off for outlier removal is set to:", cutoff,
            "and the number of individuals to remove is:", nrow(outliers), "\n"),
      paste("The new sample size after outlier removal is:", nrow(pc) - nrow(outliers), "\n")
    )
    outliers <- outliers %>% select(FID, IID)
    list(pc = pc, manh_dis_sq_cutoff = cutoff, msg = msg, outliers = outliers)
  }

  center_key <- if (robust) "pc_median" else "pc_mean"
  if (is.list(dat$pc_mean)) {
    pops <- names(dat$pc_mean)
    pop_group <- split(dat$pc_scores, f = dat$pc_scores[[pop_col]])
    group_results <- lapply(pops, function(p) {
      calc_mahalanobis_dist(pop_group[[p]], dat[[center_key]][[p]], dat$pc_cov[[p]], name = paste(dat$meta, p))
    })
    names(group_results) <- pops
    res <- list(
      msg = do.call(paste, c(lapply(pops, function(p) group_results[[p]]$msg), sep = "\n")),
      manh_dis_sq_cutoff = cbind(pops, sapply(pops, function(p) group_results[[p]]$manh_dis_sq_cutoff)),
      outliers = do.call(rbind, c(lapply(pops, function(p) group_results[[p]]$outliers))),
      pc = do.call(rbind, c(lapply(pops, function(p) group_results[[p]]$pc)))
    )
  } else {
    res <- calc_mahalanobis_dist(dat$pc_scores, dat[[center_key]], dat$pc_cov, name = dat$meta)
  }

  base_prefix <- sub("\\.rds$", "", pca_result)
  distance_out <- if (nzchar(opt$`distance-output`)) opt$`distance-output` else paste0(base_prefix, ".mahalanobis.rds")
  outliers_out <- if (nzchar(opt$`identified-outliers-output`)) opt$`identified-outliers-output` else paste0(base_prefix, ".outliers")
  summary_out <- if (nzchar(opt$`analysis-summary-output`)) opt$`analysis-summary-output` else paste0(base_prefix, ".analysis_summary.md")
  qq_out <- if (nzchar(opt$`qqplot-output`)) opt$`qqplot-output` else paste0(base_prefix, ".mahalanobis_qq.png")
  hist_out <- if (nzchar(opt$`hist-output`)) opt$`hist-output` else paste0(base_prefix, ".mahalanobis_hist.png")
  for (out_path in c(distance_out, outliers_out, summary_out, qq_out, hist_out)) {
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  }

  writeLines(c("---", "theme: base-theme", "style: |", "  img {", "    height: 80%;", "    display: block;", "    margin-left: auto;", "    margin-right: auto;", "  }", "---", "", res$msg), summary_out)

  summary_k <- if (is.list(dat$pc_mean)) length(dat[[center_key]][[1]]) else length(dat[[center_key]])
  png(qq_out, width = 4, height = 4, unit = "in", res = 300)
  qqplot(qchisq(ppoints(100), df = summary_k), res$pc$mahal,
         main = expression("Mahalanobis" * ~D^2 * " vs. quantiles of" * ~ chi[k]^2),
         xlab = expression(chi[2]^2 * ", probability points = 100"),
         ylab = expression(D^2), pch = 16)
  abline(0, 1, col = "red")
  invisible(dev.off())

  png(hist_out, width = 4, height = 4, unit = "in", res = 300)
  print(
    ggplot(res$pc, aes(x = mahal)) +
      geom_histogram(aes(y = after_stat(count)), binwidth = 0.5, colour = "#1F3552", fill = "#4271AE") +
      scale_x_continuous(name = "Mahalanobis distance") +
      theme_classic()
  )
  invisible(dev.off())

  saveRDS(res, distance_out)
  write.table(res$outliers, outliers_out, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch(opt$step,
  flashpca        = run_flashpca(opt),
  project_samples = run_project_samples(opt),
  plot_pca        = run_plot_pca(opt),
  detect_outliers = run_detect_outliers(opt),
  stop(sprintf("Unknown step '%s'. Available: flashpca, project_samples, plot_pca, detect_outliers", opt$step))
)
