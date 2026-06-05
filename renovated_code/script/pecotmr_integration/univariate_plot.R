#!/usr/bin/env Rscript
# ============================================================
# univariate_plot.R
# Mirrors: pipeline/rss_analysis.ipynb [univariate_plot]
#
# Generate PIP and z-score plots for a univariate RSS fine-mapping result.
#
# Usage:
#   Rscript univariate_plot.R --input res.rds --output plot.png
# ============================================================

suppressPackageStartupMessages(library(optparse))

opt_list <- list(
  make_option("--input",  type = "character", default = NULL,
              help = "Input RDS file produced by univariate_rss step"),
  make_option("--output", type = "character", default = NULL,
              help = "Output PNG file path")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$input))  stop("--input is required")
if (is.null(opt$output)) stop("--output is required")

extract_variant_positions <- function(ids) {
  if (length(ids) == 0) return(numeric(0))
  out <- rep(NA_real_, length(ids))

  colon_match <- regexec("^[^:]+:([0-9]+):", ids)
  colon_parts <- regmatches(ids, colon_match)
  colon_idx <- lengths(colon_parts) >= 2
  out[colon_idx] <- as.numeric(vapply(colon_parts[colon_idx], `[`, character(1), 2))

  underscore_idx <- is.na(out)
  if (any(underscore_idx)) {
    under_ids <- ids[underscore_idx]
    under_match <- regexec("^[^_]+_([0-9]+)_", under_ids)
    under_parts <- regmatches(under_ids, under_match)
    valid_under <- lengths(under_parts) >= 2
    under_values <- rep(NA_real_, length(under_ids))
    under_values[valid_under] <- as.numeric(vapply(under_parts[valid_under], `[`, character(1), 2))
    out[underscore_idx] <- under_values
  }

  out
}

extract_first_result_leaf <- function(obj, path = character()) {
  if (inherits(obj, "susie")) {
    return(list(label = paste(path, collapse = "/"), entry = list(susie_result_trimmed = obj)))
  }

  if (!is.list(obj) || length(obj) == 0) {
    return(NULL)
  }

  if (!is.null(obj$susie_result_trimmed) && inherits(obj$susie_result_trimmed, "susie")) {
    return(list(label = paste(path, collapse = "/"), entry = obj))
  }

  child_names <- names(obj)
  if (is.null(child_names)) {
    child_names <- as.character(seq_along(obj))
  }

  for (i in seq_along(obj)) {
    res <- extract_first_result_leaf(obj[[i]], c(path, child_names[[i]]))
    if (!is.null(res)) {
      return(res)
    }
  }

  NULL
}

extract_sumstats_z <- function(sumstats_obj) {
  if (is.null(sumstats_obj)) {
    return(NULL)
  }

  if (is.data.frame(sumstats_obj)) {
    if ("z" %in% names(sumstats_obj)) return(sumstats_obj$z)
    if ("z_scores" %in% names(sumstats_obj)) return(sumstats_obj$z_scores)
  }

  if (is.list(sumstats_obj) && length(sumstats_obj) > 0) {
    if (!is.null(sumstats_obj$z)) return(sumstats_obj$z)
    if (!is.null(sumstats_obj$z_scores)) return(sumstats_obj$z_scores)

    for (elt in sumstats_obj) {
      z <- extract_sumstats_z(elt)
      if (!is.null(z)) return(z)
    }
  }

  NULL
}

build_plot_payload <- function(obj) {
  leaf <- extract_first_result_leaf(obj)
  if (is.null(leaf)) {
    stop("Could not find a nested `susie_result_trimmed` object in input RDS")
  }

  entry <- leaf$entry
  susie_fit <- entry$susie_result_trimmed
  variant_ids <- entry$variant_names
  if (is.null(variant_ids) && !is.null(entry$top_loci$variant_id)) {
    variant_ids <- entry$top_loci$variant_id
  }

  pip_pos <- extract_variant_positions(variant_ids)
  pip_df <- NULL
  if (!is.null(susie_fit$pip) && length(pip_pos) == length(susie_fit$pip) && any(!is.na(pip_pos))) {
    pip_df <- data.frame(pos = pip_pos, value = as.numeric(susie_fit$pip))
    pip_df <- pip_df[!is.na(pip_df$pos), , drop = FALSE]
  }

  z_vec <- extract_sumstats_z(entry$sumstats)
  z_df <- NULL
  if (!is.null(z_vec) && length(pip_pos) == length(z_vec) && any(!is.na(pip_pos))) {
    z_df <- data.frame(pos = pip_pos, value = abs(as.numeric(z_vec)))
    z_df <- z_df[!is.na(z_df$pos), , drop = FALSE]
  } else if (!is.null(entry$top_loci) && all(c("variant_id", "z") %in% names(entry$top_loci))) {
    top_pos <- extract_variant_positions(entry$top_loci$variant_id)
    z_df <- data.frame(pos = top_pos, value = abs(as.numeric(entry$top_loci$z)))
    z_df <- z_df[!is.na(z_df$pos), , drop = FALSE]
  }

  list(label = leaf$label, pip = pip_df, z = z_df)
}

plot_metric <- function(df, title, ylab, col) {
  if (is.null(df) || nrow(df) == 0) {
    plot.new()
    title(main = title)
    text(0.5, 0.5, "No plottable data", cex = 1.1)
    return(invisible(NULL))
  }

  ord <- order(df$pos)
  plot(
    df$pos[ord],
    df$value[ord],
    pch = 16,
    cex = 0.6,
    col = col,
    xlab = "position",
    ylab = ylab,
    main = title
  )
}

payload <- build_plot_payload(readRDS(opt$input))
label <- if (nzchar(payload$label)) payload$label else basename(opt$input)
dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)
png(opt$output, width = 14, height = 6, unit = "in", res = 300)
par(mfrow = c(1, 2))
plot_metric(payload$pip, sprintf("PIP: %s", basename(opt$input)), "PIP", "#1f77b4")
plot_metric(payload$z, sprintf("Abs(z): %s", label), "|z|", "#444444")
dev.off()
