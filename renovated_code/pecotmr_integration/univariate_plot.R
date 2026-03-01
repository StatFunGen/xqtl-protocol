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

res <- readRDS(opt$input)
png(opt$output, width = 14, height = 6, unit = "in", res = 300)
par(mfrow = c(1, 2))
susieR::susie_plot(res, y = "PIP",
    pos = list(attr = "pos", start = res$pos[1], end = res$pos[length(res$pos)]),
    add_legend = TRUE, xlab = "position")
susieR::susie_plot(res, y = "z",
    pos = list(attr = "pos", start = res$pos[1], end = res$pos[length(res$pos)]),
    add_legend = TRUE, xlab = "position", ylab = "-log10(p)")
dev.off()
