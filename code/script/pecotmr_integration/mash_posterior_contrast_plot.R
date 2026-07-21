#!/usr/bin/env Rscript
# mash_posterior_contrast_plot.R
#
# Render the posterior-contrast significance summary as a symmetric heatmap --
# the `mash_posterior_contrast_3` / `posterior_cntrast_plot` step of
# mash_posterior.ipynb. Lower triangle = significant SNP-feature-pair ratio,
# upper triangle = significant-feature ratio, per pairwise condition contrast.
#
# Inputs:
#   --data <CSV>    summary CSV from mash_posterior_contrast_summary.R
#                   (rows n_sig_snp/n_snp/n_sig_feature/n_all_feature).
#   --output <PNG>  output heatmap PNG.

suppressPackageStartupMessages({
  library(argparser)
  library(dplyr)
  library(tidyverse)
  library(ggnewscale)
})

p <- arg_parser("Plot the posterior-contrast significance summary heatmap")
p <- add_argument(p, "--data", type = "character", help = "summary CSV")
p <- add_argument(p, "--output", type = "character", help = "output PNG")
argv <- parse_args(p)

df <- read.csv(argv$data, row.names = 1, check.names = FALSE)

# Normalize each pairwise column name so con1 <= con2 alphabetically.
for (i in seq_len(ncol(df))) {
  parts <- strsplit(colnames(df)[i], "_vs_", fixed = TRUE)[[1L]]
  if (length(parts) == 2L && parts[1] > parts[2])
    colnames(df)[i] <- paste0(parts[2], "_vs_", parts[1])
}

# Two ratios: SNP-feature-pair (n_sig_snp / n_snp) and feature (n_sig / n_all).
snpRatio <- as.data.frame(t(df["n_sig_snp", ] / df["n_snp", ]))
colnames(snpRatio) <- "ratio"; snpRatio$group <- "snp"
fetRatio <- as.data.frame(t(df["n_sig_feature", ] / df["n_all_feature", ]))
rownames(fetRatio) <- vapply(strsplit(rownames(fetRatio), "_vs_", fixed = TRUE),
                             function(x) paste0(x[2], "_vs_", x[1]), character(1))
colnames(fetRatio) <- "ratio"; fetRatio$group <- "feature"
ratio <- rbind(snpRatio, fetRatio)

# Make the grid symmetric by adding the con_vs_con diagonal at 0.
cons <- unique(vapply(strsplit(rownames(ratio), "_vs_", fixed = TRUE),
                      `[`, character(1), 1L))
for (con in cons) ratio[paste0(con, "_vs_", con), ] <- list(0, 0)

ratio$con1 <- vapply(strsplit(rownames(ratio), "_vs_", fixed = TRUE), `[`, character(1), 1L)
ratio$con2 <- vapply(strsplit(rownames(ratio), "_vs_", fixed = TRUE), `[`, character(1), 2L)
ratio$score1 <- ratio$score2 <- 0
ratio$score1[ratio$group == "snp"] <- ratio$ratio[ratio$group == "snp"]
ratio$score2[ratio$group == "feature"] <- ratio$ratio[ratio$group == "feature"]
ratio$label <- paste0(round(ratio$ratio, 4) * 100, "%")
ratio$label[ratio$group == 0] <- NA

numCols <- length(cons)
side <- 4 + numCols * 0.5

pl <- ggplot(ratio[ratio$group == "snp", ], aes(x = con1, y = con2)) +
  geom_tile(aes(fill = score1)) +
  scale_fill_gradient2("SNP_Feature pair", low = "#762A83", mid = "white",
                       high = "#1B7837") +
  new_scale("fill") +
  geom_tile(aes(fill = score2), data = subset(ratio, group != "snp")) +
  scale_fill_gradient2("Feature", low = "#1B7837", mid = "white",
                       high = "#762A83") +
  geom_text(data = ratio, aes(label = label)) +
  theme_bw()

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
ggsave(argv$output, plot = pl, width = side, height = side)
cat(sprintf("Wrote contrast heatmap (%d conditions) to %s\n", numCols, argv$output))
