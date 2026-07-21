#!/usr/bin/env Rscript
# mash_posterior_contrast_summary.R
#
# Summarize posterior-contrast significance across regions -- the
# `mash_posterior_contrast_2` step of mash_posterior.ipynb. For each pairwise
# condition contrast, counts significant SNP-level contrasts (p < --p-cutoff)
# and the number of features (regions) with any significant SNP, over all the
# per-region contrast RDS files. Writes a 4 x n_pairs CSV that
# mash_posterior_contrast_plot.R renders.
#
# Inputs:
#   --contrast f1 [...]  per-region contrast RDS files (from mash_posterior_contrast.R).
#   --cells c1,c2,...     conditions (columns) whose pairwise contrasts to tally.
#   --p-cutoff X         significance cutoff. Default 1e-5.
#   --output <CSV>       output summary CSV (rows: n_sig_snp / n_snp /
#                        n_sig_feature / n_all_feature; cols: pairwise contrasts).

suppressPackageStartupMessages({
  library(argparser)
})

p <- arg_parser("Summarize posterior-contrast significance across regions")
p <- add_argument(p, "--contrast", type = "character", nargs = Inf,
                  help = "per-region contrast RDS files")
p <- add_argument(p, "--cells", type = "character",
                  help = "comma-separated conditions")
p <- add_argument(p, "--p-cutoff", type = "numeric", default = 1e-5,
                  help = "significance cutoff")
p <- add_argument(p, "--output", type = "character", help = "output summary CSV")
argv <- parse_args(p)

cells <- trimws(strsplit(argv$cells, ",", fixed = TRUE)[[1L]])
files <- as.character(argv$contrast)
if (length(files) == 0L) stop("--contrast requires at least one RDS file.")

pairs <- apply(utils::combn(cells, 2L), 2L, paste, collapse = "_vs_")
summary <- matrix(0, nrow = 4L, ncol = length(pairs),
                  dimnames = list(c("n_sig_snp", "n_snp",
                                    "n_sig_feature", "n_all_feature"), pairs))

crs <- lapply(files, function(f) as.data.frame(readRDS(f)))

for (pair in pairs) {
  for (cr in crs) {
    pcol <- grep(paste0("p_contrast_", pair), names(cr), value = TRUE, fixed = FALSE)
    # exact pairwise column (avoid deviation / substring collisions)
    pcol <- pcol[pcol == paste0("p_contrast_", pair)]
    if (length(pcol) == 0L) next
    pv <- as.numeric(cr[[pcol[1L]]])
    pv <- pv[!is.na(pv)]
    if (length(pv) == 0L) next
    nSig <- sum(pv < argv$p_cutoff)
    summary["n_sig_snp", pair]     <- summary["n_sig_snp", pair] + nSig
    summary["n_snp", pair]         <- summary["n_snp", pair] + length(pv)
    summary["n_sig_feature", pair] <- summary["n_sig_feature", pair] + (nSig > 0)
    summary["n_all_feature", pair] <- summary["n_all_feature", pair] + 1L
  }
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
utils::write.csv(as.data.frame(summary), argv$output)
cat(sprintf("Wrote contrast summary (%d pairwise contrasts over %d region(s)) to %s\n",
            length(pairs), length(files), argv$output))
