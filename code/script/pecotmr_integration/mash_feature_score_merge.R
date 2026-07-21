#!/usr/bin/env Rscript
# mash_feature_score_merge.R
#
# Merge per-chunk feature-score tables (from mash_feature_score.R) into one
# table -- the shared `_2` merge step of the feature-score workflows in
# mash_posterior.ipynb. All chunks share the long format
# (gene, condition, [contrast], score, scoreType), so the merge is a row-bind.
#
# Inputs:
#   --scores f1 [f2 ...]  per-chunk feature-score RDS files.
#   --output <path>       output table; ".csv" -> CSV, else RDS.

suppressPackageStartupMessages(library(argparser))

p <- arg_parser("Merge per-chunk MASH feature-score tables")
p <- add_argument(p, "--scores", type = "character", nargs = Inf,
                  help = "per-chunk feature-score RDS files")
p <- add_argument(p, "--output", type = "character", help = "output CSV or RDS")
argv <- parse_args(p)

files <- as.character(argv$scores)
if (length(files) == 0L) stop("--scores requires at least one RDS file.")

merged <- do.call(rbind, lapply(files, function(f) as.data.frame(readRDS(f))))
if (is.null(merged)) merged <- data.frame()

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
if (grepl("\\.csv$", argv$output)) {
  utils::write.csv(merged, argv$output, row.names = FALSE)
} else {
  saveRDS(merged, argv$output, compress = "xz")
}
cat(sprintf("Merged %d chunk(s) -> %d rows to %s\n",
            length(files), nrow(merged), argv$output))
