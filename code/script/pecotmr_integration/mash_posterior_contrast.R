#!/usr/bin/env Rscript
# mash_posterior_contrast.R
#
# Per-region posterior contrasts -- the `mash_posterior_contrast_1` step of
# mash_posterior.ipynb. Reads a mash posterior (PosteriorMean + PosteriorCov,
# from mash_posterior.R) and the original effect matrix, builds the condition
# grouping vector, and calls pecotmr::mashPosteriorContrast (which maps
# fitMashContrast over every variant: deviation + pairwise contrasts).
#
# Inputs:
#   --posterior <RDS>    Posterior RDS carrying PosteriorMean + PosteriorCov.
#   --orig-data <RDS>    RDS with the original effect matrix used for the
#                        posterior (same variant rows / order as the posterior).
#   --orig-key           List element holding the effect matrix. Default "bhat"
#                        (set to "" if the RDS is a bare matrix).
#   --cells c1,c2,...     Condition order used to seed the grouping vector.
#                        Default: the posterior's columns.
#   --group1/2/3 c,...    Comma-separated condition groups -- replicate
#                        populations of one cell type share contrast weight.
#   --grouping-recipe f  File of comma-separated groups (one group per line);
#                        overrides --group1/2/3.
#   --output <RDS>       Output contrast RDS (variants x contrasts).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Per-region mash posterior contrasts via mashPosteriorContrast")
p <- add_argument(p, "--posterior", type = "character",
                  help = "posterior RDS (PosteriorMean + PosteriorCov)")
p <- add_argument(p, "--orig-data", type = "character",
                  help = "RDS with the original effect matrix")
p <- add_argument(p, "--orig-key", type = "character", default = "bhat",
                  help = "list element for the effect matrix ('' = bare matrix)")
p <- add_argument(p, "--cells", type = "character", default = "",
                  help = "comma-separated condition order for grouping")
p <- add_argument(p, "--group1", type = "character", default = "",
                  help = "comma-separated condition group 1")
p <- add_argument(p, "--group2", type = "character", default = "",
                  help = "comma-separated condition group 2")
p <- add_argument(p, "--group3", type = "character", default = "",
                  help = "comma-separated condition group 3")
p <- add_argument(p, "--grouping-recipe", type = "character", default = "",
                  help = "file of comma-separated groups, one per line")
p <- add_argument(p, "--output", type = "character", help = "output contrast RDS")
argv <- parse_args(p)

splitCsv <- function(x) if (nzchar(x)) trimws(strsplit(x, ",", fixed = TRUE)[[1L]]) else character(0)

post <- readRDS(argv$posterior)
pm <- as.matrix(post$PosteriorMean)
pv <- post$PosteriorCov
orig <- readRDS(argv$orig_data)
if (nzchar(argv$orig_key)) orig <- orig[[argv$orig_key]]
orig <- as.matrix(orig)

# Grouping vector: 0 = independent condition; positive ints tie replicate
# populations of one cell type together (see mashPosteriorContrast/fitMashContrast).
cells <- splitCsv(argv$cells)
if (length(cells) == 0L) cells <- colnames(pm)
grouping <- setNames(rep(0L, length(cells)), cells)
groups <- if (nzchar(argv$grouping_recipe)) {
  lapply(readLines(argv$grouping_recipe), function(g) trimws(strsplit(g, ",")[[1L]]))
} else {
  Filter(length, list(splitCsv(argv$group1), splitCsv(argv$group2), splitCsv(argv$group3)))
}
for (i in seq_along(groups)) grouping[intersect(groups[[i]], names(grouping))] <- i

cr <- mashPosteriorContrast(pm, pv, orig,
                            grouping = if (any(grouping > 0L)) grouping else NULL)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(cr, argv$output, compress = "xz")
cat(sprintf("Wrote posterior contrasts (%d variants x %d contrasts) to %s\n",
            nrow(cr), ncol(cr), argv$output))
