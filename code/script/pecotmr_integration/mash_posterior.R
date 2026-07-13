#!/usr/bin/env Rscript
# mash_posterior.R
#
# Compute MASH posterior matrices for a target set given a fitted model --
# the `[mash_2]` step of mash_fit.ipynb (posterior on the strong subset) and the
# per-analysis-unit `posterior_1` step of mash_posterior.ipynb. Wraps the target
# Bhat / Shat as a beta-scale QtlSumStats (no LD reference needed) and calls
# pecotmr::mashPosterior.
#
# Inputs:
#   --data           RDS carrying the target Bhat / Shat matrices.
#   --bhat-key       List element for the target Bhat. Default "strong.b".
#   --shat-key       List element for the target Shat. Default "strong.s".
#   --vhat-data      Residual correlation (Vhat) RDS.
#   --mash-model     Model RDS from mash_fit.R (uses its $mash_model element).
#   --effect-model   "EE" (alpha=0) or "EZ" (alpha=1).
#   --exclude-condition  Comma/space-separated condition names OR 1-based column
#                    indices to drop before computing posteriors. Default none.
#   --no-posterior-cov   Omit the full posterior covariance array.
#   --output         Output posterior RDS.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Compute MASH posterior matrices via pecotmr::mashPosterior")
p <- add_argument(p, "--data", type = "character",
                  help = "RDS with the target Bhat/Shat matrices")
p <- add_argument(p, "--bhat-key", type = "character", default = "strong.b",
                  help = "list element for the target Bhat")
p <- add_argument(p, "--shat-key", type = "character", default = "strong.s",
                  help = "list element for the target Shat")
p <- add_argument(p, "--vhat-data", type = "character",
                  help = "Residual correlation (Vhat) RDS")
p <- add_argument(p, "--mash-model", type = "character",
                  help = "Model RDS from mash_fit.R (uses $mash_model)")
p <- add_argument(p, "--effect-model", type = "character", default = "EE",
                  help = "EE (alpha=0) or EZ (alpha=1)")
p <- add_argument(p, "--exclude-condition", type = "character", default = "",
                  help = "condition names or 1-based indices to drop")
p <- add_argument(p, "--no-posterior-cov", flag = TRUE,
                  help = "omit the full posterior covariance array")
p <- add_argument(p, "--output", type = "character", help = "Output posterior RDS")
argv <- parse_args(p)

alpha <- if (toupper(argv$effect_model) == "EZ") 1 else 0

dat  <- readRDS(argv$data)
vhat <- readRDS(argv$vhat_data)
mm   <- readRDS(argv$mash_model)
model <- if (is.list(mm) && !is.null(mm$mash_model)) mm$mash_model else mm

bhat <- as.matrix(dat[[argv$bhat_key]])
shat <- as.matrix(dat[[argv$shat_key]])
target <- qtlSumStatsFromBetaMatrix(bhat, shat, study = "mash")
conds  <- colnames(bhat)

# Resolve --exclude-condition tokens: numeric tokens are 1-based column indices
# (the legacy mash_posterior CLI form), everything else is a condition name.
exTok <- trimws(strsplit(argv$exclude_condition, "[ ,]+")[[1L]])
exTok <- exTok[nzchar(exTok)]
exclude <- character(0)
if (length(exTok) > 0L) {
  idxLike <- grepl("^[0-9]+$", exTok)
  byIdx   <- suppressWarnings(as.integer(exTok[idxLike]))
  byIdx   <- byIdx[byIdx >= 1L & byIdx <= length(conds)]
  exclude <- unique(c(conds[byIdx], exTok[!idxLike]))
}

post <- mashPosterior(model, target, alpha = alpha, vhat = vhat,
                      excludeCondition = exclude,
                      outputPosteriorCov = !argv$no_posterior_cov)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(post, argv$output)
cat(sprintf("Wrote MASH posterior (%d variants x %d conditions) to %s\n",
            nrow(post$PosteriorMean), ncol(post$PosteriorMean), argv$output))
