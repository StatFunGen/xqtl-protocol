#!/usr/bin/env Rscript
# mash_covariance.R
#
# Build data-driven covariance component(s) via pecotmr::mashCovarianceComponents
# -- the unified backing for mixture_prior.ipynb's per-method covariance steps:
# each step fixes a single `--component`, demonstrating that estimator on the
# `strong` subset.
#
#   --component canonical      cov_canonical
#   --component pca            cov_pca
#   --component flash          cov_flash (default factors)
#   --component flash_nonneg   cov_flash(factors = "nonneg")
#
# The covariance components are residual-correlation-independent (they read only
# the standardized effect matrix), so no Vhat is required here -- it enters at the
# prior-refinement step (mash_prior.R).
#
# Inputs:
#   --data          MASH input RDS with strong.b / strong.s matrices.
#   --component     One (or a comma-separated set) of the components above.
#   --effect-model  "EE" (alpha=0) or "EZ" (alpha=1).
#   --npc           PCs for cov_pca. Default ncol(Bhat) - 1.
#   --seed          RNG seed (cov_flash is stochastic). Default 999.
#   --output        Output covariance-component RDS (a named list of matrices).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Build MASH data-driven covariance component(s)")
p <- add_argument(p, "--data", type = "character",
                  help = "MASH input RDS (strong.b / strong.s)")
p <- add_argument(p, "--component", type = "character", default = "canonical",
                  help = "canonical | pca | flash | flash_nonneg (comma-separated for several)")
p <- add_argument(p, "--effect-model", type = "character", default = "EE",
                  help = "EE (alpha=0) or EZ (alpha=1)")
p <- add_argument(p, "--npc", type = "integer", default = NA_integer_,
                  help = "PCs for cov_pca (default ncol(Bhat) - 1)")
p <- add_argument(p, "--seed", type = "integer", default = 999L,
                  help = "RNG seed (cov_flash is stochastic)")
p <- add_argument(p, "--output", type = "character",
                  help = "output covariance-component RDS")
argv <- parse_args(p)

alpha <- if (toupper(argv$effect_model) == "EZ") 1 else 0
dat <- readRDS(argv$data)

strong <- qtlSumStatsFromBetaMatrix(
  as.matrix(dat$strong.b), as.matrix(dat$strong.s), study = "mash")

components <- trimws(strsplit(argv$component, "[ ,]+")[[1L]])
components <- components[nzchar(components)]
nPcs <- if (is.na(argv$npc)) NULL else argv$npc

U <- mashCovarianceComponents(list(strong = strong), alpha = alpha,
                              components = components, nPcs = nPcs,
                              setSeed = argv$seed)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(U, argv$output)
cat(sprintf("Wrote %d covariance matrix/matrices [%s] to %s\n",
            length(U), paste(components, collapse = ","), argv$output))
