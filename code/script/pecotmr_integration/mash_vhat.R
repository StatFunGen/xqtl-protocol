#!/usr/bin/env Rscript
# mash_vhat.R
#
# Learn the MASH residual correlation matrix (Vhat) via
# pecotmr::mashResidualCorrelation. Unified backing for mixture_prior.ipynb's
# `vhat_*` steps: each step fixes a single `--method`, demonstrating that
# estimator on the same input.
#
#   --method identity          diag(nConditions)
#   --method simple            estimate_null_correlation_simple on the null set
#   --method mle               mash_estimate_corr_em on a random subset (needs
#                              --prior-data, the prior $U)
#   --method corshrink         CorShrink adaptive-shrinkage null correlation
#   --method simple_specific   nearPD(cov(nullZ), corr = TRUE)
#
# Inputs:
#   --data          MASH input RDS with strong.b/random.b/null.b (+ .s) matrices.
#   --method        Estimator (see above). Default "simple".
#   --effect-model  "EE" (alpha=0) or "EZ" (alpha=1).
#   --prior-data    Prior RDS carrying a $U list (required by --method mle).
#   --n-subset      mle random-subset size. Default 6000.
#   --max-iter      mle EM iterations. Default 6.
#   --output        Output Vhat RDS (a conditions x conditions matrix).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Estimate the MASH residual correlation (Vhat)")
p <- add_argument(p, "--data", type = "character",
                  help = "MASH input RDS (strong.b/random.b/null.b + .s)")
p <- add_argument(p, "--method", type = "character", default = "simple",
                  help = "identity | simple | mle | corshrink | simple_specific")
p <- add_argument(p, "--effect-model", type = "character", default = "EE",
                  help = "EE (alpha=0) or EZ (alpha=1)")
p <- add_argument(p, "--prior-data", type = "character", default = "",
                  help = "prior RDS with $U (required for --method mle)")
p <- add_argument(p, "--n-subset", type = "integer", default = 6000L,
                  help = "mle random-subset size")
p <- add_argument(p, "--max-iter", type = "integer", default = 6L,
                  help = "mle EM iterations")
p <- add_argument(p, "--output", type = "character", help = "output Vhat RDS")
argv <- parse_args(p)

alpha <- if (toupper(argv$effect_model) == "EZ") 1 else 0
dat <- readRDS(argv$data)

# Wrap whichever partitions are present as beta-scale QtlSumStats (no LD).
mk <- function(bKey, sKey) {
  if (is.null(dat[[bKey]])) return(NULL)
  qtlSumStatsFromBetaMatrix(as.matrix(dat[[bKey]]), as.matrix(dat[[sKey]]),
                            study = "mash")
}
ssl <- Filter(Negate(is.null), list(
  strong = mk("strong.b", "strong.s"),
  random = mk("random.b", "random.s"),
  null   = mk("null.b",   "null.s")))

U <- if (nzchar(argv$prior_data)) {
  pr <- readRDS(argv$prior_data)
  if (is.list(pr) && !is.null(pr$U)) pr$U else pr
} else NULL

vhat <- mashResidualCorrelation(ssl, alpha = alpha, method = argv$method,
                                priorCovariances = U,
                                nSubset = argv$n_subset, maxIter = argv$max_iter)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(vhat, argv$output)
cat(sprintf("Wrote Vhat [method=%s] (%d x %d) to %s\n",
            argv$method, nrow(vhat), ncol(vhat), argv$output))
