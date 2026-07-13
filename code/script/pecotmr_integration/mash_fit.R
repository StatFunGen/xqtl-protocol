#!/usr/bin/env Rscript
# mash_fit.R
#
# Fit a MASH mixture model -- the `[mash_1]` step of mash_fit.ipynb. Reads the
# pre-partitioned MASH input (random.b / random.s effect-size matrices), a
# residual correlation (Vhat), and a pre-computed prior covariance list ($U),
# wraps the random subset as a beta-scale QtlSumStats (no LD reference needed --
# mash operates across conditions per variant), and calls pecotmr::mashModelFit.
#
# The mixture weights are learned on the `random` subset (Urbut et al. 2019);
# the resulting model is consumed by mash_posterior.R on the strong / target set.
#
# Inputs:
#   --data          MASH input RDS: list(random.b, random.s, strong.b, strong.s)
#                   -- Bhat / Shat matrices (variants x conditions).
#   --vhat-data     Residual correlation (Vhat) RDS (conditions x conditions).
#   --prior-data    Prior RDS carrying a `$U` covariance list (from the
#                   mixture-prior step).
#   --effect-model  "EE" (beta scale, alpha = 0) or "EZ" (z scale, alpha = 1).
#   --output-level  mashr outputlevel forwarded to mash(). Default 4.
#   --output        Output model RDS (list(mash_model, vhat_file, prior_file)).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Fit a MASH mixture model via pecotmr::mashModelFit")
p <- add_argument(p, "--data", type = "character",
                  help = "MASH input RDS (random.b/random.s/strong.b/strong.s)")
p <- add_argument(p, "--vhat-data", type = "character",
                  help = "Residual correlation (Vhat) RDS")
p <- add_argument(p, "--prior-data", type = "character",
                  help = "Prior RDS carrying a $U covariance list")
p <- add_argument(p, "--effect-model", type = "character", default = "EE",
                  help = "EE (beta, alpha=0) or EZ (z, alpha=1)")
p <- add_argument(p, "--output-level", type = "integer", default = 4L,
                  help = "mashr outputlevel")
p <- add_argument(p, "--output", type = "character", help = "Output model RDS")
argv <- parse_args(p)

alpha <- if (toupper(argv$effect_model) == "EZ") 1 else 0

dat  <- readRDS(argv$data)
vhat <- readRDS(argv$vhat_data)
prior <- readRDS(argv$prior_data)
U <- if (is.list(prior) && !is.null(prior$U)) prior$U else prior

# Wrap the random (fit) subset as a beta-scale QtlSumStats. ldSketch stays NULL:
# mash needs no LD reference.
random <- qtlSumStatsFromBetaMatrix(
  as.matrix(dat$random.b), as.matrix(dat$random.s), study = "mash")

model <- mashModelFit(list(random = random), alpha = alpha,
                      priorCovariances = U, vhat = vhat,
                      fitOn = "random", outputLevel = argv$output_level)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(list(mash_model = model,
             vhat_file  = argv$vhat_data,
             prior_file = argv$prior_data), argv$output)
cat(sprintf("Wrote MASH model (%d prior components) to %s\n",
            length(U), argv$output))
