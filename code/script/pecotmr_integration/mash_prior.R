#!/usr/bin/env Rscript
# mash_prior.R
#
# Estimate the refined MASH prior (covariance list U + mixture weights) via
# pecotmr::mashPriorCovariances -- the unified backing for mixture_prior.ipynb's
# prior-engine steps. Each step fixes a single `--engine`:
#
#   --engine cov_ed   mashr extreme deconvolution (the exported bovy ED;
#                     the ed_bovy step). Weights from a final mash() fit.
#   --engine ud       udr ED update (the ud step). OPT-IN (numerical issues).
#   --engine ud_ted   udr TED update (the ud_unconstrained step). Needs i.i.d.
#                     (z-scale) data.
#
# mashPriorCovariances builds the covariance components (canonical / pca / flash
# / flash_nonneg) internally and refines them with the chosen engine; the
# residual correlation (Vhat) enters here (unlike the component step).
#
# Inputs:
#   --data          MASH input RDS with strong.b / strong.s matrices.
#   --engine        cov_ed | ud | ud_ted. Default cov_ed.
#   --effect-model  "EE" (alpha=0) or "EZ" (alpha=1).
#   --vhat-data     Residual correlation (Vhat) RDS. Optional (default identity).
#   --components    Comma-separated covariance components to build.
#                   Default "canonical,pca,flash,flash_nonneg".
#   --npc           PCs for cov_pca. Default ncol(Bhat) - 1.
#   --seed          RNG seed. Default 999.
#   --output        Output prior RDS (list(U, w, loglik)).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Estimate the refined MASH prior (U, weights) ")
p <- add_argument(p, "--data", type = "character",
                  help = "MASH input RDS (strong.b / strong.s)")
p <- add_argument(p, "--engine", type = "character", default = "cov_ed",
                  help = "cov_ed | ud | ud_ted")
p <- add_argument(p, "--effect-model", type = "character", default = "EE",
                  help = "EE (alpha=0) or EZ (alpha=1)")
p <- add_argument(p, "--vhat-data", type = "character", default = "",
                  help = "residual correlation (Vhat) RDS")
p <- add_argument(p, "--components", type = "character",
                  default = "canonical,pca,flash,flash_nonneg",
                  help = "comma-separated covariance components")
p <- add_argument(p, "--npc", type = "integer", default = NA_integer_,
                  help = "PCs for cov_pca (default ncol(Bhat) - 1)")
p <- add_argument(p, "--component-files", type = "character", default = "",
                  help = "comma-separated pre-built covariance-component RDSes to refine (the mixture-prior pipeline: the per-method component steps built these); overrides --components/--npc")
p <- add_argument(p, "--seed", type = "integer", default = 999L,
                  help = "RNG seed")
p <- add_argument(p, "--output", type = "character",
                  help = "output prior RDS (list(U, w, loglik))")
argv <- parse_args(p)

alpha <- if (toupper(argv$effect_model) == "EZ") 1 else 0
dat <- readRDS(argv$data)

strong <- qtlSumStatsFromBetaMatrix(
  as.matrix(dat$strong.b), as.matrix(dat$strong.s), study = "mash")

vhat <- if (nzchar(argv$vhat_data)) readRDS(argv$vhat_data) else NULL

if (nzchar(argv$component_files)) {
  # Pipeline mode: refine the components the per-method steps already built.
  files <- trimws(strsplit(argv$component_files, "[ ,]+")[[1L]])
  files <- files[nzchar(files)]
  priorComponents <- do.call(c, lapply(files, readRDS))
  prior <- mashPriorCovariances(list(strong = strong), alpha = alpha, vhat = vhat,
                                priorComponents = priorComponents,
                                engine = argv$engine, setSeed = argv$seed)
} else {
  # Self-contained mode: build the components here.
  components <- trimws(strsplit(argv$components, "[ ,]+")[[1L]])
  components <- components[nzchar(components)]
  nPcs <- if (is.na(argv$npc)) NULL else argv$npc
  prior <- mashPriorCovariances(list(strong = strong), alpha = alpha, vhat = vhat,
                                components = components, engine = argv$engine,
                                nPcs = nPcs, setSeed = argv$seed)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(prior, argv$output)
cat(sprintf("Wrote MASH prior [engine=%s] (%d U components) to %s\n",
            argv$engine, length(prior$U), argv$output))
