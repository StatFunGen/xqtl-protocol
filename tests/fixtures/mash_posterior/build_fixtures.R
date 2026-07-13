#!/usr/bin/env Rscript
# build_fixtures.R
#
# Regenerate the small committed mash_posterior fixtures FROM the MWE mash data
# (reuse the real 8-condition protocol_example.mashr_input.rds rather than
# synthesizing). Run once (checked-in output); rerun if the wrapper input
# contract changes.
#
#   pixi run Rscript tests/fixtures/mash_posterior/build_fixtures.R
#   # override the MWE location with MWE_MASH=/path/to/input/mash
#
# Provenance: reads the MWE mashr_input.rds (strong/random/null Bhat+Shat over
# 8 conditions), fits a self-consistent mash model on the random subset
# (identity Vhat + canonical prior), and computes the strong-set posterior.
#   posterior.rds     list(PosteriorMean, PosteriorCov) on the strong variants
#   orig.rds          list(bhat, sbhat) = the strong effects (aligned to posterior)
#   fine_mapping.rds  strong variant ids + a small credible-set assignment
# The contrast / feature-score math itself is covered by pecotmr testthat.

suppressPackageStartupMessages(library(pecotmr))

here <- dirname(sub("^--file=", "",
                    grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
if (is.na(here) || !nzchar(here)) here <- "tests/fixtures/mash_posterior"

mwe <- path.expand(Sys.getenv("MWE_MASH",
  "~/Downloads/fungen_xqtl/xqtl-protocol/input/mash"))
mi <- readRDS(file.path(mwe, "protocol_example.mashr_input.rds"))

mk <- function(b, s) qtlSumStatsFromBetaMatrix(as.matrix(mi[[b]]),
                                               as.matrix(mi[[s]]), study = "mash")
ssl <- list(strong = mk("strong.b", "strong.s"),
            random = mk("random.b", "random.s"),
            null   = mk("null.b",   "null.s"))
conds <- colnames(mi$strong.b)
vhat <- diag(length(conds)); dimnames(vhat) <- list(conds, conds)

prior <- mashPriorCovariances(ssl, alpha = 0, vhat = vhat, components = "canonical")
model <- mashModelFit(ssl, alpha = 0, priorCovariances = prior$U, vhat = vhat)
post  <- mashPosterior(model, mk("strong.b", "strong.s"), alpha = 0, vhat = vhat,
                       outputPosteriorCov = TRUE)

# The MWE strong matrices carry no variant ids, so mashPosterior synthesizes
# them; use the posterior's row names as the shared variant ids so orig and the
# fine-mapping table align with the contrast output.
vids <- rownames(post$PosteriorMean)
n <- length(vids)

saveRDS(list(PosteriorMean = post$PosteriorMean, PosteriorCov = post$PosteriorCov),
        file.path(here, "posterior.rds"), compress = "xz")

origB <- as.matrix(mi$strong.b); origS <- as.matrix(mi$strong.s)
rownames(origB) <- rownames(origS) <- vids
saveRDS(list(bhat = origB, sbhat = origS), file.path(here, "orig.rds"), compress = "xz")

saveRDS(data.frame(variants = vids,
                   cs_order = c(1L, 1L, rep(0L, n - 2L)),
                   pip = c(0.6, 0.4, rep(0.02, n - 2L)),
                   stringsAsFactors = FALSE),
        file.path(here, "fine_mapping.rds"), compress = "xz")

cat(sprintf("Wrote MWE-derived fixtures (%d strong variants x %d conditions) to %s\n",
            n, length(conds), here))
