#!/usr/bin/env Rscript
# legacy_twas_weights_convert.R
#
# Bridge for migrating legacy TWAS analyses to the new pecotmr S4 path: convert
# a legacy `*.univariate_twas_weights.rds` (the nested list a legacy
# load_twas_weights() consumed: gene > context > {twas_weights, twas_cv_result,
# susie_weights_intermediate, variant_names, region_info, ...}) into:
#
#   * a pecotmr::TwasWeights S4 RDS  (one row per (study, context, trait,
#     method); cvPerformance carried over so causalInferencePipeline's
#     rsqCutoff weight-selection works), and
#   * optionally a pecotmr::QtlFineMappingResult S4 RDS built from the SuSiE
#     intermediate fit, so causalInferencePipeline can run MR.
#
# The pair is the input contract of twas.R (--twas-weights / --fine-mapping-result).
#
# CAVEAT (FMR): the legacy SuSiE intermediate fit carries pip + the coefficient
# (used as the per-variant QTL effect, topLoci$posterior_mean) but NO marginal
# standard error, so topLoci$posterior_sd is a magnitude-scaled placeholder. MR
# is only run by causalInferencePipeline for TWAS-significant tuples
# (mrPvalCutoff gate), so this placeholder only matters for genes that pass it.
#
# Inputs:
#   --legacy       Legacy univariate_twas_weights.rds
#   --study        Study identifier for the emitted rows
#   --output       Output TwasWeights RDS
#   --output-fmr   Optional output QtlFineMappingResult RDS (from the SuSiE fit)

suppressPackageStartupMessages({ library(argparser); library(pecotmr) })

p <- arg_parser("Convert a legacy univariate_twas_weights.rds to pecotmr S4")
p <- add_argument(p, "--legacy", type = "character",
                  help = "legacy univariate_twas_weights.rds")
p <- add_argument(p, "--study", type = "character", default = "study",
                  help = "study identifier for the emitted rows")
p <- add_argument(p, "--output", type = "character",
                  help = "output TwasWeights RDS")
p <- add_argument(p, "--output-fmr", type = "character", default = "",
                  help = "optional output QtlFineMappingResult RDS (from the SuSiE fit)")
argv <- parse_args(p)

legacy <- readRDS(argv$legacy)

# Legacy per-method CV performance (1 x 6 matrix: corr,rsq,adj_rsq,pval,RMSE,MAE)
# -> the new cvPerformance shape list(metrics = named vector).
perfToCv <- function(perf) {
  if (is.null(perf)) return(NULL)
  v   <- as.numeric(perf)
  nms <- colnames(perf)
  if (is.null(nms) || length(nms) != length(v))
    nms <- c("corr", "rsq", "adj_rsq", "pval", "RMSE", "MAE")[seq_along(v)]
  list(metrics = stats::setNames(v, nms))
}

# ---- TwasWeights -----------------------------------------------------------
rs <- character(0); rc <- character(0); rt <- character(0)
rm <- character(0); entries <- list()
for (trait in names(legacy)) {
  for (ctxKey in names(legacy[[trait]])) {
    inner   <- legacy[[trait]][[ctxKey]]
    context <- sub(paste0("_", trait, "$"), "", ctxKey)   # bulk_rnaseq_ENSG... -> bulk_rnaseq
    perfL   <- inner$twas_cv_result$performance
    for (wnm in names(inner$twas_weights)) {
      tok  <- sub("_weights$", "", wnm)                   # susie_weights -> susie
      wmat <- inner$twas_weights[[wnm]]
      vids <- if (!is.null(rownames(wmat))) rownames(wmat) else inner$variant_names
      wvec <- stats::setNames(as.numeric(wmat), vids)
      cv   <- perfToCv(perfL[[paste0(tok, "_performance")]])
      entries[[length(entries) + 1L]] <- TwasWeightsEntry(
        variantIds = vids, weights = wvec, cvResult = cv,
        standardized = FALSE, dataType = context)
      rs <- c(rs, argv$study); rc <- c(rc, context)
      rt <- c(rt, trait);      rm <- c(rm, tok)
    }
  }
}
tw <- TwasWeights(study = rs, context = rc, trait = rt, method = rm, entry = entries)
saveRDS(tw, argv$output)
cat(sprintf("Wrote TwasWeights (%d rows: %s) to %s\n",
            nrow(tw), paste(unique(rm), collapse = ","), argv$output))

# ---- QtlFineMappingResult from the SuSiE intermediate fit (optional) -------
if (nzchar(argv$output_fmr)) {
  fs <- character(0); fc <- character(0); ft <- character(0)
  fm <- character(0); fe <- list()
  for (trait in names(legacy)) {
    for (ctxKey in names(legacy[[trait]])) {
      inner   <- legacy[[trait]][[ctxKey]]
      context <- sub(paste0("_", trait, "$"), "", ctxKey)
      s <- inner$susie_weights_intermediate
      if (is.null(s) || is.null(s$pip)) next
      vids <- names(s$pip)
      sw   <- inner$twas_weights[["susie_weights"]]
      bx   <- as.numeric(sw)[match(vids, rownames(sw))]; bx[is.na(bx)] <- 0
      sx   <- pmax(abs(bx) * 0.5, 0.05)   # placeholder SE (legacy fit has none)
      # getTopLoci()/.projectPosteriorView re-derives variant_id from
      # chrom:pos:A1:A2 and sources beta<-posterior_mean, se<-posterior_sd, so
      # supply those identity + effect columns under the canonical names.
      vp <- strsplit(vids, ":", fixed = TRUE)
      g  <- function(i) vapply(vp, function(x)
              if (length(x) >= i) x[[i]] else NA_character_, character(1))
      topLoci <- data.frame(
        variant_id = vids, chrom = g(1L),
        pos = suppressWarnings(as.integer(g(2L))), A1 = g(3L), A2 = g(4L),
        pip = as.numeric(s$pip), posterior_mean = bx, posterior_sd = sx,
        stringsAsFactors = FALSE)
      fe[[length(fe) + 1L]] <- FineMappingEntry(
        variantIds = vids, susieFit = s, topLoci = topLoci)
      fs <- c(fs, argv$study); fc <- c(fc, context)
      ft <- c(ft, trait);      fm <- c(fm, "susie")
    }
  }
  fmr <- QtlFineMappingResult(study = fs, context = fc, trait = ft,
                              method = fm, entry = fe)
  saveRDS(fmr, argv$output_fmr)
  cat(sprintf("Wrote QtlFineMappingResult (%d rows) to %s\n",
              nrow(fmr), argv$output_fmr))
}
