# pecotmr_aliases.R
#
# Shared name aliases for the pecotmr_integration wrapper scripts. Each script
# sources this from its own directory:
#
#   .d <- dirname(sub("^--file=", "",
#           grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
#   source(file.path(.d, "pecotmr_aliases.R"))
#
# pecotmr 0.8.2 camelCased its MASH component/Vhat names and its TWAS method
# tokens. The snake_case spellings remain the public interface on this side --
# they are SoS step names in mixture_prior.ipynb (`[flash_nonneg]`,
# `[vhat_simple_specific]`), they appear in output filenames that downstream
# steps consume, and they are the documented `--twas-methods` values in
# mnm_regression.ipynb -- so the wrappers keep accepting them and translate
# here. Both spellings are accepted, so callers already using the pecotmr
# names work unchanged.
#
# This is argument / return-value marshalling only; no analysis logic (that
# lives in pecotmr).

# mashCovarianceComponents(components=) / mashPriorCovariances(components=)
MASH_COMPONENT_ALIASES <- c(flash_nonneg = "flashNonneg")

# mashResidualCorrelation(method=)
MASH_VHAT_ALIASES <- c(simple_specific = "simpleSpecific")

# mashPriorCovariances(engine=). 0.9.0 renamed the engines camelCase AND folded
# udr's ED-vs-TED choice out of the engine name into the options object, so the
# three names this side exposes (and mixture_prior.ipynb's [ud] /
# [ud_unconstrained] / [ed_bovy] steps pass) no longer map one-to-one:
#
#   cov_ed  -> "covEd"    mashr's exported cov_ed(), unchanged
#   ud      -> covUdr + unconstrained.update = "ed"
#   ud_ted  -> covUdr + unconstrained.update = "ted"
#
# Both udr engines pin the update explicitly. Left unset, 0.9.0 hands the
# choice to udr, which picks TED on z-scale data and ED on the beta scale --
# correct on its own terms, but it would silently merge the `ud` and `ud_ted`
# steps into the same fit on z-scale input, and those are two different
# documented engines on this side.
mash_prior_engine <- function(value) {
  switch(value,
    cov_ed = "covEd",
    ud     = CovUdrOptions(fit = UdFitOptions(unconstrained.update = "ed")),
    ud_ted = CovUdrOptions(fit = UdFitOptions(unconstrained.update = "ted")),
    # Already a pecotmr spelling (or an unknown one): hand it over untouched
    # and let pecotmr report it.
    value)
}

# fineMappingPipeline(methods=). Its token set is susie / susieInf / susieAsh /
# ser / mvsusie / fsusie -- only the two multiword ones need translating.
FM_METHOD_ALIASES <- c(
  susie_ash = "susieAsh",
  susie_inf = "susieInf"
)

# twasWeightsPipeline(methods=). Mirrors pecotmr's
# .twasKnownMethodLookupNames(); tokens that are already single words
# (susie, mrash, enet, lasso, scad, mcp, l0learn, mvsusie, mrmash) need no alias.
TWAS_METHOD_ALIASES <- c(
  susie_ash          = "susieAsh",
  susie_inf          = "susieInf",
  bayes_r            = "bayesR",
  bayes_l            = "bayesL",
  bayes_a            = "bayesA",
  bayes_b            = "bayesB",
  bayes_c            = "bayesC",
  bayes_n            = "bayesN",
  b_lasso            = "bLasso",
  dpr_vb             = "dprVb",
  dpr_gibbs          = "dprGibbs",
  dpr_adaptive_gibbs = "dprAdaptiveGibbs"
)

# Map any aliased names to the spelling pecotmr expects, passing others through
# unchanged so pecotmr reports unknown values itself.
apply_pecotmr_aliases <- function(values, aliases) {
  hit <- match(values, names(aliases))
  ifelse(is.na(hit), values, unname(aliases[hit]))
}

# Split a comma/space separated CLI list and normalise the names in it.
split_pecotmr_names <- function(value, aliases, split = "[ ,]+") {
  parts <- trimws(strsplit(value, split)[[1L]])
  parts <- parts[nzchar(parts)]
  apply_pecotmr_aliases(parts, aliases)
}

# pecotmr 0.9.0 returns the top-loci tables (topLoci(), overlapTopLoci()) as
# GRanges rather than data.frames. Every consumer here wants the flat
# per-variant table, and as.data.frame() on a GRanges PREPENDS five range
# columns (seqnames/start/end/width/strand) that are not part of the published
# schema -- chrom and pos are already mcols. mcols() takes the data columns
# only, so the frame keeps exactly the shape it had before 0.9.0.
#
# Only for those variant tables: a GRanges whose RANGES are the data
# (fsusieAffectedRegions()) must keep going through as.data.frame().
as_variant_df <- function(x) {
  if (methods::is(x, "GenomicRanges")) return(as.data.frame(S4Vectors::mcols(x)))
  as.data.frame(x)
}

# ---- summaryStatsQc() argument routing -------------------------------------
#
# pecotmr 0.9.0 split summaryStatsQc()'s flat argument list into four bundles.
# This side still speaks the flat names: they are the wrapper flags, and
# --qc-args is a documented JSON passthrough whose own examples
# ({"mafCutoff":0.01,"nCutoff":0,"alleleFlipKriging":true}, rss_analysis.ipynb)
# span three different destinations. So route each flat name to whichever
# bundle now owns it, and leave everything else at top level -- where
# alleleFlipKriging / effectiveN / impute / skipRegion / keepVariants /
# matchMinProp still live, and where summaryStatsQc reports an unknown name
# itself rather than this table swallowing it.
SUMSTATS_QC_RENAMES <- c(zMismatchQc = "ldMismatchQcMethod")

SUMSTATS_QC_BUNDLE_CTOR <- c(
  sumstatsFilterArgs   = "SumstatsFilterParam",
  panelFilterParam     = "PanelFilterParam",
  signalScreenArgs     = "SignalScreenParam",
  sumstatsCleaningArgs = "SumstatsCleaningParam")

# flat name -> "<bundle>/<field in that bundle>"
SUMSTATS_QC_ROUTING <- c(
  removeIndels          = "sumstatsFilterArgs/removeIndels",
  removeStrandAmbiguous = "sumstatsFilterArgs/removeStrandAmbiguous",
  infoCutoff            = "sumstatsFilterArgs/infoCutoff",
  nCutoff               = "sumstatsFilterArgs/nCutoff",
  mafCutoff             = "panelFilterParam/mafCutoff",
  macCutoff             = "panelFilterParam/macCutoff",
  imissCutoff           = "panelFilterParam/imissCutoff",
  # the *CutoffToSkip screens became one bundle, one field per metric
  pipCutoffToSkip       = "signalScreenArgs/pip",
  absZCutoffToSkip      = "signalScreenArgs/absZ",
  bfCutoffToSkip        = "signalScreenArgs/bf",
  logBfCutoffToSkip     = "signalScreenArgs/logBf",
  coerceNumeric         = "sumstatsCleaningArgs/coerceNumeric",
  normalizeChr          = "sumstatsCleaningArgs/normalizeChr",
  dropNonstandardChr    = "sumstatsCleaningArgs/dropNonstandardChr",
  dropMissData          = "sumstatsCleaningArgs/dropMissData",
  dropPOutOfRange       = "sumstatsCleaningArgs/dropPOutOfRange",
  clampSmallP           = "sumstatsCleaningArgs/clampSmallP",
  smallPFloor           = "sumstatsCleaningArgs/smallPFloor",
  dropZeroEffect        = "sumstatsCleaningArgs/dropZeroEffect",
  dropNonpositiveSe     = "sumstatsCleaningArgs/dropNonpositiveSe")

# Turn a flat named list of summaryStatsQc() arguments into the 0.9.0 call
# shape: the routed ones grouped into their bundle constructors, the rest
# passed straight through.
summary_stats_qc_args <- function(flat) {
  nms <- names(flat)
  hit <- match(nms, names(SUMSTATS_QC_RENAMES))
  names(flat) <- ifelse(is.na(hit), nms, unname(SUMSTATS_QC_RENAMES[hit]))

  out <- list()
  bundled <- list()
  for (nm in names(flat)) {
    # [ ] not [[ ]]: on a named character vector a missing name is an
    # out-of-bounds ERROR with [[ ]], but NA with [ ].
    route <- unname(SUMSTATS_QC_ROUTING[nm])
    if (is.na(route)) {
      out[nm] <- list(flat[[nm]])     # out[nm] <- list(x) keeps a NULL value
      next
    }
    part <- strsplit(route, "/", fixed = TRUE)[[1L]]
    bundled[[part[[1L]]]][[part[[2L]]]] <- flat[[nm]]
  }
  for (b in names(bundled))
    out[[b]] <- do.call(SUMSTATS_QC_BUNDLE_CTOR[[b]], bundled[[b]])
  out
}

