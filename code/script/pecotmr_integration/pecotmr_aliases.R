# pecotmr_aliases.R
#
# Shared name aliases for the pecotmr_integration wrapper scripts. Each script
# sources this from its own directory:
#
#   .d <- dirname(sub("^--file=", "",
#           grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
#   source(file.path(.d, "pecotmr_aliases.R"))
#
# pecotmr spells its MASH component / Vhat names and its TWAS method tokens
# camelCase. The snake_case spellings are the public interface on this side --
# they are SoS step names in mixture_prior.ipynb (`[flash_nonneg]`,
# `[vhat_simple_specific]`), they appear in output filenames that downstream
# steps consume, and they are the documented `--twas-methods` values in
# mnm_regression.ipynb -- so the wrappers accept them and translate here. Both
# spellings are accepted, so a caller using the pecotmr names works too.
#
# This is argument / return-value marshalling only; no analysis logic (that
# lives in pecotmr).

# mashCovarianceComponents(components=) / mashPriorCovariances(components=)
MASH_COMPONENT_ALIASES <- c(flash_nonneg = "flashNonneg")

# mashResidualCorrelation(method=)
MASH_VHAT_ALIASES <- c(simple_specific = "simpleSpecific")

# mashPriorCovariances(engine=). udr's ED-vs-TED choice lives in the options
# object rather than in the engine name, so the three names this side exposes
# (and mixture_prior.ipynb's [ud] / [ud_unconstrained] / [ed_bovy] steps pass)
# do not map one-to-one onto engines:
#
#   cov_ed  -> "covEd"    mashr's exported cov_ed(), unchanged
#   ud      -> covUdr + unconstrained.update = "ed"
#   ud_ted  -> covUdr + unconstrained.update = "ted"
#
# Both udr engines pin the update explicitly. Left unset, the choice falls to
# udr, which picks TED on z-scale data and ED on the beta scale --
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

# The top-loci tables (topLoci(), overlapTopLoci()) come back as GRanges.
# Every consumer here wants the flat per-variant table, and as.data.frame() on
# a GRanges PREPENDS five range columns (seqnames/start/end/width/strand) that
# are not part of the published schema -- chrom and pos are already mcols.
# mcols() takes the data columns only, so the frame keeps the published shape.
#
# Only for those variant tables: a GRanges whose RANGES are the data
# (fsusieAffectedRegions()) must keep going through as.data.frame().
as_variant_df <- function(x) {
  if (methods::is(x, "GenomicRanges")) return(as.data.frame(S4Vectors::mcols(x)))
  as.data.frame(x)
}

# ---- summaryStatsQc() argument routing -------------------------------------
#
# summaryStatsQc() takes its arguments in four bundles. This side speaks flat
# names instead: they are the wrapper flags, and
# --qc-args is a documented JSON passthrough whose own examples
# ({"mafCutoff":0.01,"nCutoff":0,"alleleFlipKriging":true}, rss_analysis.ipynb)
# span three different destinations. So route each flat name to whichever
# bundle now owns it, and leave everything else at top level -- where
# alleleFlipKriging / effectiveN / impute / skipRegion / keepVariants /
# matchMinProp still live, and where summaryStatsQc reports an unknown name
# itself rather than this table swallowing it.
SUMSTATS_QC_RENAMES <- c(zMismatchQc = "ldMismatchQcMethod")

SUMSTATS_QC_BUNDLE_CTOR <- c(
  sumstatsFilterParam   = "SumstatsFilterParam",
  panelFilterParam     = "PanelFilterParam",
  signalScreenParam     = "SignalScreenParam",
  sumstatsCleaningParam = "SumstatsCleaningParam")

# flat name -> "<bundle>/<field in that bundle>"
SUMSTATS_QC_ROUTING <- c(
  removeIndels          = "sumstatsFilterParam/removeIndels",
  removeStrandAmbiguous = "sumstatsFilterParam/removeStrandAmbiguous",
  infoCutoff            = "sumstatsFilterParam/infoCutoff",
  nCutoff               = "sumstatsFilterParam/nCutoff",
  mafCutoff             = "panelFilterParam/mafCutoff",
  macCutoff             = "panelFilterParam/macCutoff",
  imissCutoff           = "panelFilterParam/imissCutoff",
  # the *CutoffToSkip screens became one bundle, one field per metric
  pipCutoffToSkip       = "signalScreenParam/pip",
  absZCutoffToSkip      = "signalScreenParam/absZ",
  bfCutoffToSkip        = "signalScreenParam/bf",
  logBfCutoffToSkip     = "signalScreenParam/logBf",
  coerceNumeric         = "sumstatsCleaningParam/coerceNumeric",
  normalizeChr          = "sumstatsCleaningParam/normalizeChr",
  dropNonstandardChr    = "sumstatsCleaningParam/dropNonstandardChr",
  dropMissData          = "sumstatsCleaningParam/dropMissData",
  dropPOutOfRange       = "sumstatsCleaningParam/dropPOutOfRange",
  clampSmallP           = "sumstatsCleaningParam/clampSmallP",
  smallPFloor           = "sumstatsCleaningParam/smallPFloor",
  dropZeroEffect        = "sumstatsCleaningParam/dropZeroEffect",
  dropNonpositiveSe     = "sumstatsCleaningParam/dropNonpositiveSe")

# Turn a flat named list of summaryStatsQc() arguments into the call shape it
# expects: the routed ones grouped into their bundle constructors, the rest
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


# ---- analysis windows ------------------------------------------------------
#
# pecotmr REFUSES a "chr:start-end" string wherever it takes a window (a
# pipeline's `region`, a loader's `which`): a window is a GRanges, or the path
# of a BED file, so that none is misread by one base -- a BED start is 0-based
# where a GRanges is 1-based.
#
# The wrappers' windows come from the CLI, where they are 1-based inclusive
# "chr:start-end" (or the "chr_start_end" spelling that LD-block ids use), so
# that parsing happens here: vectorised, either spelling, no "chr" prefix
# required, chromosome canonicalised (M -> MT, 23/24 -> X/Y).
as_region_granges <- function(regions) {
  if (methods::is(regions, "GenomicRanges")) return(regions)
  if (!is.character(regions) || !length(regions) || any(!nzchar(regions)))
    stop("region must be a non-empty character vector of 'chr:start-end' ",
         "or 'chr_start_end' windows (got ", class(regions)[[1L]], ")")
  parts <- do.call(rbind, strsplit(regions, "[_:-]"))
  if (ncol(parts) != 3L)
    stop("Region format must be 'chr:start-end' or 'chr_start_end', got: ",
         paste(regions, collapse = ", "))
  bare <- toupper(sub("^ch(r)?", "", parts[, 1L], ignore.case = TRUE))
  bare <- sub("^0+(?=[0-9])", "", bare, perl = TRUE)
  bare[bare == "23"] <- "X"
  bare[bare == "24"] <- "Y"
  bare[bare == "M"]  <- "MT"
  start <- suppressWarnings(as.integer(parts[, 2L]))
  end   <- suppressWarnings(as.integer(parts[, 3L]))
  if (anyNA(start) || anyNA(end))
    stop("Region start/end must be integers, got: ",
         paste(regions, collapse = ", "))
  GenomicRanges::GRanges(paste0("chr", bare), IRanges::IRanges(start, end))
}

# ---- parallel workers ------------------------------------------------------
#
# CrossValidationParam() and the cTWAS steps (estCtwasGroupPriors,
# screenCtwasRegions, finemapCtwasRegions, mergeCtwasBoundaryRegions) take a
# BiocParallel `BPPARAM`: the caller picks the backend.
#
# The wrapper flags are integer counts (--cv-threads, --ncore) -- documented
# notebook arguments, and SoS passes a core count -- so translate, keeping
# their meaning that 1 = serial and -1 = every core. SerialParam() for a
# single worker matters beyond speed: it runs the folds in THIS process, so
# set.seed() here reaches them.
as_bpparam <- function(numThreads) {
  n <- suppressWarnings(as.integer(numThreads))
  if (length(n) != 1L || is.na(n) || n == 0L)
    stop("thread count must be a non-zero integer (-1 = all cores), got: ",
         paste(numThreads, collapse = ", "))
  if (n == 1L) return(BiocParallel::SerialParam())
  if (n < 0L)  return(BiocParallel::MulticoreParam())
  BiocParallel::MulticoreParam(workers = n)
}
