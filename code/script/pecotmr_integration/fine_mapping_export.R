#!/usr/bin/env Rscript
# fine_mapping_export.R
#
# Export a view (topLoci / credibleSets / pip / marginals) of one or more
# QtlFineMappingResult / GwasFineMappingResult RDSes as a single concatenated
# TSV. Adds identifier columns derived from the FMR row (study, context,
# trait, method for QTL; study, method, region_id for GWAS) so a downstream
# consumer can split the table back per-tuple.
#
# Inputs:
#   --input <RDS> [<RDS> ...]  One or more FineMappingResult RDS paths.
#   --view {topLoci|cs|pip|marginals}  Which per-entry view to export.
#                                        Default "topLoci".
#   --signal-cutoff  PIP threshold for topLoci/pip exports. Default 0
#                    (no filter). Pass 0.025 to mirror the legacy
#                    susie default.
#   --output <TSV>  Output TSV path (gzipped if path ends in .gz).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Export a FineMappingResult view as a TSV")
parser <- add_argument(parser, "--input",
                       help = "One or more FineMappingResult RDS paths",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--view",
                       help = "topLoci | cs | cs_summary | pip | marginals | credible_band | affected_regions",
                       type = "character", default = "topLoci")
parser <- add_argument(parser, "--signal-cutoff",
                       help = "PIP cutoff for topLoci/pip exports",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--min-purity",
                       help = "Optional CS purity (min.abs.corr) cutoff for topLoci/cs (independent of coverage/pip); omit = no filter",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--output",
                       help = "Output TSV path", type = "character")
argv <- parse_args(parser)

view <- match.arg(argv$view, c("topLoci", "cs", "cs_summary", "pip", "marginals",
                               "lbf", "credible_band", "affected_regions"))
inputs <- as.character(argv$input)
if (length(inputs) == 0L)
  stop("--input requires at least one RDS path.")
minPurity <- if (is.na(argv$min_purity)) NULL else argv$min_purity

# Each view maps to a collection-level pecotmr accessor that already aggregates
# every entry into one tidy table carrying the row identity columns (study /
# context / trait / region_id / method) alongside the per-variant columns. The
# wrapper only tags each row with its source RDS and concatenates across inputs.
view_fn <- switch(view,
  topLoci   = function(fmr)
    as.data.frame(getTopLoci(fmr, signalCutoff = argv$signal_cutoff,
                             minPurity = minPurity)),
  cs        = function(fmr) as.data.frame(getCs(fmr, minPurity = minPurity)),
  # cs_summary: one row per credible set (size / purity / V / logBF / lead).
  cs_summary = function(fmr) as.data.frame(getCredibleSetSummary(fmr)),
  # lbf: wide variant x effect log Bayes factor matrix (lbf_L1..lbf_LL).
  lbf       = function(fmr) as.data.frame(getLbf(fmr)),
  marginals = function(fmr) as.data.frame(getMarginalEffects(fmr)),
  # fSuSiE functional views (require untrimmed fits; degrade to empty otherwise).
  # credible_band = fitted effect curve + band; affected_regions = GRanges of
  # the intervals where a CS's band excludes zero (coerced to a flat table).
  credible_band    = function(fmr) as.data.frame(fsusieCredibleBand(fmr)),
  affected_regions = function(fmr) as.data.frame(fsusieAffectedRegions(fmr)),
  # pip view: the (identity + variant_id + pip) projection of topLoci.
  pip       = function(fmr) {
    tl <- as.data.frame(getTopLoci(fmr, signalCutoff = argv$signal_cutoff))
    tl[, intersect(c("study", "context", "trait", "region_id", "method",
                     "variant_id", "pip"), names(tl)), drop = FALSE]
  })

pieces <- list()
for (path in inputs) {
  fmr <- readRDS(path)
  if (!methods::is(fmr, "FineMappingResultBase")) {
    warning("Skipping non-FineMappingResult input: ", path)
    next
  }
  df <- tryCatch(view_fn(fmr),
                 error = function(e) {
                   message(basename(path), ": ", conditionMessage(e))
                   NULL
                 })
  if (is.null(df) || nrow(df) == 0L) next
  df$source <- basename(path)
  pieces[[length(pieces) + 1L]] <- df
}
if (length(pieces) == 0L) {
  message("No rows produced; writing an empty TSV with the id-column ",
          "header so downstream consumers don't error on missing files.")
  out <- data.frame(study = character(0), context = character(0),
                    trait = character(0), region_id = character(0),
                    method = character(0), source = character(0),
                    stringsAsFactors = FALSE)
} else {
  # Pad missing columns across pieces so rbind doesn't lose rows.
  all_cols <- unique(unlist(lapply(pieces, names)))
  for (k in seq_along(pieces)) {
    miss <- setdiff(all_cols, names(pieces[[k]]))
    for (m in miss) pieces[[k]][[m]] <- NA
    pieces[[k]] <- pieces[[k]][, all_cols, drop = FALSE]
  }
  out <- do.call(rbind, pieces)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
conn <- if (grepl("\\.gz$", argv$output)) {
  gzfile(argv$output, "w")
} else {
  file(argv$output, "w")
}
write.table(out, file = conn, sep = "\t", quote = FALSE,
            row.names = FALSE, na = "")
close(conn)
cat(sprintf("Wrote %s view (%d rows from %d input RDS file(s)) to %s\n",
            view, nrow(out), length(inputs), argv$output))
