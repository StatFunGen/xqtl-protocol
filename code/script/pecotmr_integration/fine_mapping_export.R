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
                       help = "Which view to export: topLoci | cs | pip | marginals",
                       type = "character", default = "topLoci")
parser <- add_argument(parser, "--signal-cutoff",
                       help = "PIP cutoff for topLoci/pip exports",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--output",
                       help = "Output TSV path", type = "character")
argv <- parse_args(parser)

view <- match.arg(argv$view, c("topLoci", "cs", "pip", "marginals"))
inputs <- as.character(argv$input)
if (length(inputs) == 0L)
  stop("--input requires at least one RDS path.")

# Pull the row's identifier columns regardless of QTL vs GWAS shape; the
# union of slots produces a NA-padded data frame the downstream rbind can
# concatenate across mixed inputs.
.idCols <- function(fmr, i) {
  list(
    study     = as.character(fmr$study)[[i]],
    context   = if ("context"   %in% names(fmr)) as.character(fmr$context)[[i]]   else NA_character_,
    trait     = if ("trait"     %in% names(fmr)) as.character(fmr$trait)[[i]]     else NA_character_,
    region_id = if ("region_id" %in% names(fmr)) as.character(fmr$region_id)[[i]] else NA_character_,
    method    = as.character(fmr$method)[[i]],
    source    = basename(attr(fmr, ".source", exact = TRUE) %||% ""))
}
`%||%` <- function(a, b) if (is.null(a)) b else a

.extract <- function(entry, view, cutoff) {
  if (view == "topLoci") {
    df <- as.data.frame(getTopLoci(entry, signalCutoff = cutoff))
  } else if (view == "cs") {
    df <- as.data.frame(getCs(entry))
  } else if (view == "pip") {
    pip <- as.numeric(getPip(entry))
    ids <- as.character(getVariantIds(entry))
    df  <- data.frame(variant_id = ids, pip = pip,
                       stringsAsFactors = FALSE)
    if (cutoff > 0) df <- df[df$pip > cutoff, , drop = FALSE]
  } else {  # marginals
    df <- as.data.frame(getMarginalEffects(entry))
  }
  if (nrow(df) == 0L) return(NULL)
  df
}

pieces <- list()
for (path in inputs) {
  fmr <- readRDS(path)
  if (!methods::is(fmr, "FineMappingResultBase")) {
    warning("Skipping non-FineMappingResult input: ", path)
    next
  }
  attr(fmr, ".source") <- path
  for (i in seq_len(nrow(fmr))) {
    entry <- fmr$entry[[i]]
    inner <- tryCatch(.extract(entry, view, argv$signal_cutoff),
                      error = function(e) {
                        message("Entry ", i, " of ", basename(path),
                                ": ", conditionMessage(e))
                        NULL
                      })
    if (is.null(inner)) next
    ids <- .idCols(fmr, i)
    ids$source <- basename(path)
    for (k in names(ids)) inner[[k]] <- ids[[k]]
    pieces[[length(pieces) + 1L]] <- inner
  }
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
