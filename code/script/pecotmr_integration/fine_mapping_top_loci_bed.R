#!/usr/bin/env Rscript
# fine_mapping_top_loci_bed.R
#
# Export the top-loci BED for one or more QtlFineMappingResult /
# GwasFineMappingResult RDSes. The schema, the join to the marginal effects,
# the sort and the compression all live in pecotmr::writeBed(); this
# wrapper only resolves the CLI, reads the inputs, and supplies the one
# published column no S4 object can know -- `block_id`, parsed from the source
# filename.
#
# Inputs:
#   --input <RDS> [<RDS> ...]  FineMappingResult RDS paths.
#   --signal-cutoff  PIP cutoff (topLoci). Default 0.025.
#   --min-purity     Optional CS purity (min.abs.corr) cutoff, independent of
#                    coverage/pip. Omit for no purity filter.
#   --output <BED>   Output BED path (bgzipped/gzipped if it ends in .gz).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Export the top-loci BED of FineMappingResult(s)")
parser <- add_argument(parser, "--input", help = "FineMappingResult RDS paths",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--signal-cutoff", help = "PIP cutoff",
                       type = "numeric", default = 0.025)
parser <- add_argument(parser, "--min-purity",
                       help = "CS purity (min.abs.corr) cutoff; omit for none",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--output", help = "Output BED path",
                       type = "character")
argv <- parse_args(parser)

inputs <- as.character(argv$input)
if (length(inputs) == 0L) stop("--input requires at least one RDS path.")
minPurity <- if (is.na(argv$min_purity)) NULL else argv$min_purity

# LD-block id from the RDS filename ({study}.{block_id}.{suffix}.rds), as a
# FALLBACK for an input that does not carry one. NA when the filename has no
# chrN_start_end token.
#
# `block_id` is a first-class published column now: a GwasFineMappingResult
# records `blockId` and writeBed() writes it, so supplying it again through
# extraColumns would be refused as overwriting a published column. It is only
# needed for the QTL side, whose result has no block axis -- there blockId()
# is NA for every row, the column is dropped as empty, and the
# filename-derived value fills it.
.blockIdFromName <- function(bn) {
  m <- regmatches(bn, regexpr("chr[0-9XYMT]+_[0-9]+_[0-9]+", bn))
  if (length(m) == 1L) m else NA_character_
}

fmrs <- lapply(inputs, function(path) {
  fmr <- readRDS(path)
  if (!methods::is(fmr, "FineMappingResultBase")) {
    warning("Skipping non-FineMappingResult input: ", path); return(NULL)
  }
  fmr
})
keep <- !vapply(fmrs, is.null, logical(1))
fmrs <- fmrs[keep]
if (length(fmrs) == 0L) stop("No FineMappingResult inputs could be read.")

# `block_id` is per INPUT, and writeBed() folds the inputs into one table in
# the order given, attaching extraColumns BEFORE the final sort -- so a vector
# built in that same order stays aligned with its rows.
recorded <- vapply(fmrs, function(fmr) any(!is.na(blockId(fmr))), logical(1))

extraColumns <- if (any(recorded)) {
  NULL
} else {
  rowsPerInput <- vapply(fmrs, function(fmr) {
    nrow(as.data.frame(topLoci(fmr, signalCutoff = argv$signal_cutoff,
                               minPurity = minPurity)))
  }, integer(1))
  blockIds <- vapply(inputs[keep],
                     function(path) .blockIdFromName(basename(path)),
                     character(1))
  list(block_id = rep(unname(blockIds), times = rowsPerInput))
}

writeBed(fmrs, argv$output, schema = "topLoci",
            signalCutoff = argv$signal_cutoff, minPurity = minPurity,
            extraColumns = extraColumns)
cat(sprintf("Wrote the top-loci BED from %d input(s) to %s\n",
            length(fmrs), argv$output))
