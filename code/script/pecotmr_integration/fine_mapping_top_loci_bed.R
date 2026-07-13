#!/usr/bin/env Rscript
# fine_mapping_top_loci_bed.R
#
# Export the top-loci BED for one or more QtlFineMappingResult / GwasFineMapping
# RDSes: the posterior view (getTopLoci, PIP/CS-filtered, with the independent
# purity filter) joined to the marginal effects (getMarginalEffects) and
# relabelled to the published BED schema. lfsr / conditional_effect are already
# per-(variant, context) numeric columns (no semicolon packing / re-splitting).
#
# Inputs:
#   --input <RDS> [<RDS> ...]  FineMappingResult RDS paths.
#   --signal-cutoff  PIP cutoff (getTopLoci). Default 0.025.
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
idCols <- c("study", "context", "trait", "region_id", "method", "variant_id")

# The published BED columns, in order; only those available are emitted.
bedFrom <- function(tl, me) {
  # marginal effect columns (betahat/sebetahat/z), matched to the posterior rows
  mm <- me[match(
    do.call(paste, c(tl[intersect(idCols, names(tl))], sep = "\r")),
    do.call(paste, c(me[intersect(idCols, names(me))], sep = "\r"))), , drop = FALSE]
  pick <- function(df, col) if (col %in% names(df)) df[[col]] else NA
  out <- data.frame(
    chr        = pick(tl, "chrom"),
    pos        = pick(tl, "pos"),
    a1         = pick(tl, "A1"),
    a2         = pick(tl, "A2"),
    variant_ID = pick(tl, "variant_id"),
    MAF        = pick(tl, "af"),
    betahat    = pick(mm, "beta"),
    sebetahat  = pick(mm, "se"),
    z          = pick(mm, "z"),
    gene_ID    = pick(tl, "trait"),
    event_ID   = pick(tl, "context"),
    cs_coverage_0.95 = pick(tl, "cs_95"),
    cs_coverage_0.7  = pick(tl, "cs_70"),
    cs_coverage_0.5  = pick(tl, "cs_50"),
    cs_95_purity     = pick(tl, "cs_95_purity"),
    PIP        = pick(tl, "pip"),
    logBF      = pick(tl, "logBF"),
    conditional_effect = pick(tl, "conditional_effect"),
    lfsr       = pick(tl, "lfsr"),
    region_id  = pick(tl, "region_id"),
    context    = pick(tl, "context"),
    stringsAsFactors = FALSE)
  # Keep the full, stable BED schema (NA where a column does not apply, e.g.
  # lfsr / conditional_effect for univariate methods) rather than dropping
  # all-NA columns, so the published schema does not vary by input. Per-CS
  # variant-level fullFit columns (within_cs_pip default +, when built with
  # --full-fit, the wide within_cs_pip_cs<k> / cs_logbf_ / cs_effect_ sets) are
  # appended dynamically.
  for (cc in grep("^(within_cs_pip|cs_logbf_|cs_effect_)", names(tl), value = TRUE))
    out[[cc]] <- tl[[cc]]
  out
}

pieces <- list()
for (path in inputs) {
  fmr <- readRDS(path)
  if (!methods::is(fmr, "FineMappingResultBase")) {
    warning("Skipping non-FineMappingResult input: ", path); next
  }
  tl <- tryCatch(as.data.frame(getTopLoci(fmr, signalCutoff = argv$signal_cutoff,
                                          minPurity = minPurity)),
                 error = function(e) { message(basename(path), ": ", conditionMessage(e)); NULL })
  if (is.null(tl) || nrow(tl) == 0L) next
  me <- as.data.frame(getMarginalEffects(fmr))
  pieces[[length(pieces) + 1L]] <- bedFrom(tl, me)
}

out <- if (length(pieces) == 0L) {
  data.frame(chr = character(0), pos = integer(0), stringsAsFactors = FALSE)
} else {
  cols <- unique(unlist(lapply(pieces, names)))
  for (k in seq_along(pieces)) {
    for (m in setdiff(cols, names(pieces[[k]]))) pieces[[k]][[m]] <- NA
    pieces[[k]] <- pieces[[k]][, cols, drop = FALSE]
  }
  o <- do.call(rbind, pieces)
  o[order(o$chr, suppressWarnings(as.numeric(o$pos))), , drop = FALSE]
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
conn <- if (grepl("\\.gz$", argv$output)) gzfile(argv$output, "w") else file(argv$output, "w")
write.table(out, file = conn, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
close(conn)
cat(sprintf("Wrote %d top-loci BED rows from %d input(s) to %s\n",
            nrow(out), length(inputs), argv$output))
