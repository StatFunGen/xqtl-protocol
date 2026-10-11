#!/usr/bin/env Rscript
# fine_mapping_cis_db_export.R
#
# Modern replacement for the `cis_results_export` monster's per-region work.
# In the FMR pipeline the heavy legacy steps are already done upstream by
# fine_mapping.R (allele alignment via matchVariants, top-loci processing, CS
# purity), so per region this wrapper only:
#   1. combines the per-study/context FineMappingResult RDS(es) into the
#      cis_results_db (itself an FMR) via combineFineMappingResults, and saves it;
#   2. saves the marginal sumstats table (getMarginalEffects) as the
#      combined_data_sumstats db;
#   3. emits the enrichment meta row SuSiE_enloc reads and the pip_sum table.
#
# Meta schema (one tab-separated row), exactly as SuSiE_enloc consumes it:
#   chr start end region_id TSS original_data combined_data
#   combined_data_sumstats conditions conditions_top_loci
# TSS = start(traitPosition(db)) (NA when the trait position was never
# supplied, e.g. a QtlSumStats-derived FMR); conditions = getContexts;
# conditions_top_loci = contexts with >=1 top locus; chr/start/end default to
# genomicRegion(db) when not supplied via --chr/--start/--end.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Combine per-region FineMappingResult(s) into the cis_results_db + emit enrichment meta")
parser <- add_argument(parser, "--input", help = "Per-study/context FineMappingResult RDS path(s) to combine",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--combined-data-output", help = "Path to write the combined db (FMR) RDS",
                       type = "character")
parser <- add_argument(parser, "--combined-data-sumstats-output",
                       help = "Optional path to write the marginal sumstats db RDS",
                       type = "character", default = NA)
parser <- add_argument(parser, "--region-id", help = "Region/gene id (default: unique trait, else region string)",
                       type = "character", default = NA)
parser <- add_argument(parser, "--chr",   help = "Region chrom (default: getRegion)", type = "character", default = NA)
parser <- add_argument(parser, "--start", help = "Region start (default: getRegion)", type = "character", default = NA)
parser <- add_argument(parser, "--end",   help = "Region end (default: getRegion)",   type = "character", default = NA)
parser <- add_argument(parser, "--signal-cutoff", help = "PIP cutoff for conditions_top_loci",
                       type = "numeric", default = 0.025)
parser <- add_argument(parser, "--min-purity", help = "CS purity cutoff for conditions_top_loci; omit for none",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--meta-output", help = "Meta TSV output path", type = "character")
parser <- add_argument(parser, "--pip-sum-output", help = "Optional pip_sum TSV output path",
                       type = "character", default = NA)
argv <- parse_args(parser)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "pecotmr_aliases.R"))

inputs <- as.character(argv$input)
if (length(inputs) == 0L) stop("--input requires at least one FineMappingResult RDS path.")
minPurity <- if (is.na(argv$min_purity)) NULL else argv$min_purity

# ---- 1. Combine per-study/context FMRs into the cis_results_db --------------
fmrs <- lapply(inputs, function(p) {
  x <- tryCatch(readRDS(p), error = function(e) NULL)
  if (!methods::is(x, "FineMappingResultBase")) {
    message("Skipping non-FineMappingResult input: ", p); return(NULL)
  }
  x
})
fmrs <- Filter(Negate(is.null), fmrs)
if (length(fmrs) == 0L) stop("No FineMappingResult inputs could be read.")
db <- if (length(fmrs) == 1L) fmrs[[1L]] else Reduce(combineFineMappingResults, fmrs)

dir.create(dirname(argv$combined_data_output), showWarnings = FALSE, recursive = TRUE)
saveRDS(db, argv$combined_data_output)

# ---- 2. Marginal sumstats db (optional) ------------------------------------
if (!is.na(argv$combined_data_sumstats_output)) {
  me <- as.data.frame(marginalEffects(db))
  dir.create(dirname(argv$combined_data_sumstats_output), showWarnings = FALSE, recursive = TRUE)
  saveRDS(me, argv$combined_data_sumstats_output)
}

# ---- 3. FMR-derived meta fields --------------------------------------------
tp  <- traitPosition(db)
TSS <- if (methods::is(tp, "GRanges") && length(tp) > 0L) GenomicRanges::start(tp)[1L] else NA

ctx <- contexts(db)
conditions <- if (is.null(ctx)) "" else paste(unique(as.character(ctx)), collapse = ",")

# Unguarded: an empty top-loci set is a 0-row table, which the next line
# handles; an accessor error is a real break and must surface.
tl <- as_variant_df(topLoci(db, signalCutoff = argv$signal_cutoff, minPurity = minPurity))
conditions_top_loci <- if (nrow(tl) > 0L && "context" %in% names(tl))
  paste(unique(as.character(tl$context)), collapse = ",") else ""

regionId <- if (!is.na(argv$region_id)) argv$region_id else {
  tr <- unique(as.character(db$trait)); if (length(tr) == 1L) tr else NA
}
rg  <- genomicRegion(db)
gr1 <- function(fn, i) if (methods::is(rg, "GRanges") && length(rg) > 0L) fn(rg)[i] else NA
chr   <- if (!is.na(argv$chr))   argv$chr   else as.character(gr1(GenomicRanges::seqnames, 1L))
start <- if (!is.na(argv$start)) argv$start else gr1(GenomicRanges::start, 1L)
end   <- if (!is.na(argv$end))   argv$end   else gr1(GenomicRanges::end, 1L)

meta <- data.frame(
  chr = chr, start = start, end = end, region_id = regionId, TSS = TSS,
  original_data          = paste(basename(inputs), collapse = ","),
  combined_data          = basename(argv$combined_data_output),
  combined_data_sumstats = if (!is.na(argv$combined_data_sumstats_output))
                             basename(argv$combined_data_sumstats_output) else "",
  conditions             = conditions,
  conditions_top_loci    = conditions_top_loci,
  stringsAsFactors = FALSE)

dir.create(dirname(argv$meta_output), showWarnings = FALSE, recursive = TRUE)
write.table(meta, file = argv$meta_output, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
cat(sprintf("Wrote cis_results_db (%d FMR input(s) -> %d rows) + meta (region_id=%s, TSS=%s) to %s\n",
            length(fmrs), nrow(db), regionId, as.character(TSS), argv$combined_data_output))

# ---- 4. pip_sum (optional) -------------------------------------------------
if (!is.na(argv$pip_sum_output)) {
  # The identity columns live in mcols(): on these collection classes `[[`
  # selects an ELEMENT, not a column, so db[["study"]] is NULL and every
  # selector below used to come out NA -- which silently produced an EMPTY
  # pip_sum (the error that NA selection raises was being swallowed).
  idm <- S4Vectors::mcols(db)
  rows <- lapply(seq_len(nrow(db)), function(i) {
    # NULL-safe per-row identity read: a GwasFineMappingResult has no context /
    # trait column, so that column is absent from mcols() entirely. Substitute
    # a length-1 NA, and drop those fields from the selection -- selecting on
    # NA matches no entry. The remaining ones come straight off this row, so
    # together they always select it.
    .col1 <- function(nm) {
      v <- idm[[nm]]
      if (is.null(v)) NA_character_ else as.character(v[i])
    }
    sel <- list(study = .col1("study"), context = .col1("context"),
                trait = .col1("trait"), method = .col1("method"))
    context <- sel$context
    sel <- sel[!vapply(sel, is.na, logical(1))]
    pipVec <- do.call(pip, c(list(db), sel))
    data.frame(pip_sum = sum(pipVec[pipVec > 0], na.rm = TRUE), condition = context,
               stringsAsFactors = FALSE)
  })
  pipSum <- do.call(rbind, Filter(Negate(is.null), rows))
  if (is.null(pipSum)) pipSum <- data.frame(pip_sum = numeric(0), condition = character(0))
  dir.create(dirname(argv$pip_sum_output), showWarnings = FALSE, recursive = TRUE)
  write.table(pipSum, file = argv$pip_sum_output, sep = "\t", quote = FALSE, row.names = FALSE)
}
