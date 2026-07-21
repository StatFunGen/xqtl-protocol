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
# TSS = start(getTraitPosition(db)) (NA when the trait position was never
# supplied, e.g. a QtlSumStats-derived FMR); conditions = getContexts;
# conditions_top_loci = contexts with >=1 top locus; chr/start/end default to
# getRegion(db) when not supplied via --chr/--start/--end.

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
  me <- tryCatch(as.data.frame(getMarginalEffects(db)), error = function(e) NULL)
  dir.create(dirname(argv$combined_data_sumstats_output), showWarnings = FALSE, recursive = TRUE)
  saveRDS(me, argv$combined_data_sumstats_output)
}

# ---- 3. FMR-derived meta fields --------------------------------------------
tp  <- getTraitPosition(db)
TSS <- if (methods::is(tp, "GRanges") && length(tp) > 0L) GenomicRanges::start(tp)[1L] else NA

ctx <- getContexts(db)
conditions <- if (is.null(ctx)) "" else paste(unique(as.character(ctx)), collapse = ",")

tl <- tryCatch(as.data.frame(getTopLoci(db, signalCutoff = argv$signal_cutoff, minPurity = minPurity)),
               error = function(e) NULL)
conditions_top_loci <- if (!is.null(tl) && nrow(tl) > 0L && "context" %in% names(tl))
  paste(unique(as.character(tl$context)), collapse = ",") else ""

regionId <- if (!is.na(argv$region_id)) argv$region_id else {
  tr <- unique(as.character(db$trait)); if (length(tr) == 1L) tr else NA
}
rg  <- tryCatch(getRegion(db), error = function(e) NULL)
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
  rows <- lapply(seq_len(nrow(db)), function(i) {
    sel <- list(study = as.character(db$study[i]), context = as.character(db$context[i]),
                trait = as.character(db$trait[i]), method = as.character(db$method[i]))
    pip <- tryCatch(do.call(getPip, c(list(db), sel)), error = function(e) NULL)
    if (is.null(pip)) return(NULL)
    data.frame(pip_sum = sum(pip[pip > 0], na.rm = TRUE), condition = sel$context,
               stringsAsFactors = FALSE)
  })
  pipSum <- do.call(rbind, Filter(Negate(is.null), rows))
  if (is.null(pipSum)) pipSum <- data.frame(pip_sum = numeric(0), condition = character(0))
  dir.create(dirname(argv$pip_sum_output), showWarnings = FALSE, recursive = TRUE)
  write.table(pipSum, file = argv$pip_sum_output, sep = "\t", quote = FALSE, row.names = FALSE)
}
