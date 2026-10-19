#!/usr/bin/env Rscript
# coloc.R
#
# Colocalization worker. Loads an S4 `QtlFineMappingResult` and either
# an S4 `GwasFineMappingResult` or a `GwasSumStats` (colocPipeline does
# inline fine-mapping when given sumstats), and calls
# `pecotmr::colocPipeline()`. When `--enrichment` points at the output
# of `qtl_enrichment.R`, the per-pair `p12` prior is scaled by
# `(1 + enrichment)`, the "enloc" variant of coloc. (pecotmr 0.9.0 removed
# the cap on that adjusted prior, so the former --p12-max flag is gone.)
#
# Inputs:
#   --qtl-fine-mapping     Path to S4 QtlFineMappingResult RDS
#   --gwas-input           Path to S4 GwasFineMappingResult RDS, or
#                          S4 GwasSumStats RDS (colocPipeline dispatches)
#   --enrichment           Optional enrichment data.frame RDS (output of
#                          qtl_enrichment.R)
#   --filter-lbf-cs        Flag: keep only effects that produced a CS
#   --filter-lbf-cs-secondary  Secondary coverage for CS-concentration filter (default NULL)
#   --p1                   Per-variant QTL prior (default 1e-4)
#   --p2                   Per-variant GWAS prior (default 1e-4)
#   --p12                  Per-variant shared prior (default 5e-6)
#   --no-adjust-pips       Disable adjustPips (default: PIPs renormalized to overlap)
#   --finemapping-methods  Comma-separated list for inline GWAS FM when --gwas-input is GwasSumStats (default 'susie')
#   --output               Output RDS path (ColocResult collection)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(coloc)
})

parser <- arg_parser("Colocalization via colocPipeline()")
parser <- add_argument(parser, "--qtl-fine-mapping",
                       help = "Path(s) to S4 QtlFineMappingResult RDS (combined if >1)",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--gwas-input",
                       help = "Path(s) to S4 GwasFineMappingResult (combined if >1) OR one GwasSumStats RDS",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--enrichment",
                       help = "Optional enrichment data.frame RDS (from qtl_enrichment.R)",
                       type = "character", default = "")
parser <- add_argument(parser, "--filter-lbf-cs",
                       help = "Keep only effects that produced a CS",
                       flag = TRUE)
parser <- add_argument(parser, "--filter-lbf-cs-secondary",
                       help = "Secondary coverage for CS-concentration filter (default NA = off)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--p1",
                       help = "Per-variant QTL prior",
                       type = "numeric", default = 1e-4)
parser <- add_argument(parser, "--p2",
                       help = "Per-variant GWAS prior",
                       type = "numeric", default = 1e-4)
parser <- add_argument(parser, "--p12",
                       help = "Per-variant shared prior",
                       type = "numeric", default = 5e-6)
parser <- add_argument(parser, "--no-adjust-pips",
                       help = "Disable adjustPips (use FMRs as supplied)",
                       flag = TRUE)
parser <- add_argument(parser, "--finemapping-methods",
                       help = "Comma-separated methods for inline GWAS FM (when --gwas-input is GwasSumStats)",
                       type = "character", default = "susie")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Load one-or-more RDS(es) and combine FineMappingResults (the modern pipeline
# supplies per-study/per-block FMRs directly, replacing the legacy converter's
# multi --rds-files combine). A single non-FMR input (e.g. GwasSumStats) is used
# as supplied.
.loadCombine <- function(paths) {
  objs <- lapply(as.character(paths), readRDS)
  if (length(objs) == 1L) return(objs[[1L]])
  if (all(vapply(objs, function(o) methods::is(o, "FineMappingResultBase"), logical(1)))) {
    # Preserve the (shared) ldSketch; combineFineMappingResults defaults it NULL.
    ld <- tryCatch(objs[[1L]]@ldSketch, error = function(e) NULL)
    return(do.call(combineFineMappingResults, c(objs, list(ldSketch = ld))))
  }
  objs[[1L]]
}
qtlFmr   <- .loadCombine(argv$qtl_fine_mapping)
gwasIn   <- .loadCombine(argv$gwas_input)
enrich   <- if (nzchar(argv$enrichment) && argv$enrichment != "." &&
                file.exists(argv$enrichment)) readRDS(argv$enrichment) else NULL

methods <- trimws(strsplit(argv$finemapping_methods, ",", fixed = TRUE)[[1L]])

# pecotmr 0.9.0 groups these into ColocPriorParam / ColocLbfFilterParam /
# GwasFineMappingParam. `priors` is passed only WITHOUT an enrichment table:
# 0.9.0 refuses both at once (with an enrichment table the priors are enloc's,
# derived from the genome-wide QTL results, so supplying them here too would
# mean scoring with one set and enriching with another), and it detects
# "supplied" as missing(priors) -- so even default values must be withheld.
# Priors the caller actually chose are not silently dropped: that combination
# is rejected here, with pecotmr's own reasoning.
prior_defaults <- c(p1 = 1e-4, p2 = 1e-4, p12 = 5e-6)
prior_values   <- c(p1 = argv$p1, p2 = argv$p2, p12 = argv$p12)
priors_set     <- !isTRUE(all.equal(prior_values, prior_defaults))
if (!is.null(enrich) && priors_set)
  stop("--p1/--p2/--p12 cannot be combined with --enrichment: with an ",
       "enrichment table the colocalization priors are enloc's, derived from ",
       "the genome-wide QTL results by qtl_enrichment.R. Drop the prior flags, ",
       "or drop --enrichment to use your own.")

coloc_args <- list(
  qtlFineMappingResult = qtlFmr,
  gwasInput            = gwasIn,
  lbfFilterArgs        = ColocLbfFilterParam(
    filterLbfCs = argv$filter_lbf_cs,
    secondary   = if (is.na(argv$filter_lbf_cs_secondary)) NULL
                  else argv$filter_lbf_cs_secondary),
  gwasFineMappingArgs  = GwasFineMappingParam(methods = methods),
  enrichment           = enrich,
  adjustPips           = !argv$no_adjust_pips)
if (is.null(enrich))
  coloc_args$priors <- ColocPriorParam(p1 = argv$p1, p2 = argv$p2, p12 = argv$p12)

res <- do.call(colocPipeline, coloc_args)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote colocPipeline result (%d row(s)) to %s\n",
            nrow(res), argv$output))
