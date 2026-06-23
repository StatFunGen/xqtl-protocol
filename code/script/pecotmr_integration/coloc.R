#!/usr/bin/env Rscript
# coloc.R
#
# Colocalization worker. Loads an S4 `QtlFineMappingResult` and either
# an S4 `GwasFineMappingResult` or a `GwasSumStats` (colocPipeline does
# inline fine-mapping when given sumstats), and calls
# `pecotmr::colocPipeline()`. When `--enrichment` points at the output
# of `qtl_enrichment.R`, the per-pair `p12` prior is scaled by
# `(1 + enrichment)` (capped at `--p12-max`), the "enloc" variant of
# coloc.
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
#   --p12-max              Cap on enrichment-adjusted p12 (default 1e-3)
#   --no-adjust-pips       Disable adjustPips (default: PIPs renormalized to overlap)
#   --finemapping-methods  Comma-separated list for inline GWAS FM when --gwas-input is GwasSumStats (default 'susie')
#   --output               Output RDS path (data.frame of coloc results)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(coloc)
})

parser <- arg_parser("Colocalization via colocPipeline()")
parser <- add_argument(parser, "--qtl-fine-mapping",
                       help = "Path to S4 QtlFineMappingResult RDS",
                       type = "character")
parser <- add_argument(parser, "--gwas-input",
                       help = "Path to S4 GwasFineMappingResult OR GwasSumStats RDS",
                       type = "character")
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
parser <- add_argument(parser, "--p12-max",
                       help = "Cap on enrichment-adjusted p12",
                       type = "numeric", default = 1e-3)
parser <- add_argument(parser, "--no-adjust-pips",
                       help = "Disable adjustPips (use FMRs as supplied)",
                       flag = TRUE)
parser <- add_argument(parser, "--finemapping-methods",
                       help = "Comma-separated methods for inline GWAS FM (when --gwas-input is GwasSumStats)",
                       type = "character", default = "susie")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

qtlFmr   <- readRDS(argv$qtl_fine_mapping)
gwasIn   <- readRDS(argv$gwas_input)
enrich   <- if (nzchar(argv$enrichment) && argv$enrichment != "." &&
                file.exists(argv$enrichment)) readRDS(argv$enrichment) else NULL

methods <- trimws(strsplit(argv$finemapping_methods, ",", fixed = TRUE)[[1L]])

res <- colocPipeline(
  qtlFineMappingResult     = qtlFmr,
  gwasInput                = gwasIn,
  filterLbfCs              = argv$filter_lbf_cs,
  filterLbfCsSecondary     = if (is.na(argv$filter_lbf_cs_secondary)) NULL
                              else argv$filter_lbf_cs_secondary,
  p1                       = argv$p1,
  p2                       = argv$p2,
  p12                      = argv$p12,
  finemappingMethods       = methods,
  enrichment               = enrich,
  p12Max                   = argv$p12_max,
  adjustPips               = !argv$no_adjust_pips)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote colocPipeline result (%d row(s)) to %s\n",
            nrow(res), argv$output))
