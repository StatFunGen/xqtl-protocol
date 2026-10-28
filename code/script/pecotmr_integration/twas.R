#!/usr/bin/env Rscript
# twas.R
#
# Per-gene TWAS Z-score + Mendelian-Randomization worker. Loads a
# pre-computed per-gene TwasWeights RDS and a per-LD-block GwasSumStats
# RDS, optionally a matching per-gene FineMappingResult, and calls
# pecotmr::causalInferencePipeline().
#
# Inputs:
#   --twas-weights         Per-gene TwasWeights RDS; comma-separated for several genes
#                          of one LD block (the block LD is then computed once)
#   --gwas-sumstats        Per-region GwasSumStats RDS (one or more studies)
#   --fine-mapping-result  Optional per-gene FineMappingResult RDS
#   --rsq-cutoff           CV-R^2 weight-selection cutoff (legacy rsq_cutoff)
#   --rsq-pval-cutoff      CV-p-value gate for selection (legacy rsq_pval_cutoff)
#   --rsq-option           CV r-squared metric: rsq / adj_rsq (legacy rsq_option)
#   --rsq-pval-option      CV p-value metric candidates (legacy rsq_pval_option)
#   --mr-pip-cutoff        Pass-through (default 0.5)
#   --mr-method            Pass-through; "ivwPerVariant" or "csAware".
#                          Default "ivwPerVariant".
#   --mr-pval-cutoff       Run MR only where TWAS p < this (legacy mr_pval_cutoff)
#   --output               Output RDS path

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Per-gene TWAS Z + MR via causalInferencePipeline()")
parser <- add_argument(parser, "--twas-weights",
                       help = "Per-gene TwasWeights RDS; comma-separated for several genes of the same LD block",
                       type = "character")
parser <- add_argument(parser, "--gwas-sumstats",
                       help = "Per-LD-block GwasSumStats RDS",
                       type = "character")
parser <- add_argument(parser, "--fine-mapping-result",
                       help = "Optional per-gene FineMappingResult RDS; comma-separated, one per --twas-weights entry",
                       type = "character", default = "")
parser <- add_argument(parser, "--mr-pip-cutoff",
                       help = "Pass-through PIP cutoff for MR",
                       type = "numeric", default = 0.5)
parser <- add_argument(parser, "--mr-method",
                       help = "MR method: ivwPerVariant or csAware",
                       type = "character", default = "ivwPerVariant")
parser <- add_argument(parser, "--rsq-cutoff",
                       help = "CV weight selection: per (study,context,trait,gwasStudy) keep only the best method whose cvPerformance rsq >= this; 0 disables (legacy twas_pipeline rsq_cutoff)",
                       type = "numeric", default = 0.01)
parser <- add_argument(parser, "--rsq-pval-cutoff",
                       help = "CV-p-value gate for weight selection: a method is eligible only when its cvPerformance p-value < this; Inf disables the gate (legacy rsq_pval_cutoff)",
                       type = "numeric", default = Inf)
parser <- add_argument(parser, "--rsq-option",
                       help = "cvPerformance metric used as 'r-squared' for the cutoff/ranking: 'rsq' or 'adj_rsq' (legacy rsq_option)",
                       type = "character", default = "rsq")
parser <- add_argument(parser, "--rsq-pval-option",
                       help = "Comma-separated candidate cvPerformance p-value metric names; the first present is used (legacy rsq_pval_option)",
                       type = "character", default = "adj_rsq_pval,pval")
parser <- add_argument(parser, "--mr-pval-cutoff",
                       help = "Run MR only where TWAS p-value < this; 1 disables the gate (legacy twas_pipeline mr_pval_cutoff)",
                       type = "numeric", default = 0.05)
parser <- add_argument(parser, "--mr-cpip-cutoff",
                       help = "Cumulative-PIP cutoff for csAware MR (causalInferencePipeline mrCpipCutoff)",
                       type = "numeric", default = 0.5)
parser <- add_argument(parser, "--combine-methods",
                       help = "Comma-separated method tokens for cross-method p-value combination (combinePValues); empty = none",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path; comma-separated, one per --twas-weights entry", type = "character")
argv <- parse_args(parser)

# Cross-method combination: empty / "." -> NULL (skip), else split on comma.
combine_methods <- if (nzchar(argv$combine_methods) && argv$combine_methods != ".")
  trimws(strsplit(argv$combine_methods, ",", fixed = TRUE)[[1L]]) else NULL

# CV-p-value gate metric candidates (first present in cvPerformance is used).
rsq_pval_option <- trimws(strsplit(argv$rsq_pval_option, ",", fixed = TRUE)[[1L]])

# One call may score several genes of the same LD block: --twas-weights,
# --fine-mapping-result and --output are comma-separated lists, one entry per gene.
splitList <- function(x) {
  x <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  x[nzchar(x)]
}
twPaths <- splitList(argv$twas_weights)
outPaths <- splitList(argv$output)
fmPaths <- if (nzchar(argv$fine_mapping_result)) trimws(strsplit(sub("^,", "", argv$fine_mapping_result), ",", fixed = TRUE)[[1L]]) else character(0)
if (length(outPaths) != length(twPaths)) stop("--output needs one path per --twas-weights entry.")
if (length(fmPaths) == 0L) fmPaths <- rep("", length(twPaths))
if (length(fmPaths) != length(twPaths)) stop("--fine-mapping-result needs one path per --twas-weights entry.")

gss <- readRDS(argv$gwas_sumstats)

readFmr <- function(p) if (nzchar(p) && p != "." && file.exists(p)) readRDS(p) else NULL

# The eight rsq*/mr* arguments travel as two bundles: --rsq-option feeds
# `metric`, --rsq-pval-option feeds `pvalMetric`, whose default
# c("adj_rsq_pval", "pval") is this flag's default too.
# A gene with no scorable (QTL, GWAS) pair gets an empty list with a `noResult`
# attribute, so the other genes of the block are still scored.
for (i in seq_along(twPaths)) {
  res <- tryCatch(causalInferencePipeline(
    gwasSumStats = gss,
    twasWeights = readRDS(twPaths[i]),
    fineMappingResult = readFmr(fmPaths[i]),
    weightSelectionParam = WeightSelectionParam(
      cutoff = argv$rsq_cutoff,
      pvalCutoff = argv$rsq_pval_cutoff,
      metric = argv$rsq_option,
      pvalMetric = rsq_pval_option),
    mrParam = MrParam(
      method = argv$mr_method,
      pipCutoff = argv$mr_pip_cutoff,
      cpipCutoff = argv$mr_cpip_cutoff,
      pvalCutoff = argv$mr_pval_cutoff),
    combineMethods = combine_methods),
    error = function(e) {
      message(sprintf("No TWAS result for %s: %s", twPaths[i], conditionMessage(e)))
      structure(list(), noResult = conditionMessage(e))
    })
  dir.create(dirname(outPaths[i]), showWarnings = FALSE, recursive = TRUE)
  saveRDS(res, outPaths[i])
  cat(sprintf("Wrote causalInferencePipeline result to %s\n", outPaths[i]))
}
