#!/usr/bin/env Rscript
# twas.R
#
# Per-gene TWAS Z-score + Mendelian-Randomization worker. Loads a
# pre-computed per-gene TwasWeights RDS and a per-LD-block GwasSumStats
# RDS, optionally a matching per-gene FineMappingResult, and calls
# pecotmr::causalInferencePipeline().
#
# Inputs:
#   --twas-weights         Per-gene TwasWeights RDS (output of twas_weights.ipynb)
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
                       help = "Per-gene TwasWeights RDS",
                       type = "character")
parser <- add_argument(parser, "--gwas-sumstats",
                       help = "Per-LD-block GwasSumStats RDS",
                       type = "character")
parser <- add_argument(parser, "--fine-mapping-result",
                       help = "Optional per-gene FineMappingResult RDS",
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
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Cross-method combination: empty / "." -> NULL (skip), else split on comma.
combine_methods <- if (nzchar(argv$combine_methods) && argv$combine_methods != ".")
  trimws(strsplit(argv$combine_methods, ",", fixed = TRUE)[[1L]]) else NULL

# CV-p-value gate metric candidates (first present in cvPerformance is used).
rsq_pval_option <- trimws(strsplit(argv$rsq_pval_option, ",", fixed = TRUE)[[1L]])

tw  <- readRDS(argv$twas_weights)
gss <- readRDS(argv$gwas_sumstats)

fmr_path <- argv$fine_mapping_result
fmr <- if (nzchar(fmr_path) && fmr_path != "." && file.exists(fmr_path)) {
  readRDS(fmr_path)
} else {
  NULL
}

res <- causalInferencePipeline(
  gwasSumStats      = gss,
  twasWeights       = tw,
  fineMappingResult = fmr,
  rsqCutoff         = argv$rsq_cutoff,
  rsqPvalCutoff     = argv$rsq_pval_cutoff,
  rsqOption         = argv$rsq_option,
  rsqPvalOption     = rsq_pval_option,
  mrPipCutoff       = argv$mr_pip_cutoff,
  mrMethod          = argv$mr_method,
  mrCpipCutoff      = argv$mr_cpip_cutoff,
  mrPvalCutoff      = argv$mr_pval_cutoff,
  combineMethods    = combine_methods)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote causalInferencePipeline result to %s\n", argv$output))
