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
#   --gwas-sumstats        Per-LD-block GwasSumStats RDS
#   --fine-mapping-result  Optional per-gene FineMappingResult RDS
#   --mr-pip-cutoff        Pass-through (default 0.5)
#   --mr-method            Pass-through; "ivwPerVariant" or "csAware".
#                          Default "ivwPerVariant".
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
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

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
  mrPipCutoff       = argv$mr_pip_cutoff,
  mrMethod          = argv$mr_method)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote causalInferencePipeline result to %s\n", argv$output))
