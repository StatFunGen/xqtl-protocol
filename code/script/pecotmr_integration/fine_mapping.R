#!/usr/bin/env Rscript
# fine_mapping.R
#
# Per-gene fine-mapping worker. Loads a pre-built pecotmr::QtlDataset RDS
# and calls fineMappingPipeline() for a single trait. Designed to be
# invoked once per gene by the SoS fan-out in fine_mapping.ipynb.
#
# Inputs:
#   --qtl-dataset    Path to a QtlDataset RDS (output of qtl_dataset_construct.R)
#   --gene-id        Single trait identifier to fine-map
#   --cis-window     cis-window in bp around the trait's TSS
#   --coverage       SuSiE credible-set coverage (default 0.95)
#   --output         Output RDS path (one QtlFineMappingResult)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Per-gene SuSiE fine-mapping over a pre-built QtlDataset")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS",
                       type = "character")
parser <- add_argument(parser, "--gene-id",
                       help = "Single trait identifier to fine-map",
                       type = "character")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the trait's TSS",
                       type = "integer", default = 1000000L)
parser <- add_argument(parser, "--coverage",
                       help = "SuSiE credible-set coverage",
                       type = "numeric", default = 0.95)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

qd <- readRDS(argv$qtl_dataset)

res <- fineMappingPipeline(
  qd,
  methods    = "susie",
  traitId    = argv$gene_id,
  cisWindow  = argv$cis_window,
  coverage   = argv$coverage)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote fineMapping result for gene '%s' (%d row(s)) to %s\n",
            argv$gene_id, nrow(res), argv$output))
