#!/usr/bin/env Rscript
# colocboost.R
#
# Per-gene colocboost worker. Loads a pre-built pecotmr::QtlDataset RDS
# and calls colocboostPipeline() for a single focal trait, colocalizing
# its signal across the QtlDataset's contexts. Designed to be invoked
# once per gene by the SoS fan-out in colocboost.ipynb.
#
# Inputs:
#   --qtl-dataset    Path to a QtlDataset RDS
#   --gene-id        Focal trait identifier (and the only traitId passed
#                    to the pipeline; colocboost colocalizes the focal
#                    trait's signal across the QtlDataset's contexts)
#   --cis-window     cis-window in bp around the trait's TSS
#   --output         Output RDS path (one colocboost-pipeline list)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Per-gene cross-context colocboost over a pre-built QtlDataset")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS",
                       type = "character")
parser <- add_argument(parser, "--gene-id",
                       help = "Focal trait identifier",
                       type = "character")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the trait's TSS",
                       type = "integer", default = 1000000L)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

qd <- readRDS(argv$qtl_dataset)

res <- colocboostPipeline(
  qd,
  traitId    = argv$gene_id,
  cisWindow  = argv$cis_window,
  focalTrait = argv$gene_id,
  xqtlColoc  = TRUE)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote colocboost result for gene '%s' to %s\n",
            argv$gene_id, argv$output))
