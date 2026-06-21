#!/usr/bin/env Rscript
# twas_weights.R
#
# Per-gene TWAS-weights worker. Loads a pre-built pecotmr::QtlDataset RDS
# and calls twasWeightsPipeline() for a single trait. Designed to be
# invoked once per gene by the SoS fan-out in twas_weights.ipynb.
#
# Inputs:
#   --qtl-dataset           Path to a QtlDataset RDS
#   --gene-id               Single trait identifier
#   --cis-window            cis-window in bp around the trait's TSS
#   --fine-mapping-result   Optional pre-fit FineMappingResult RDS; when
#                           supplied, SuSiE-family TWAS methods reuse the
#                           fits via the `fineMappingResult` cache instead
#                           of refitting. Pass an empty string ("") or
#                           "." to skip.
#   --output                Output RDS path (one TwasWeights)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Per-gene default-preset TWAS weights over a pre-built QtlDataset")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS",
                       type = "character")
parser <- add_argument(parser, "--gene-id",
                       help = "Single trait identifier",
                       type = "character")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the trait's TSS",
                       type = "integer", default = 1000000L)
parser <- add_argument(parser, "--fine-mapping-result",
                       help = "Optional pre-fit FineMappingResult RDS",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

qd <- readRDS(argv$qtl_dataset)

fmr_path <- argv$fine_mapping_result
fmr <- if (nzchar(fmr_path) && fmr_path != "." && file.exists(fmr_path)) {
  readRDS(fmr_path)
} else {
  NULL
}

res <- twasWeightsPipeline(
  qd,
  methods           = "default",
  traitId           = argv$gene_id,
  cisWindow         = argv$cis_window,
  fineMappingResult = fmr)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote TWAS weights for gene '%s' (%d row(s)) to %s\n",
            argv$gene_id, nrow(res), argv$output))
