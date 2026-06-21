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
  library(optparse)
  library(pecotmr)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--qtl-dataset",         type = "character"),
  make_option("--gene-id",             type = "character"),
  make_option("--cis-window",          type = "integer",   default = 1000000L),
  make_option("--fine-mapping-result", type = "character", default = ""),
  make_option("--output",              type = "character")
)))

qd <- readRDS(opt[["qtl-dataset"]])

fmr_path <- opt[["fine-mapping-result"]]
fmr <- if (nzchar(fmr_path) && fmr_path != "." && file.exists(fmr_path)) {
  readRDS(fmr_path)
} else {
  NULL
}

res <- twasWeightsPipeline(
  qd,
  methods           = "default",
  traitId           = opt[["gene-id"]],
  cisWindow         = opt[["cis-window"]],
  fineMappingResult = fmr)

dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, opt$output)
cat(sprintf("Wrote TWAS weights for gene '%s' (%d row(s)) to %s\n",
            opt[["gene-id"]], nrow(res), opt$output))
