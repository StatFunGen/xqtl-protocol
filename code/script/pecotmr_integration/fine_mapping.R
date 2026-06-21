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
  library(optparse)
  library(pecotmr)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--qtl-dataset", type = "character"),
  make_option("--gene-id",     type = "character"),
  make_option("--cis-window",  type = "integer", default = 1000000L),
  make_option("--coverage",    type = "double",  default = 0.95),
  make_option("--output",      type = "character")
)))

qd <- readRDS(opt[["qtl-dataset"]])

res <- fineMappingPipeline(
  qd,
  methods    = "susie",
  traitId    = opt[["gene-id"]],
  cisWindow  = opt[["cis-window"]],
  coverage   = opt$coverage)

dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, opt$output)
cat(sprintf("Wrote fineMapping result for gene '%s' (%d row(s)) to %s\n",
            opt[["gene-id"]], nrow(res), opt$output))
