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
  library(optparse)
  library(pecotmr)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--qtl-dataset", type = "character"),
  make_option("--gene-id",     type = "character"),
  make_option("--cis-window",  type = "integer", default = 1000000L),
  make_option("--output",      type = "character")
)))

qd <- readRDS(opt[["qtl-dataset"]])

res <- colocboostPipeline(
  qd,
  traitId    = opt[["gene-id"]],
  cisWindow  = opt[["cis-window"]],
  focalTrait = opt[["gene-id"]],
  xqtlColoc  = TRUE)

dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, opt$output)
cat(sprintf("Wrote colocboost result for gene '%s' to %s\n",
            opt[["gene-id"]], opt$output))
