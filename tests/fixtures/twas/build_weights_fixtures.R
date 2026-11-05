#!/usr/bin/env Rscript
# build_weights_fixtures.R
#
# Regenerate protocol_example.ctwas_weights.s4.chr22.rds from the committed legacy
# protocol_example.ctwas_weights.chr22.rds. Run once (checked-in output); rerun if
# the pecotmr TwasWeights S4 layout changes. The weights are already on the
# correlation scale, hence weightStandardized = TRUE. Only the weights inside the
# gene's home LD block (chr22:10516173-17414263, the first row of
# ld_reference/ld_meta_file.ctwas.tsv) are kept: with all 168 weights the
# single-gene toy's gene z-score reaches 6.19 and ctwas's EM breaks down on it.
#
#   Rscript tests/fixtures/twas/build_weights_fixtures.R
#
# The sibling *.twas_weights.s4.rds comes from the protocol's own converter:
#   Rscript code/script/pecotmr_integration/twas_weights_conversion.R \
#     --legacy tests/fixtures/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.univariate_twas_weights.rds \
#     --study protocol_example \
#     --output tests/fixtures/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.twas_weights.s4.rds \
#     --meta-row <scratch>/meta_row.tsv

suppressPackageStartupMessages(library(pecotmr))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
if (is.na(here) || !nzchar(here)) here <- "tests/fixtures/twas"

legacy <- readRDS(file.path(here, "protocol_example.ctwas_weights.chr22.rds"))[[1L]]
pos <- as.integer(sub("^[^:]+:([0-9]+):.*$", "\\1", rownames(legacy$wgt)))
legacy$wgt <- legacy$wgt[pos >= 10516173L & pos < 17414263L, , drop = FALSE]
tw <- TwasWeights(
  studyName = "protocol_example", context = legacy$context,
  trait = legacy$molecular_id, method = "susie",
  entry = list(TwasWeightsRow(
    variantIds = rownames(legacy$wgt), weights = as.numeric(legacy$wgt[, 1L]),
    weightStandardized = TRUE, weightsDataType = legacy$type)),
  traitPos = GenomicRanges::GRanges(paste0("chr", legacy$chrom),
                                    IRanges::IRanges(legacy$p0, legacy$p1)))
saveRDS(tw, file.path(here, "protocol_example.ctwas_weights.s4.chr22.rds"), compress = "xz")
