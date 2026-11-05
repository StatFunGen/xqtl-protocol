#!/usr/bin/env Rscript
# build_fine_mapping_fixtures.R
#
# Regenerate the protocol_example.enloc.*.fine_mapping.rds collections from the
# committed legacy fits in this directory. Run once (checked-in output); rerun if
# the pecotmr fine-mapping S4 layout changes.
#
#   Rscript tests/fixtures/susie_enloc/build_fine_mapping_fixtures.R
#
# QTL side: RDS[[gene]][[context]]$preset_variants_result (the preset-subset fit).
# GWAS side: RDS[[block]][[study]]$RSS_QC_RAISS_imputed. The GWAS fits carry no
# mu/mu2, so their posterior columns stay at the 0 / 0.05 placeholders.
# qtlEnrichmentPipeline requires every GWAS row to name an ldReference, and the
# fits' chr1 panel is not committed, so the GWAS rows carry an empty placeholder.

suppressPackageStartupMessages(library(pecotmr))

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
if (is.na(here) || !nzchar(here)) here <- "tests/fixtures/susie_enloc"
fixture <- function(...) file.path(here, paste0("protocol_example.enloc.", ...))

csLabels <- function(fit, n) {
  out <- rep("0", n)
  cs <- fit$sets$cs
  for (k in seq_along(cs)) out[cs[[k]]] <- names(cs)[[k]]
  out
}

mkEntry <- function(fit, vids) {
  names(fit$pip) <- vids
  parts <- do.call(rbind, strsplit(vids, ":", fixed = TRUE))   # chr:pos:A2:A1
  hasMoments <- !is.null(fit$mu) && !is.null(fit$mu2)
  tl <- data.frame(
    variant_id = vids, chrom = parts[, 1], pos = as.integer(parts[, 2]),
    A1 = parts[, 4], A2 = parts[, 3], pip = unname(fit$pip),
    posterior_mean = if (hasMoments) colSums(fit$alpha * fit$mu) else 0,
    posterior_sd = if (hasMoments) sqrt(pmax(colSums(fit$alpha * fit$mu2 - (fit$alpha * fit$mu)^2), 0)) else 0.05,
    cs_95 = csLabels(fit, length(vids)),
    stringsAsFactors = FALSE, row.names = NULL)
  FineMappingRow(variantIds = vids, susieFit = fit, topLoci = tl)
}

for (study in c("MiGA_eQTL", "KNIGHT_pQTL")) {
  gene <- "ENSG00000142798"
  legacy <- readRDS(fixture(study, ".", gene, ".univariate_susie_twas_weights.rds"))[[gene]]
  fmr <- QtlFineMappingResult(
    studyName = rep(study, length(legacy)),
    context = names(legacy),
    trait = rep(gene, length(legacy)),
    method = rep("susie", length(legacy)),
    entry = lapply(legacy, function(ctx) {
      mkEntry(ctx$preset_variants_result$susie_result_trimmed,
              as.character(ctx$preset_variants_result$variant_names))
    }))
  saveRDS(fmr, fixture(study, ".", gene, ".fine_mapping.rds"), compress = "xz")
}

for (block in c("chr1_20110062_22020160", "chr1_22020160_24199848")) {
  study <- "AD_Bellenguez_2022"
  node <- readRDS(fixture("RSS_QC_RAISS_imputed.", block, ".univariate_susie_rss.rds"))[[block]][[study]]$RSS_QC_RAISS_imputed
  fmr <- GwasFineMappingResult(
    studyName = study, method = "susie_rss", blockId = block,
    ldReference = SummarizedExperiment::SummarizedExperiment(rowRanges = GenomicRanges::GRanges()),
    entry = list(mkEntry(node$susie_result_trimmed, as.character(node$variant_names))))
  saveRDS(fmr, fixture("gwas.", block, ".fine_mapping.rds"), compress = "xz")
}
