#!/usr/bin/env Rscript
# build_qtl_fine_mapping_result.R
#
# Regenerate protocol_example.QtlFineMappingResult.rds from the committed
# per-condition TSVs in this directory. Run once (checked-in output); rerun if
# the pecotmr FineMapping S4 layout changes.
#
#   pixi run Rscript tests/fixtures/mash/build_qtl_fine_mapping_result.R
#
# Provenance: the TSVs (variant_id, z, pip, cs_95) were derived once from the
# toy MWE fine-mapping fixtures (protocol_example.susie_fit.rds credible-set
# table + protocol_example.sumstats_db.rds z-scores) for region
# chr22:15528191-15529138, contexts Mic_De_Jager_eQTL / Ast_De_Jager_eQTL.
# Effect-size scale is z (marginal_beta = z, marginal_se = 1) since the toy
# source carries only z-scores.

suppressPackageStartupMessages(library(pecotmr))

here  <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
if (is.na(here) || !nzchar(here)) here <- "tests/fixtures/mash"
conds <- c("Mic_De_Jager_eQTL", "Ast_De_Jager_eQTL")

mkEntry <- function(cond) {
  d <- read.table(file.path(here, paste0(cond, ".tsv")), header = TRUE,
                  sep = "\t", stringsAsFactors = FALSE)
  parts <- do.call(rbind, strsplit(d$variant_id, ":", fixed = TRUE))
  cs95  <- ifelse(d$cs_95 == 0L, "susie_0", paste0("susie_", d$cs_95))
  tl <- data.frame(
    variant_id = d$variant_id, chrom = parts[, 1], pos = as.integer(parts[, 2]),
    A1 = parts[, 3], A2 = parts[, 4], N = 1000L, MAF = 0.2,
    marginal_beta = d$z, marginal_se = 1, marginal_z = d$z,
    marginal_p = 2 * pnorm(-abs(d$z)),
    pip = d$pip, posterior_mean = d$z, posterior_sd = 1, cs_95 = cs95,
    stringsAsFactors = FALSE)
  FineMappingEntry(variantIds = d$variant_id, susieFit = list(payload = cond),
                   topLoci = tl)
}

fmr <- QtlFineMappingResult(
  study   = rep("protocol_example", length(conds)),
  context = conds,
  trait   = rep("mash", length(conds)),
  method  = rep("susie", length(conds)),
  entry   = lapply(conds, mkEntry))

out <- file.path(here, "protocol_example.QtlFineMappingResult.rds")
saveRDS(fmr, out, compress = "xz")
cat(sprintf("Wrote %s (%d contexts, %d CS members)\n",
            out, nrow(fmr), nrow(getCs(fmr))))
