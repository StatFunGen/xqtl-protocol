#!/usr/bin/env Rscript
# fine_mapping_conversion.R
#
# Convert legacy FunGen-xQTL QTL fine-mapping RDS files (the frozen Synapse
# release, *.univariate_bvsr.rds; list layout gene -> "<context>_<gene>" ->
# susie_result_trimmed / sumstats / variant_names / region_info) into ONE
# pecotmr (>= 0.8.2) QtlFineMappingResult per gene -- the fineMappingResult
# that causalInferencePipeline (twas.R --fine-mapping-result) uses for MR.
#
# Study and context labels follow twas_weights_conversion.R (context = the
# legacy name without the trailing gene id), so MR pairs each fine-mapping
# row with the converted TWAS weights of the same (study, context, gene).
# The SuSiE fit is kept unchanged; topLoci carries the marginal QTL effects,
# PIP, posterior mean / sd and the 95% credible set of every variant. cs_95 holds
# plain integer labels (0 = none) and betahat / sebetahat duplicate the
# marginal effects, because causalInferencePipeline MR reads those names.
# No LD sketch is attached (causalInferencePipeline skips the LD check then).
#
# Usage:
#   Rscript fine_mapping_conversion.R --legacy MSBB_eQTL.ENSG....univariate_bvsr.rds \
#     --study MSBB_eQTL --output gene.fine_mapping.s4.rds

suppressPackageStartupMessages({ library(argparser); library(pecotmr) })
p <- arg_parser("Legacy QTL fine-mapping -> pecotmr QtlFineMappingResult")
p <- add_argument(p, "--legacy", help = "Legacy *.univariate_bvsr.rds for ONE gene")
p <- add_argument(p, "--study", help = "Study label, e.g. MSBB_eQTL (must match the TWAS weights)")
p <- add_argument(p, "--output", help = "Output QtlFineMappingResult RDS")
a <- parse_args(p)

lg <- readRDS(a$legacy)
nCS <- 0L; S <- C <- Tr <- character(0); E <- list(); TPc <- character(0); TPp <- integer(0)
for (gene in names(lg)) for (cn in names(lg[[gene]])) {
  co <- lg[[gene]][[cn]]
  fit <- co$susie_result_trimmed; ss <- co$sumstats; vids <- co$variant_names
  if (is.null(fit) || is.null(ss$betahat) || length(vids) == 0L) next
  stopifnot(length(ss$betahat) == length(vids), length(fit$pip) == length(vids))
  parts <- do.call(rbind, strsplit(vids, ":", fixed = TRUE))   # chr:pos:A2:A1
  pm <- colSums(fit$alpha * fit$mu)
  psd <- sqrt(pmax(colSums(fit$alpha * fit$mu2) - pm^2, 0))
  cs95 <- rep("0", length(vids))   # 0 = not in any 95% CS; k = in CS k
  cs <- fit$sets$cs
  if (length(cs)) for (k in seq_along(cs)) cs95[cs[[k]]] <- if (is.null(names(cs))) as.character(k) else sub("^L", "", names(cs)[k])
  z <- ss$betahat / ss$sebetahat
  tl <- data.frame(variant_id = vids, chrom = parts[, 1], pos = as.integer(parts[, 2]),
                   A1 = parts[, 4], A2 = parts[, 3], N = length(co$sample_names), MAF = NA_real_,
                   marginal_beta = ss$betahat, marginal_se = ss$sebetahat, marginal_z = z,
                   betahat = ss$betahat, sebetahat = ss$sebetahat,
                   marginal_p = 2 * pnorm(-abs(z)), pip = unname(fit$pip),
                   posterior_mean = unname(pm), posterior_sd = unname(psd), cs_95 = cs95,
                   stringsAsFactors = FALSE)
  nCS <- nCS + sum(cs95 != "0")
  E[[length(E) + 1L]] <- fineMappingRow(variantIds = vids, susieFit = fit, topLoci = tl)
  S <- c(S, a$study); C <- c(C, sub(paste0("[_:]", gene, "$"), "", cn)); Tr <- c(Tr, gene)
  rc <- co$region_info$region_coord
  TPc <- c(TPc, paste0("chr", sub("^chr", "", rc$chrom))); TPp <- c(TPp, as.integer(rc$start))
}
if (!length(E)) stop("No usable fine-mapping entries in: ", a$legacy)
fmr <- QtlFineMappingResult(study = S, context = C, trait = Tr, method = rep("susie", length(E)),
                            entry = E, traitPos = GenomicRanges::GRanges(TPc, IRanges::IRanges(TPp, width = 1L)))
dir.create(dirname(a$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(fmr, a$output)
cat(sprintf("Wrote %d fine-mapping rows (%d contexts, %d CS members) for %s -> %s\n",
            length(E), length(unique(C)), nCS, paste(unique(Tr), collapse = ","), a$output))
