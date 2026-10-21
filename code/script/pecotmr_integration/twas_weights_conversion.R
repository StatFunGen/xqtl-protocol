#!/usr/bin/env Rscript
# twas_weights_conversion.R
#
# Convert legacy FunGen-xQTL TWAS weight RDS files (the frozen Synapse release;
# list layout gene -> "<context>_<gene>" -> twas_weights / twas_cv_result /
# region_info) into ONE pecotmr (>= 0.8.2) TwasWeights object per gene, and
# write the matching xqtl_meta row
# (#chr start end region_id TSS original_data contexts).
#
# Several legacy files for the same gene (e.g. univariate + multicontext) are
# merged into one TwasWeights, because twas_ctwas.ipynb uses one weight file
# per gene. No LD sketch is attached (ldSketch = NULL), as in the protocol
# fixture protocol_example.twas.reshaped_toy.*.twas_weights.s4.rds.
#
# Usage:
#   Rscript twas_weights_conversion.R \
#     --legacy a.univariate_twas_weights.rds,a.multicontext_twas_weights.rds \
#     --study MSBB_eQTL --output gene.twas_weights.s4.rds --meta-row gene.meta.tsv
#     [--data-type eQTL --type-rows gene.types.tsv]

suppressPackageStartupMessages({ library(argparser); library(pecotmr) })
p <- arg_parser("Legacy TWAS weights -> pecotmr TwasWeights")
p <- add_argument(p, "--legacy", help = "Comma-separated legacy RDS files for ONE gene")
p <- add_argument(p, "--study", help = "Study label, e.g. MSBB_eQTL")
p <- add_argument(p, "--output", help = "Output TwasWeights RDS")
p <- add_argument(p, "--meta-row", help = "Output one-row xqtl_meta TSV (no header)")
p <- add_argument(p, "--data-type", default = "", help = "Optional data type of this study (eQTL, pQTL, sQTL, ...)")
p <- add_argument(p, "--type-rows", default = "", help = "Optional output: context<TAB>data_type rows (no header); needs --data-type")
a <- parse_args(p)

files <- trimws(strsplit(a$legacy, ",")[[1]])
S <- C <- Tr <- M <- character(0); E <- list(); ctxs <- character(0); info <- NULL
# traitPos: the gene TSS for every row; cTWAS places each gene into its LD block by it
TPc <- character(0); TPp <- integer(0)

# legacy CV performance (1-row matrix: corr rsq adj_rsq pval RMSE MAE)
# -> list(metrics = named numeric), the layout used by the protocol fixture
cvMetrics <- function(perf) {
  if (is.null(perf)) return(NULL)
  v <- unlist(as.data.frame(perf)[1, , drop = TRUE])
  list(metrics = setNames(as.numeric(v), names(v)))
}

for (f in files) {
  lg <- readRDS(f)
  for (gene in names(lg)) for (cn in names(lg[[gene]])) {
    co <- lg[[gene]][[cn]]
    if (!is.list(co) || is.null(co$twas_weights)) next
    if (is.null(info)) info <- co$region_info
    ctx <- sub(paste0("[_:]", gene, "$"), "", cn)  # sQTL names end in ":<gene>"
    perf <- co$twas_cv_result$performance
    names(perf) <- sub("_performance$", "", names(perf))
    for (wn in names(co$twas_weights)) {
      w <- co$twas_weights[[wn]]
      vids <- if (is.matrix(w)) rownames(w) else names(w)
      if (is.null(vids)) vids <- co$variant_names
      wv <- if (is.matrix(w)) as.numeric(w[, 1]) else as.numeric(w)
      if (all(is.na(wv)) || all(wv == 0, na.rm = TRUE)) next
      m <- sub("_weights$", "", wn)
      fits <- if (m %in% c("susie", "susie_inf", "susie_ash")) co$susie_weights_intermediate else NULL
      E[[length(E) + 1L]] <- twasWeightsRow(variantIds = vids, weights = wv,
                                           fits = fits, cvResult = cvMetrics(perf[[m]]))
      S <- c(S, a$study); C <- c(C, ctx); Tr <- c(Tr, gene); M <- c(M, m)
      TPc <- c(TPc, paste0("chr", sub("^chr", "", co$region_info$region_coord$chrom)))
      TPp <- c(TPp, as.integer(co$region_info$region_coord$start))
    }
    ctxs <- union(ctxs, ctx)
  }
}
if (!length(E)) stop("No usable (non-zero) weights in: ", a$legacy)

tw <- TwasWeights(study = S, context = C, trait = Tr, method = M, entry = E,
                  traitPos = GenomicRanges::GRanges(TPc, IRanges::IRanges(TPp, width = 1L)))
dir.create(dirname(a$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(tw, a$output)

g <- info$grange; rc <- info$region_coord
# Released weights store the TSS as a 1-bp region_coord; warn if it is a range
if (rc$end - rc$start > 1)
  warning("region_coord spans ", rc$start, "-", rc$end, " (not a single TSS); using its start as TSS")
row <- data.frame(paste0("chr", sub("^chr", "", g$chrom)), g$start, g$end,
                  unique(Tr)[1], rc$start, normalizePath(a$output),
                  paste(ctxs, collapse = ","))
write.table(row, a$meta_row, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
if (nzchar(a$type_rows)) {
  if (!nzchar(a$data_type)) stop("--type-rows needs --data-type")
  write.table(data.frame(ctxs, a$data_type), a$type_rows, sep = "\t", quote = FALSE,
              row.names = FALSE, col.names = FALSE)
}
cat(sprintf("Wrote %d weight rows (%d contexts) for %s -> %s\n",
            length(E), length(ctxs), unique(Tr)[1], a$output))
