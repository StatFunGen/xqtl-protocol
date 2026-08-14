#!/usr/bin/env Rscript
# seurat_project.R <in.rds> <out_dir>
#
# Project a (sctk_qc) Seurat object onto its reproducible, value-comparable pieces so
# a regression test can compare THOSE instead of the 15 MB serialized object — which
# carries non-deterministic timestamps / environment state and is too deeply nested to
# compare whole (recursion overflows). Under the pipeline's seed the pieces below are
# reproducible run-to-run (verified: counts + every meta.data column + PCA/UMAP embeddings).
#
# Writes into <out_dir>:
#   meta_data.tsv        per-cell metadata (QC metrics + cluster labels), cell barcode first col
#   <reduction>_embeddings.tsv   PCA / UMAP cell embeddings, cell barcode first col
#   counts_summary.tsv   a cheap counts invariant (dims + total + nnz) instead of the full matrix
suppressWarnings(suppressMessages(library(SeuratObject)))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("usage: seurat_project.R <in.rds> <out_dir>")
obj <- readRDS(args[[1L]])
dir <- args[[2L]]
dir.create(dir, showWarnings = FALSE, recursive = TRUE)

wr <- function(df, name) write.table(df, file.path(dir, name), sep = "\t",
                                     quote = FALSE, row.names = FALSE)

# per-cell metadata (barcode as an explicit first column so ordering is comparable)
md <- obj@meta.data
wr(cbind(cell = rownames(md), md), "meta_data.tsv")

# reduction embeddings (pca, umap, ...)
for (rn in names(obj@reductions)) {
  emb <- SeuratObject::Embeddings(obj, rn)
  wr(cbind(cell = rownames(emb), as.data.frame(emb)), paste0(rn, "_embeddings.tsv"))
}

# counts invariant: dims + total + non-zero count (full matrix would be a heavy fixture)
cts <- tryCatch(SeuratObject::GetAssayData(obj, slot = "counts"),
                error = function(e) SeuratObject::LayerData(obj, layer = "counts"))
nnz <- tryCatch(Matrix::nnzero(cts), error = function(e) sum(cts != 0))
writeLines(c("metric\tvalue",
             paste0("nrow\t", nrow(cts)),
             paste0("ncol\t", ncol(cts)),
             paste0("sum\t",  format(sum(cts), scientific = FALSE, trim = TRUE)),
             paste0("nnz\t",  nnz)),
           file.path(dir, "counts_summary.tsv"))
