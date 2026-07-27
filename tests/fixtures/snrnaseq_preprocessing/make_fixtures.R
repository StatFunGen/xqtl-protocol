#!/usr/bin/env Rscript
# Build committed CI fixtures for the snRNAseq_preprocessing worker from the
# external MWE (~/Downloads/fungen_xqtl/xqtl-protocol/input/snrnaseq). Not run in
# CI — run once by hand to (re)generate the small fixtures under this directory.
#
#   pixi run --frozen Rscript tests/fixtures/snrnaseq_preprocessing/make_fixtures.R
#
# Produces (all small enough to commit):
#   cellranger/<sample>-counts/outs/filtered_feature_bc_matrix/{barcodes,features,matrix}
#       — 800-barcode CellRanger subset (FMG + mito + top-expressed genes kept)
#   id_mapping.csv          — EXACT barcode->individualID rows for the retained
#                             barcodes (the full MWE file is 967 MB but only the
#                             rows matching this one sample matter)
#   seurat_ref_SE.rds       — reference SummarizedExperiment downsampled to
#                             ~150 cells/label (the full MWE ref is 1.25 GB)
suppressPackageStartupMessages({
  library(singleCellTK)
  library(DropletUtils)
  library(SummarizedExperiment)
})
set.seed(1)

MWE     <- path.expand("~/Downloads/fungen_xqtl/xqtl-protocol/input/snrnaseq")
OUT     <- "tests/fixtures/snrnaseq_preprocessing"
SAMPLE  <- "200225-B10-A-counts"
BATCH   <- "200225-B10-A"
N_CELLS <- 800
N_TOPGENE <- 6000
N_REF_PER_LABEL <- 150

dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── CellRanger subset ─────────────────────────────────────────────────────────
sce <- importCellRanger(cellRangerDirs = file.path(MWE, "protocol_example.snrnaseq.cellranger"))
m   <- counts(sce)
# barcodes: the raw cellBarcode (strip the "<sample>_" prefix importCellRanger adds)
barcodes_full <- sub(paste0("^", SAMPLE, "_"), "", colnames(sce))

# id_mapping rows for this sample (only these barcodes carry a patient ID)
idmap <- read.csv(file.path(MWE, "protocol_example.snrnaseq.id_mapping.csv"))
idmap <- idmap[idmap$libraryBatch == BATCH, ]
with_id <- barcodes_full %in% idmap$cellBarcode

# keep a mix: mostly patient-ID cells (survive the join) + some without
keep_idx <- sort(c(
  sample(which(with_id),  min(sum(with_id),  N_CELLS - 100)),
  sample(which(!with_id), min(sum(!with_id), 100))
))
m  <- m[, keep_idx, drop = FALSE]
bc <- barcodes_full[keep_idx]

# features: FMG (chr-prefixed, to exercise FMG removal) + human mito (versioned
# Ensembl -> strip version to match singleCellTK's unversioned set, so
# mito_percent is real) + top-expressed, so the 0.5%-of-cells filter also bites
fmg    <- grep("^(1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|X|Y)_",
               rownames(m), value = TRUE)
sce_m  <- singleCellTK::importMitoGeneSet(
            SingleCellExperiment::SingleCellExperiment(
              list(counts = `rownames<-`(m, sub("\\..*$", "", rownames(m))))),
            reference = "human", id = "ensembl", by = "rownames",
            collectionName = "mito")
mito_ids <- tryCatch(
  unlist(GSEABase::geneIds(S4Vectors::metadata(sce_m)$sctk$genesets$mito)),
  error = function(e) character(0))
mito   <- rownames(m)[sub("\\..*$", "", rownames(m)) %in% mito_ids]
tot    <- Matrix::rowSums(m)
top    <- names(sort(tot[tot > 0], decreasing = TRUE))[seq_len(min(N_TOPGENE, sum(tot > 0)))]
keep_g <- union(union(top, mito), sample(fmg, min(length(fmg), 200)))
m      <- m[rownames(m) %in% keep_g, , drop = FALSE]
cat("mito genes kept:", length(mito), "\n")

cr_dir <- file.path(OUT, "cellranger", SAMPLE, "outs", "filtered_feature_bc_matrix")
unlink(file.path(OUT, "cellranger"), recursive = TRUE)
dir.create(dirname(cr_dir), recursive = TRUE, showWarnings = FALSE)
colnames(m) <- bc
write10xCounts(cr_dir, m, barcodes = bc, gene.id = rownames(m),
               gene.symbol = rowData(sce)[rownames(m), 2], version = "3", overwrite = TRUE)
cat("cellranger fixture:", nrow(m), "genes x", ncol(m), "cells\n")

# ── id_mapping fixture (exact rows for the retained, patient-ID barcodes) ──────
idmap_fx <- idmap[idmap$cellBarcode %in% bc, c("libraryBatch", "cellBarcode", "individualID")]
write.csv(idmap_fx, file.path(OUT, "id_mapping.csv"), row.names = FALSE)
cat("id_mapping fixture:", nrow(idmap_fx), "rows\n")

# ── reference SE subset (downsample cells per label, keep all genes) ──────────
ref <- readRDS(file.path(MWE, "protocol_example.snrnaseq.seurat_ref_SE.rds"))
sel <- unlist(lapply(split(seq_len(ncol(ref)), ref$ref_label), function(ix)
  sample(ix, min(length(ix), N_REF_PER_LABEL))))
ref <- ref[, sort(sel)]
saveRDS(ref, file.path(OUT, "seurat_ref_SE.rds"))
cat("ref SE fixture:", nrow(ref), "genes x", ncol(ref), "cells;",
    length(unique(ref$ref_label)), "labels\n")
