#!/usr/bin/env Rscript
# ============================================================
# snRNAseq_preprocessing.R
# Extracts the two inline-R cells of snRNAseq_preprocessing.ipynb into a single
# argparser worker. This is a faithful lift of the notebook: same CLI params,
# same singleCellTK + Seurat stack, same thresholds, output layout and file
# formats. The inline cell helpers become internal functions; all I/O comes
# from CLI params (no hardcoded paths).
#
# Steps (--step):
#   sctk_qc         — CellRanger counts -> SCTK/Seurat QC -> clustered
#                     filtered_seuratobj.rds + SCTK_QC_table.csv + QC_summary.csv
#   cell_annotation — QC'd object -> SingleR cluster-majority labels ->
#                     annotated_seuratobj.rds + celltyped_seuratobj.rds
#
# NOTE: the deeper singleCellTK->light-stack / Seurat->SCE swap is deferred
# pending collaborator sign-off (it is a method change, not a byte port) —
# see dev/snrnaseq_preprocessing_migration.md.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("snRNAseq preprocessing worker (see --step)")
parser <- add_argument(parser, "--step", type = "character",
                       help = "sctk_qc | cell_annotation")
parser <- add_argument(parser, "--output-dir", type = "character", default = "output",
                       help = "output directory")
parser <- add_argument(parser, "--num-threads", type = "numeric", default = 8,
                       help = "future multicore workers")
# sctk_qc
parser <- add_argument(parser, "--input-dir", type = "character", default = "",
                       help = "[sctk_qc] CellRanger output directory")
parser <- add_argument(parser, "--sample-meta", type = "character", default = "",
                       help = "[sctk_qc] barcode->individualID mapping CSV")
parser <- add_argument(parser, "--max-mito-percent", type = "numeric", default = 5,
                       help = "[sctk_qc] max mitochondrial percent")
parser <- add_argument(parser, "--min-ncounts", type = "numeric", default = 250,
                       help = "[sctk_qc] min total counts (nUMI)")
parser <- add_argument(parser, "--min-ngenes", type = "numeric", default = 500,
                       help = "[sctk_qc] min detected genes (nFeature)")
parser <- add_argument(parser, "--doublet-method", type = "character", default = "scds",
                       help = "[sctk_qc] scds | scrublet | doubletFinder | doubletCells | none")
parser <- add_argument(parser, "--scds-call", type = "character", default = "Singlet",
                       help = "[sctk_qc] scds_hybrid_call value to keep")
parser <- add_argument(parser, "--scrublet-threshold", type = "numeric", default = 0.25,
                       help = "[sctk_qc] max scrublet_score")
parser <- add_argument(parser, "--ambient-method", type = "character", default = "decontX",
                       help = "[sctk_qc] decontX | none")
parser <- add_argument(parser, "--max-decontxsc", type = "numeric", default = 0.5,
                       help = "[sctk_qc] max decontX_contamination")
# cell_annotation
parser <- add_argument(parser, "--sctk-rds", type = "character", default = "",
                       help = "[cell_annotation] filtered_seuratobj.rds from sctk_qc")
parser <- add_argument(parser, "--seurat-ref", type = "character", default = "",
                       help = "[cell_annotation] SingleR reference SummarizedExperiment RDS")
parser <- add_argument(parser, "--downsample-n", type = "numeric", default = 1000,
                       help = "[cell_annotation] per-cluster downsample for SingleR")
parser <- add_argument(parser, "--cluster-threshold", type = "numeric", default = 0.90,
                       help = "[cell_annotation] cluster-majority agreement to assign a label")

argv <- parse_args(parser)


# ============================================================
# Step: sctk_qc
# ============================================================
run_sctk_qc <- function(argv) {
  options(future.globals.maxSize = 8 * 1024^3)
  suppressPackageStartupMessages({
    library(future)
    library(Seurat)
    library(singleCellTK)
    library(SummarizedExperiment)
    library(scuttle)
    library(dplyr)
  })
  plan(multicore, workers = argv$num_threads)

  input_dir      <- argv$input_dir
  output_dir     <- argv$output_dir
  rds_out        <- file.path(output_dir, "SCTK_results", "filtered_seuratobj.rds")
  qc_table_out   <- file.path(output_dir, "QC_table", "SCTK_QC_table.csv")
  doublet_method <- argv$doublet_method
  ambient_method <- argv$ambient_method

  # Validate method choices
  valid_doublet <- c("scds", "scrublet", "doubletFinder", "doubletCells", "none")
  valid_ambient <- c("decontX", "none")
  if (!doublet_method %in% valid_doublet)
    stop("Invalid doublet_method '", doublet_method, "'. Choose from: ", paste(valid_doublet, collapse = ", "))
  if (!ambient_method %in% valid_ambient)
    stop("Invalid ambient_method '", ambient_method, "'. Choose from: ", paste(valid_ambient, collapse = ", "))

  message("Doublet detection method : ", doublet_method)
  message("Ambient RNA removal method: ", ambient_method)

  # ── Create output directories ─────────────────────────────────
  dir.create(dirname(rds_out),      showWarnings = FALSE, recursive = TRUE)
  dir.create(dirname(qc_table_out), showWarnings = FALSE, recursive = TRUE)

  # ── QC summary table ──────────────────────────────────────────
  qc_summary <- data.frame(Step = character(),
                            Cells = numeric(), Genes = numeric(),
                            stringsAsFactors = FALSE)
  update_summary <- function(step, obj) {
    qc_summary <<- rbind(qc_summary, data.frame(
      Step  = step,
      Cells = ncol(obj), Genes = nrow(obj)))
  }

  # ── Sample metadata ───────────────────────────────────────────
  samplelist <- read.csv(argv$sample_meta)
  samplelist$column_name <- paste0(samplelist$libraryBatch, "-counts_", samplelist$cellBarcode)
  samplelist <- samplelist[, c("individualID", "column_name")]
  colnames(samplelist) <- c("sample", "column_name")

  # ── Helper: convert SCE to Seurat ─────────────────────────────
  convert_to_seurat <- function(sce) {
    obj <- CreateSeuratObject(counts = counts(sce),
                              meta.data = as.data.frame(colData(sce)),
                              min.cells = ncol(sce) * 0.005)
    update_summary("Removing Genes expressed in < 0.5% of cells", obj)
    return(obj)
  }

  # ── Helper: assign sample IDs, convert back to SCE ────────────
  process_sce <- function(seurat_obj, samplelist) {
    seurat_obj <- seurat_obj[, colnames(seurat_obj) %in% samplelist$column_name]
    update_summary("Removing Cells with no associated Patient ID", seurat_obj)
    seurat_obj@meta.data$sample <- NULL
    seurat_obj@meta.data <- seurat_obj@meta.data %>%
      inner_join(samplelist, by = "column_name")
    seurat_obj@meta.data <- seurat_obj@meta.data[
      !duplicated(seurat_obj@meta.data$column_name), ]
    rownames(seurat_obj@meta.data) <- seurat_obj@meta.data$column_name
    return(as.SingleCellExperiment(seurat_obj))
  }

  # ── Helper: build QC algorithms list from chosen methods ──────
  # Doublet detection methods (runCellQC YAML names):
  #   "scds"          → cxds_bcds_hybrid  (cxds + bcds hybrid score)
  #   "scrublet"      → scrublet
  #   "doubletFinder" → doubletFinder
  #   "doubletCells"  → doubletCells      (scran-based)
  #   "none"          → skip doublet detection
  # Ambient RNA methods:
  #   "decontX"       → decontX
  #   "none"          → skip ambient RNA removal
  build_algorithms <- function(doublet_method, ambient_method) {
    algos <- "QCMetrics"
    if (doublet_method == "scds")          algos <- c(algos, "cxds_bcds_hybrid")
    if (doublet_method == "scrublet")      algos <- c(algos, "scrublet")
    if (doublet_method == "doubletFinder") algos <- c(algos, "doubletFinder")
    if (doublet_method == "doubletCells")  algos <- c(algos, "doubletCells")
    if (ambient_method == "decontX")       algos <- c(algos, "decontX")
    return(algos)
  }

  # ── Helper: build filter strings from chosen methods ──────────
  build_filters <- function(doublet_method, ambient_method) {
    filters <- c(
      sprintf("mito_percent < %s", argv$max_mito_percent),
      sprintf("total > %s", argv$min_ncounts),
      sprintf("detected > %s", argv$min_ngenes)
    )
    if (doublet_method == "scds") {
      filters <- c(filters, sprintf("scds_hybrid_call=='%s'", argv$scds_call))
    } else if (doublet_method == "scrublet") {
      filters <- c(filters, sprintf("scrublet_score < %s", argv$scrublet_threshold))
    } else if (doublet_method == "doubletFinder") {
      filters <- c(filters, "doubletFinder_doublet_label == 'Singlet'")
    } else if (doublet_method == "doubletCells") {
      filters <- c(filters, "scDblFinder_class == 'singlet'")
    }
    if (ambient_method == "decontX") {
      filters <- c(filters, sprintf("decontX_contamination < %s", argv$max_decontxsc))
    }
    return(filters)
  }

  # ── Helper: run SCTK-QC, filter, cluster ─────────────────────
  process_sce_to_seurat <- function(sce) {
    algos <- build_algorithms(doublet_method, ambient_method)
    message("Running QC algorithms: ", paste(algos, collapse = ", "))

    sce <- runCellQC(sce, sample = NULL,
                     algorithms       = algos,
                     mitoRef          = "human",
                     mitoIDType       = "ensembl",
                     mitoGeneLocation = "rownames",
                     seed             = 12345)

    # ── Write per-sample QC table ─────────────────────────────
    sce_temp <- sampleSummaryStats(sce, sample = colData(sce)$sample, simple = FALSE)
    sst      <- getSampleSummaryStatsTable(sce_temp, statsName = "qc_table")
    write.csv(sst, qc_table_out)
    rm(sce_temp)

    # ── Apply filters ─────────────────────────────────────────
    qc_filters <- build_filters(doublet_method, ambient_method)
    message("Applying filters:")
    for (f in qc_filters) message("  ", f)
    sce <- subsetSCECols(sce, colData = qc_filters)
    update_summary("SCTK-QC Results", sce)

    # ── Convert to Seurat and cluster ─────────────────────────
    seurat_obj <- CreateSeuratObject(counts = counts(sce),
                                     meta.data = as.data.frame(colData(sce)),
                                     min.cells = ncol(sce) * 0.005)
    update_summary("Final Seurat Object", seurat_obj)

    seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
    seurat_obj <- FindVariableFeatures(seurat_obj)
    seurat_obj <- ScaleData(seurat_obj)
    seurat_obj <- RunPCA(seurat_obj)
    seurat_obj <- RunUMAP(seurat_obj, dims = 1:10, seed.use = 1)
    seurat_obj <- FindNeighbors(seurat_obj)
    seurat_obj <- FindClusters(seurat_obj)
    return(seurat_obj)
  }

  # ── 1. Import ─────────────────────────────────────────────────
  sce <- importCellRanger(cellRangerDirs = input_dir)
  update_summary("Raw Data", sce)

  # ── 2. Remove fake mapping genes ──────────────────────────────
  fake_genes <- grep("^(1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|X|Y)_",
                     rownames(sce), value = TRUE)
  sce <- sce[setdiff(rownames(sce), fake_genes), ]
  update_summary("Removing FMGs", sce)

  # ── 3. Convert → filter → QC → cluster ───────────────────────
  seurat_obj       <- convert_to_seurat(sce);              rm(sce);        gc()
  processed_sce    <- process_sce(seurat_obj, samplelist); rm(seurat_obj); gc()
  processed_seurat <- process_sce_to_seurat(processed_sce)

  # ── 4. Save RDS ───────────────────────────────────────────────
  saveRDS(processed_seurat, rds_out)
  message("Saved: ", rds_out)
  rm(processed_seurat); gc()

  qc_csv <- file.path(dirname(rds_out), "QC_summary.csv")
  write.csv(qc_summary, qc_csv, row.names = FALSE)
  message("QC summary saved: ", qc_csv)
}


# ============================================================
# Step: cell_annotation
# ============================================================
run_cell_annotation <- function(argv) {
  options(future.globals.maxSize = 16 * 1024^3)
  suppressPackageStartupMessages({
    library(future)
    library(Seurat)
    library(singleCellTK)
    library(SummarizedExperiment)
    library(scuttle)
    library(SingleR)
    library(org.Hs.eg.db)
    library(dplyr)
  })
  plan(multicore, workers = argv$num_threads)

  sctk_file         <- argv$sctk_rds
  output_dir        <- argv$output_dir
  downsample_n      <- argv$downsample_n
  cluster_threshold <- argv$cluster_threshold

  anno_file <- file.path(output_dir, "annotated_seuratobj.rds")
  out_file  <- file.path(output_dir, "celltyped_seuratobj.rds")

  # ── Create output directory ───────────────────────────────────
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  get_most_common_label <- function(cell_types, threshold = cluster_threshold) {
    label_table <- table(cell_types)
    most_common <- max(label_table)
    if ((most_common / sum(label_table)) >= threshold) {
      return(names(which.max(label_table)))
    } else {
      return(NA)
    }
  }

  # 1. Load and downsample
  seuratObj <- readRDS(sctk_file)
  seuratObj <- subset(x = seuratObj, downsample = downsample_n)

  # 2. Extract counts, strip Ensembl version suffixes
  sce_count <- GetAssayData(seuratObj[["RNA"]], layer = "counts")
  rownames(sce_count) <- gsub("\\..*$", "", rownames(sce_count))
  rm(seuratObj); gc()

  # 3. Ensembl to HGNC symbol conversion (qualify select: dplyr masks AnnotationDbi::select)
  IDs <- AnnotationDbi::select(org.Hs.eg.db,
                keys    = rownames(sce_count),
                columns = c("ENSEMBL", "SYMBOL"),
                keytype = "ENSEMBL")
  colnames(IDs) <- c("ensembl_gene_id", "hgnc_symbol")
  IDs <- IDs[!is.na(IDs$hgnc_symbol) & IDs$hgnc_symbol != "", ]
  IDs <- IDs[!duplicated(IDs$hgnc_symbol), ]
  IDs <- IDs[!duplicated(IDs$ensembl_gene_id), ]
  sce_count <- sce_count[rownames(sce_count) %in% IDs$ensembl_gene_id, ]
  rownames(IDs) <- IDs$ensembl_gene_id
  rownames(sce_count) <- IDs[rownames(sce_count), "hgnc_symbol"]
  rm(IDs); gc()

  # 4. Load reference, intersect, free reference immediately after
  seurat_ref_SE <- readRDS(argv$seurat_ref)
  common_genes  <- intersect(rownames(sce_count), rownames(seurat_ref_SE))
  ref_subset    <- seurat_ref_SE[common_genes, ]
  rm(seurat_ref_SE); gc()

  sce_count <- sce_count[common_genes, ]
  sce_SE    <- SummarizedExperiment(assays = list(counts = sce_count))
  sce_SE    <- logNormCounts(sce_SE)
  rm(sce_count); gc()

  # 5. SingleR annotation
  singleR_res <- SingleR(test   = sce_SE,
                         ref    = ref_subset,
                         labels = ref_subset$ref_label)
  rm(sce_SE, ref_subset); gc()

  # 6. Add labels to downsampled object
  seuratObj <- readRDS(sctk_file)
  seuratObj <- subset(x = seuratObj, downsample = downsample_n)
  anno_df   <- data.frame(celltype    = singleR_res$labels,
                          column_name = rownames(singleR_res),
                          stringsAsFactors = FALSE)
  seuratObj@meta.data <- seuratObj@meta.data %>%
    inner_join(anno_df, by = "column_name")
  rownames(seuratObj@meta.data) <- seuratObj@meta.data$column_name
  saveRDS(seuratObj, anno_file)
  message("Saved annotated (downsampled): ", anno_file)

  # 7. Transfer labels to full object
  full_obj <- readRDS(sctk_file)
  full_obj@meta.data$cell <- rownames(full_obj@meta.data)

  singleR_data <- data.frame(
    celltype        = seuratObj@meta.data$celltype,
    seurat_clusters = seuratObj@meta.data$seurat_clusters
  )
  rm(seuratObj); gc()

  most_common_labels <- singleR_data %>%
    group_by(seurat_clusters) %>%
    summarise(most_common_label = get_most_common_label(celltype), .groups = "drop")

  merged_meta  <- merge(full_obj@meta.data, most_common_labels,
                        by = "seurat_clusters", all.x = TRUE)
  ordered_meta <- merged_meta[match(colnames(full_obj@assays$RNA), merged_meta$cell), ]

  full_obj@meta.data           <- ordered_meta
  rownames(full_obj@meta.data) <- full_obj@meta.data$cell
  full_obj@meta.data$celltype  <- full_obj@meta.data$most_common_label
  Idents(full_obj)             <- full_obj@meta.data$celltype

  saveRDS(full_obj, out_file)
  message("Saved celltyped: ", out_file)
  rm(full_obj, singleR_res, anno_df, singleR_data,
     most_common_labels, merged_meta, ordered_meta); gc()
}


switch(argv$step,
       sctk_qc         = run_sctk_qc(argv),
       cell_annotation = run_cell_annotation(argv),
       stop(sprintf("Unknown step: %s", argv$step)))
