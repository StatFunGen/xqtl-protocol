#!/usr/bin/env Rscript
# ============================================================
# pseudobulk_preprocessing.R
# Extracts the four inline cells of pseudobulk_preprocessing.ipynb into a single
# argparser worker. The two python cells (sampleid_mapping, phenotype_formatting)
# are reimplemented in R (vroom/dplyr/rtracklayer + Rsamtools bgzip/tabix); the
# two R cells drop data.table for vroom/dplyr/tidyr per the house rule.
#
# Steps (--step):
#   pseudobulk_counts    — Seurat objects -> per-sample aggregated counts csv.gz
#   sampleid_mapping     — remap individualID -> sampleid in metadata + count headers
#   pseudobulk_qc        — edgeR/limma voom residuals (+ offset) per cell type
#   phenotype_formatting — residual matrix -> TSS/peak-midpoint BED + bgzip/tabix
#
# Faithful to the notebook: same CLI, same filtering/thresholds, same output
# layout and file formats (matrix outputs keep the base-R write.table col.names=NA
# corner; the BED keeps python's %.15f). Modality (snRNA gene / snATAC peak) is
# auto-detected exactly as before.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("pseudobulk preprocessing worker (see --step)")
parser <- add_argument(parser, "--step", type = "character",
                       help = "pseudobulk_counts | sampleid_mapping | pseudobulk_qc | phenotype_formatting")
parser <- add_argument(parser, "--output-dir", type = "character", default = "output",
                       help = "output directory")
# pseudobulk_counts
parser <- add_argument(parser, "--seurat-files", type = "character", nargs = Inf, default = character(0),
                       help = "[pseudobulk_counts] Seurat .rds objects")
parser <- add_argument(parser, "--celltype", type = "character", default = "MIC",
                       help = "[pseudobulk_counts] cell type to aggregate")
parser <- add_argument(parser, "--min-cells", type = "numeric", default = 10,
                       help = "[pseudobulk_counts] drop samples with fewer cells")
# sampleid_mapping
parser <- add_argument(parser, "--map-file", type = "character", default = "",
                       help = "[sampleid_mapping] individualID -> sampleid CSV")
parser <- add_argument(parser, "--meta-files", type = "character", nargs = Inf, default = character(0),
                       help = "[sampleid_mapping/pseudobulk_qc] metadata CSVs")
parser <- add_argument(parser, "--count-files", type = "character", nargs = Inf, default = character(0),
                       help = "[sampleid_mapping/pseudobulk_qc] count matrix csv.gz")
# pseudobulk_qc
parser <- add_argument(parser, "--tech-vars-file", type = "character", default = "",
                       help = "[pseudobulk_qc] technical covariates CSV (sampleid + vars)")
parser <- add_argument(parser, "--blacklist-file", type = "character", default = "",
                       help = "[pseudobulk_qc] blacklist BED (optional)")
parser <- add_argument(parser, "--batch-correction", type = "character", default = "FALSE",
                       help = "[pseudobulk_qc] TRUE/FALSE apply batch correction")
parser <- add_argument(parser, "--batch-method", type = "character", default = "limma",
                       help = "[pseudobulk_qc] limma | combat")
parser <- add_argument(parser, "--quant-norm", type = "character", default = "FALSE",
                       help = "[pseudobulk_qc] TRUE/FALSE quantile-normalize residuals")
parser <- add_argument(parser, "--min-count", type = "numeric", default = 5,
                       help = "[pseudobulk_qc] filterByExpr min.count")
parser <- add_argument(parser, "--min-total-count", type = "numeric", default = 15,
                       help = "[pseudobulk_qc] filterByExpr min.total.count")
parser <- add_argument(parser, "--min-prop", type = "numeric", default = 0.1,
                       help = "[pseudobulk_qc] filterByExpr min.prop")
parser <- add_argument(parser, "--min-nuclei", type = "numeric", default = 20,
                       help = "[pseudobulk_qc] drop samples with <= min_nuclei")
parser <- add_argument(parser, "--regions", type = "character", default = "",
                       help = "[pseudobulk_qc] comma-separated chr:start:end peak filter (ATAC)")
parser <- add_argument(parser, "--gene-list", type = "character", default = "",
                       help = "[pseudobulk_qc] comma-separated gene ids to keep (RNA)")
# phenotype_formatting
parser <- add_argument(parser, "--residual-files", type = "character", nargs = Inf, default = character(0),
                       help = "[phenotype_formatting] residual matrices (one per cell type dir)")
parser <- add_argument(parser, "--gtf-file", type = "character", default = "",
                       help = "[phenotype_formatting] gene GTF for TSS (RNA gene branch)")
argv <- parse_args(parser)

`%||%` <- function(a, b) if (length(a) == 0 || is.null(a)) b else a

# ===========================================================================
# STEP: pseudobulk_counts
# ===========================================================================
run_pseudobulk_counts <- function(argv) {
  suppressPackageStartupMessages(library(Seurat))
  seurat_files <- argv$seurat_files
  celltype   <- argv$celltype
  min_cells  <- argv$min_cells
  outdir     <- file.path(argv$output_dir, "0_pseudobulk_counts")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  out_file   <- file.path(outdir, sprintf("pseudobulk_counts_%s.csv.gz", celltype))

  message("Loading and subsetting Seurat objects for: ", celltype)
  subsets <- list()
  for (f in seurat_files) {
    message("Loading: ", basename(f))
    obj <- readRDS(f)
    if (celltype %in% obj$celltype) {
      # Faithful to the notebook: inputs are pre-subset per cell type, so this
      # keeps all non-NA-celltype cells (the celltype guard above gates it).
      subsets[[length(subsets) + 1]] <- subset(obj, subset = celltype == celltype)
    } else {
      message("  Skipping - celltype not found in: ", basename(f))
    }
    rm(obj); gc()
  }
  if (length(subsets) == 0) stop("No Seurat objects contain celltype: ", celltype)
  message("Found ", celltype, " in ", length(subsets), " objects")

  merged <- Reduce(merge, subsets); rm(subsets); gc()
  merged <- JoinLayers(merged)
  merged <- SetIdent(merged, value = "sample")

  cell_counts <- table(merged@meta.data$sample)
  expr <- AggregateExpression(merged, group.by = "sample", slot = "counts")$RNA
  rm(merged); gc()

  valid_samples <- names(cell_counts[cell_counts >= min_cells])
  # AggregateExpression turns "_" into "-" in identities; map back for subsetting.
  valid_cols <- gsub("_", "-", valid_samples)
  valid_cols <- valid_cols[valid_cols %in% colnames(expr)]
  expr <- expr[, valid_cols, drop = FALSE]
  colnames(expr) <- gsub("-", "_", colnames(expr))
  message("Samples after min_cells (>= ", min_cells, ") filter: ", ncol(expr))

  rownames(expr) <- gsub("\\..*$", "", rownames(expr))
  message("Genes: ", nrow(expr), " | Samples: ", ncol(expr))

  out <- dplyr::bind_cols(tibble::tibble(gene_id = rownames(expr)),
                          tibble::as_tibble(as.data.frame(expr)))
  vroom::vroom_write(out, out_file, delim = ",", quote = "none", escape = "none")
  message("Saved: ", out_file)
}

# ===========================================================================
# STEP: sampleid_mapping
# ===========================================================================
run_sampleid_mapping <- function(argv) {
  suppressPackageStartupMessages(library(dplyr))
  outdir <- file.path(argv$output_dir, "1_files_with_sampleid")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  map_df <- vroom::vroom(argv$map_file, delim = ",", show_col_types = FALSE, progress = FALSE)
  if ("sampleid" %in% colnames(map_df)) {
    id_map <- setNames(as.character(map_df$sampleid), as.character(map_df$individualID))
  } else {
    id_map <- setNames(as.character(map_df$individualID), as.character(map_df$individualID))
  }
  map_id <- function(x) { m <- id_map[x]; ifelse(is.na(m), x, unname(m)) }

  # metadata: read as character (preserve exact value formatting), add sampleid,
  # reorder sampleid + individualID to the front.
  for (in_path in argv$meta_files) {
    fname <- basename(in_path)
    meta <- vroom::vroom(in_path, delim = ",", col_types = vroom::cols(.default = "c"),
                         show_col_types = FALSE, progress = FALSE)
    if (!"individualID" %in% colnames(meta)) {
      message("Warning: individualID column not found in ", fname, ", skipping.")
      next
    }
    meta$sampleid <- map_id(meta$individualID)
    rest <- setdiff(colnames(meta), c("sampleid", "individualID"))
    meta <- meta[, c("sampleid", "individualID", rest)]
    vroom::vroom_write(meta, file.path(outdir, fname), delim = ",",
                       quote = "none", escape = "none")
  }

  # count files: remap sample column names in the header, keep the body verbatim.
  for (in_path in argv$count_files) {
    fname <- basename(in_path)
    con <- gzfile(in_path, "r"); header_line <- readLines(con, n = 1); close(con)
    cols <- strsplit(header_line, ",", fixed = TRUE)[[1]]
    new_header <- paste(c(cols[1], map_id(cols[-1])), collapse = ",")
    body <- readLines(gzfile(in_path, "r"))[-1]
    out_con <- gzfile(file.path(outdir, fname), "w")
    writeLines(c(new_header, body), out_con); close(out_con)
  }
}

# ===========================================================================
# STEP: pseudobulk_qc
# ===========================================================================
run_pseudobulk_qc <- function(argv) {
  suppressPackageStartupMessages({
    library(edgeR); library(limma); library(GenomicRanges); library(dplyr); library(tidyr)
  })
  batch_correction <- as.logical(argv$batch_correction)
  batch_method     <- argv$batch_method
  quant_norm       <- as.logical(argv$quant_norm)
  if (isTRUE(batch_correction) && batch_method == "combat") suppressPackageStartupMessages(library(sva))

  predictOffset <- function(fit, tech_vars) {
    D <- fit$design; Dm <- D
    for (col in colnames(D)) {
      if (col == "(Intercept)") next
      is_tech <- any(vapply(tech_vars, function(v) grepl(paste0("^", v), col), logical(1)))
      if (is_tech) {
        if (is.numeric(D[, col]) && !all(D[, col] %in% c(0, 1)))
          Dm[, col] <- median(D[, col], na.rm = TRUE)
        else Dm[, col] <- 0
      } else Dm[, col] <- 0
    }
    B <- fit$coefficients; B[is.na(B)] <- 0
    off <- B %*% t(Dm); colnames(off) <- rownames(fit$design); off
  }

  peak_coords <- function(ids) {
    m <- do.call(rbind, strsplit(gsub("_", "-", ids), "-", fixed = TRUE))
    data.frame(chr = m[, 1], start = as.numeric(m[, 2]), end = as.numeric(m[, 3]))
  }
  filter_blacklist <- function(mat, bed, feat_label) {
    pk <- peak_coords(rownames(mat))
    bl <- vroom::vroom(bed, delim = "\t", col_names = FALSE, show_col_types = FALSE, progress = FALSE)[, 1:3]
    colnames(bl) <- c("chr", "start", "end")
    gr1 <- GRanges(pk$chr, IRanges(pk$start, pk$end))
    gr2 <- GRanges(bl$chr, IRanges(as.numeric(bl$start), as.numeric(bl$end)))
    hits <- unique(queryHits(findOverlaps(gr1, gr2)))
    if (length(hits) > 0) {
      message("Blacklisted ", feat_label, " removed: ", length(hits))
      return(mat[-hits, , drop = FALSE])
    }
    mat
  }
  parse_regions <- function(region_str) {
    if (is.null(region_str) || region_str == "") return(NULL)
    lapply(strsplit(region_str, ",")[[1]], function(r) {
      parts <- strsplit(trimws(r), ":|−|-")[[1]]
      list(chr = parts[1], start = as.integer(parts[2]), end = as.integer(parts[3]))
    })
  }
  filter_regions <- function(mat, regions) {
    pk <- peak_coords(rownames(mat))
    gr_peaks <- GRanges(pk$chr, IRanges(as.integer(pk$start), as.integer(pk$end)))
    gr_regions <- GRanges(vapply(regions, `[[`, "", "chr"),
                          IRanges(vapply(regions, `[[`, 0L, "start"),
                                  vapply(regions, `[[`, 0L, "end")))
    keep <- unique(queryHits(findOverlaps(gr_peaks, gr_regions)))
    if (length(keep) == 0) stop("No peaks overlap the specified regions.")
    message("Peaks after region filter: ", length(keep))
    mat[keep, , drop = FALSE]
  }

  meta_files  <- argv$meta_files
  count_files <- argv$count_files
  if (length(meta_files) != length(count_files))
    stop("meta_files and count_files must have the same length and order.")

  tech_df   <- as.data.frame(vroom::vroom(argv$tech_vars_file, delim = ",",
                                          show_col_types = FALSE, progress = FALSE))
  tech_vars <- setdiff(colnames(tech_df), "sampleid")
  message("Tech vars: ", paste(tech_vars, collapse = ", "))

  regions   <- parse_regions(argv$regions)
  gene_list <- trimws(strsplit(argv$gene_list, ",")[[1]])
  gene_list <- gene_list[gene_list != ""]

  for (i in seq_along(meta_files)) {
    meta_file <- meta_files[i]; counts_file <- count_files[i]
    ct <- sub("\\.csv$", "", sub("^.*metadata_", "", basename(meta_file)))
    message("\nProcessing: ", ct)
    outdir <- file.path(argv$output_dir, "2_residuals", ct)
    dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

    counts_raw <- as.data.frame(vroom::vroom(counts_file, delim = ",",
                                             show_col_types = FALSE, progress = FALSE))
    counts <- as.matrix(counts_raw[, -1, drop = FALSE]); rownames(counts) <- counts_raw[[1]]

    is_atac <- grepl("^chr.*-[0-9]+-[0-9]+$", rownames(counts)[1])
    feat_label <- if (is_atac) "peaks" else "genes"
    message("Modality: ", if (is_atac) "snATAC-seq" else "snRNA-seq")
    message("Loaded: ", nrow(counts), " ", feat_label, " x ", ncol(counts), " samples")

    if (is_atac && !is.null(regions)) {
      counts <- filter_regions(counts, regions)
    } else if (!is_atac && length(gene_list) > 0) {
      genes_present <- intersect(rownames(counts), gene_list)
      if (length(genes_present) == 0) stop("No matching genes found in count matrix.")
      message("Genes after gene_list filter: ", length(genes_present))
      counts <- counts[genes_present, , drop = FALSE]
    }

    meta  <- as.data.frame(vroom::vroom(meta_file, delim = ",", show_col_types = FALSE, progress = FALSE))
    idcol <- intersect(c("sampleid", "sampleID", "individualID", "projid"), colnames(meta))[1]
    if (is.na(idcol)) stop("Cannot find sample ID column in metadata.")

    n_nuclei_col <- intersect(c("n_nuclei", "n.nuclei", "nNuclei", "nuclei_count"), colnames(meta))[1]
    if (!is.na(n_nuclei_col)) {
      meta <- meta[meta[[n_nuclei_col]] > argv$min_nuclei, , drop = FALSE]
      message("Samples after nuclei (>", argv$min_nuclei, ") filter: ", nrow(meta))
    }

    common <- intersect(meta[[idcol]], colnames(counts))
    if (length(common) == 0) stop("Zero sample overlap between metadata and count matrix.")
    counts <- counts[, common, drop = FALSE]
    message("Samples after alignment: ", length(common))

    if (nzchar(argv$blacklist_file) && file.exists(argv$blacklist_file)) {
      counts <- filter_blacklist(counts, argv$blacklist_file, feat_label)
      message(feat_label, " after blacklist filter: ", nrow(counts))
    } else message("No blacklist file - skipping.")

    tech_sub <- tech_df[tech_df$sampleid %in% common, , drop = FALSE]
    tech_sub <- tech_sub[match(common, tech_sub$sampleid), , drop = FALSE]
    keep_rows <- complete.cases(tech_sub[, tech_vars, drop = FALSE])
    tech_sub  <- tech_sub[keep_rows, , drop = FALSE]
    counts    <- counts[, tech_sub$sampleid, drop = FALSE]
    message("Valid samples for modelling: ", nrow(tech_sub))

    dge <- DGEList(counts = counts, samples = tech_sub)
    dge$samples$group <- factor(rep("all", ncol(dge)))
    message(feat_label, " before filter: ", nrow(dge))
    keep <- filterByExpr(dge, group = dge$samples$group,
                         min.count = argv$min_count, min.total.count = argv$min_total_count,
                         min.prop = argv$min_prop)
    dge <- dge[keep, , keep.lib.sizes = FALSE]
    message(feat_label, " after filter: ", nrow(dge))

    write.table(dge$counts, file.path(outdir, paste0(ct, "_filtered_raw_counts.txt")),
                sep = "\t", quote = FALSE, col.names = NA)

    dge <- calcNormFactors(dge, method = "TMM")

    if (isTRUE(batch_correction) && "sequencingBatch" %in% colnames(dge$samples)) {
      batches <- dge$samples$sequencingBatch
      valid_batches <- names(table(batches)[table(batches) > 1])
      keep_bc <- batches %in% valid_batches
      dge <- dge[, keep_bc, keep.lib.sizes = FALSE]; batches <- batches[keep_bc]
      message("Samples after singleton batch removal: ", ncol(dge))
      logCPM <- cpm(dge, log = TRUE, prior.count = 1)
      if (batch_method == "combat") {
        logCPM <- ComBat(dat = logCPM, batch = factor(batches))
        message("ComBat applied on log-CPM.")
      } else {
        logCPM <- removeBatchEffect(logCPM, batch = factor(batches))
        message("limma removeBatchEffect applied.")
      }
      dge$counts <- round(pmax(2^logCPM, 0))
    }

    batch_vars <- c()
    if ("sequencingBatch" %in% colnames(dge$samples) &&
        length(unique(dge$samples$sequencingBatch)) > 1) {
      dge$samples$sequencingBatch_factor <- factor(dge$samples$sequencingBatch)
      batch_vars <- c(batch_vars, "sequencingBatch_factor")
    }
    if ("Library" %in% colnames(dge$samples) &&
        length(unique(dge$samples$Library)) > 1) {
      dge$samples$Library_factor <- factor(dge$samples$Library)
      batch_vars <- c(batch_vars, "Library_factor")
    }

    all_model_vars <- intersect(c(tech_vars, batch_vars), colnames(dge$samples))
    form <- as.formula(paste("~", paste(all_model_vars, collapse = " + ")))
    design <- model.matrix(form, data = dge$samples)
    message("Formula: ", deparse(form))
    if (!is.fullrank(design)) {
      message("Design not full rank - trimming.")
      qr_d <- qr(design); design <- design[, qr_d$pivot[seq_len(qr_d$rank)], drop = FALSE]
    }
    message("Design matrix: ", nrow(design), " x ", ncol(design))

    v <- voom(dge, design, plot = FALSE)
    fit <- eBayes(lmFit(v, design))
    off <- predictOffset(fit, tech_vars = tech_vars)
    res <- residuals(fit, v$E)
    final <- off + res

    out_file <- file.path(outdir, paste0(ct, "_residuals.txt"))
    write.table(final, out_file, sep = "\t", quote = FALSE, col.names = NA)
    message("Saved: ", out_file)

    common_rds <- list(dge = dge, offset = off, residuals = res, final_data = final,
                       valid_samples = colnames(dge), design = design, fit = fit, model = form,
                       tech_vars = tech_vars, batch_vars = batch_vars,
                       batch_correction = batch_correction,
                       batch_method = if (batch_correction) batch_method else "none",
                       modality = if (is_atac) "snATAC-seq" else "snRNA-seq")
    if (quant_norm) {
      final_qn <- t(apply(final, 1, rank, ties.method = "average"))
      final_qn <- stats::qnorm(final_qn / (ncol(final_qn) + 1))
      write.table(final_qn, file.path(outdir, paste0(ct, "_residuals_qn.txt")),
                  sep = "\t", quote = FALSE, col.names = NA)
      saveRDS(c(common_rds, list(final_data_qn = final_qn, quant_norm = TRUE)),
              file.path(outdir, paste0(ct, "_results_qn.rds")))
    } else {
      saveRDS(c(common_rds, list(quant_norm = FALSE)),
              file.path(outdir, paste0(ct, "_results.rds")))
    }
    message("Completed: ", ct, " -> ", outdir)
  }
}

# ===========================================================================
# STEP: phenotype_formatting
# ===========================================================================
run_phenotype_formatting <- function(argv) {
  suppressPackageStartupMessages({ library(dplyr); library(rtracklayer) })
  out_dir <- file.path(argv$output_dir, "3_pheno_reformat")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  gene_positions <- NULL
  if (nzchar(argv$gtf_file)) {
    gr <- rtracklayer::import(argv$gtf_file, feature.type = "gene")
    tss <- ifelse(as.character(strand(gr)) == "+", start(gr), end(gr))
    gene_positions <- data.frame(gene_id = gr$gene_id,
                                 chr = as.character(seqnames(gr)), tss = tss,
                                 stringsAsFactors = FALSE)
    gene_positions <- gene_positions[!duplicated(gene_positions$gene_id), ]
    rownames(gene_positions) <- gene_positions$gene_id
  }

  read_residuals <- function(path) {
    samples <- strsplit(readLines(path, n = 1), "\t")[[1]][-1]
    d <- as.data.frame(vroom::vroom(path, delim = "\t", skip = 1, col_names = FALSE,
                                    show_col_types = FALSE, progress = FALSE))
    ids <- as.character(d[[1]])
    vals <- d[, -1, drop = FALSE]; colnames(vals) <- samples
    list(ids = ids, vals = vals)
  }

  fmt <- function(x) sprintf("%.15f", x)

  for (res_path in argv$residual_files) {
    ct <- basename(dirname(res_path))
    message("\nPhenotype Formatting: ", ct)
    if (!file.exists(res_path)) { message("WARNING: ", res_path, " not found, skipping."); next }
    r <- read_residuals(res_path)
    ids <- r$ids; vals <- r$vals
    message("Loaded ", length(ids), " features x ", ncol(vals), " samples")

    parts <- strsplit(ids, "-", fixed = TRUE)
    if (max(lengths(parts)) >= 3) {
      chrs   <- vapply(parts, `[`, "", 1)
      starts <- as.numeric(vapply(parts, `[`, "", 2))
      ends   <- as.numeric(vapply(parts, `[`, "", 3))
      mids   <- (starts + ends) %/% 2
      bed <- data.frame(`#chr` = chrs, start = mids, end = mids + 1, ID = ids,
                        check.names = FALSE, stringsAsFactors = FALSE)
      bed <- cbind(bed, vals)
    } else {
      if (is.null(gene_positions)) stop("gene branch requires --gtf-file")
      gene_id <- sub("\\..*$", "", ids)
      keep <- which(gene_id %in% gene_positions$gene_id)
      gp <- gene_positions[gene_id[keep], ]
      bed <- data.frame(`#chr` = gp$chr, start = gp$tss, end = gp$tss + 1, ID = ids[keep],
                        check.names = FALSE, stringsAsFactors = FALSE)
      bed <- cbind(bed, vals[keep, , drop = FALSE])
    }
    bed <- bed[order(bed$`#chr`, bed$start), , drop = FALSE]

    val_cols <- setdiff(colnames(bed), c("#chr", "start", "end", "ID"))
    bed[val_cols] <- lapply(bed[val_cols], fmt)
    out_bed <- file.path(out_dir, paste0(ct, "_phenotype.bed"))
    vroom::vroom_write(bed, out_bed, delim = "\t", quote = "none", escape = "none")
    Rsamtools::bgzip(out_bed, dest = paste0(out_bed, ".gz"), overwrite = TRUE)
    file.remove(out_bed)
    Rsamtools::indexTabix(paste0(out_bed, ".gz"), format = "bed")
    message("Completed: ", ct, " -> ", out_dir)
  }
}

switch(argv$step,
       pseudobulk_counts    = run_pseudobulk_counts(argv),
       sampleid_mapping     = run_sampleid_mapping(argv),
       pseudobulk_qc        = run_pseudobulk_qc(argv),
       phenotype_formatting = run_phenotype_formatting(argv),
       stop(sprintf("Unknown step: %s", argv$step)))
