#!/usr/bin/env Rscript
# enloc_manifest.R
#
# Resolve SuSiE_enloc.ipynb's per-analysis-unit QTL x GWAS pairings into a
# manifest TSV, replacing the in-notebook pandas machinery
# (get_analysis_regions / get_overlapped_analysis_regions and their 11 helper
# defs). Downstream [xqtl_gwas_enrichment] / [susie_coloc] fan out over the
# manifest rows with inline csv.DictReader (no notebook-local Python).
#
# Two modes (one fan-out unit per output row):
#   --mode enrichment  (drives xqtl_gwas_enrichment): an xQTL region is paired
#                      with every GWAS block it overlaps by COORDINATE
#                      ([start,end] intersect); units are grouped by CONDITION
#                      (unit_id = condition).
#   --mode coloc       (drives susie_coloc): an xQTL region is paired with the
#                      GWAS blocks that share top-loci variants (its
#                      `block_top_loci` column); one unit per (condition,region)
#                      (unit_id = "condition@region_id").
#
# In both modes each condition (from conditions_top_loci[_minp]) is routed to the
# QTL file of ITS study via --context-meta: a context that is a substring of the
# condition maps to an analysis_name, and only QTL files whose name contains that
# analysis_name (case-insensitive) are kept. With no --context-meta and a single
# QTL file per region, the context step is skipped.
#
# Output columns (one row per unit): unit_id, qtl_files, gwas_files
# (qtl_files / gwas_files are comma-separated, path-prefixed by --qtl-path /
# --gwas-path). No data-layout path is hardcoded.

suppressPackageStartupMessages(library(argparser))

p <- arg_parser("Resolve enloc QTL x GWAS analysis units into a manifest TSV")
p <- add_argument(p, "--mode", type = "character",
                  help = "enrichment | coloc")
p <- add_argument(p, "--xqtl-meta", type = "character", help = "xQTL meta TSV")
p <- add_argument(p, "--gwas-meta", type = "character", help = "GWAS meta TSV")
p <- add_argument(p, "--context-meta", type = "character", default = "",
                  help = "context->analysis_name meta TSV (analysis_name, context)")
p <- add_argument(p, "--qtl-path", type = "character", default = "",
                  help = "prefix dir for QTL files")
p <- add_argument(p, "--gwas-path", type = "character", default = "",
                  help = "prefix dir for GWAS files")
p <- add_argument(p, "--region-list", type = "character", default = "",
                  help = "optional file; LAST column lists region_ids to keep")
p <- add_argument(p, "--region-name", type = "character", default = "",
                  help = "optional comma-separated region_ids to keep")
p <- add_argument(p, "--gwas-finemapping-obj", type = "character", nargs = Inf,
                  default = NA_character_,
                  help = "optional cohort/method tokens to filter GWAS conditions_top_loci")
p <- add_argument(p, "--minp", flag = TRUE,
                  help = "use conditions_top_loci_minp (top isoform only)")
p <- add_argument(p, "--output", type = "character", help = "output manifest TSV")
argv <- parse_args(p)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "manifest_common.R"))

mode <- match.arg(argv$mode, c("enrichment", "coloc"))

notNa      <- function(v) !is.na(v) & nzchar(trimws(as.character(v)))

xqtl <- readMeta(argv$xqtl_meta)
gwas <- readMeta(argv$gwas_meta)
gfo  <- { v <- argv$gwas_finemapping_obj; v <- v[!is.na(v)]; trimws(v)[nzchar(trimws(v))] }
tlcol <- if (argv$minp) "conditions_top_loci_minp" else "conditions_top_loci"

# ---- mode-specific: filter + per-region block_data (the overlapping GWAS) ----
if (mode == "enrichment") {
  xqtl <- xqtl[notNa(xqtl[[tlcol]]), , drop = FALSE]
  gwas <- gwas[notNa(gwas$conditions_top_loci), , drop = FALSE]
  gfo2 <- setdiff(gfo, "susie_result_trimmed")
  if (length(gfo2) > 0L)
    gwas <- gwas[grepl(paste(gfo2, collapse = "|"), gwas$conditions_top_loci), , drop = FALSE]
  # xQTL region [start,end] intersects GWAS block [start,end] on the same chrom
  blk <- vector("list", nrow(xqtl))
  for (j in seq_len(nrow(gwas))) {
    g <- gwas[j, ]
    hit <- which(xqtl[["#chr"]] == g[["#chr"]] &
                 suppressWarnings(as.integer(xqtl$start)) <= as.integer(g$end) &
                 suppressWarnings(as.integer(xqtl$end))   >= as.integer(g$start))
    for (k in hit) blk[[k]] <- c(blk[[k]], g$original_data)
  }
  xqtl$block_data <- vapply(blk, function(b) if (is.null(b)) NA_character_ else makeUnique(joinC(b)),
                            character(1))
  xqtl <- xqtl[notNa(xqtl$block_data), , drop = FALSE]
} else {
  xqtl <- xqtl[notNa(xqtl$block_top_loci), , drop = FALSE]
  gwas <- gwas[notNa(gwas$conditions_top_loci), , drop = FALSE]
  gfo2 <- setdiff(gfo, c("susie_result_trimmed", "RSS_QC_RAISS_imputed"))
  if (length(gfo2) > 0L)
    gwas <- gwas[grepl(paste(gfo2, collapse = "|"), gwas$conditions_top_loci), , drop = FALSE]
  btlIds <- unique(unlist(lapply(xqtl$block_top_loci, splitC)))
  gwas <- gwas[gwas$region_id %in% btlIds, , drop = FALSE]
  r2d <- setNames(gwas$original_data, gwas$region_id)
  xqtl$block_data <- vapply(xqtl$block_top_loci, function(bt) {
    ids <- splitC(bt); makeUnique(joinC(unname(r2d[ids[ids %in% names(r2d)]])))
  }, character(1))
}
if (nrow(xqtl) == 0L) stop("enloc_manifest: no xQTL regions after ", mode, " filtering.")

# ---- optional region restriction --------------------------------------------
regionIds <- character(0)
if (nzchar(argv$region_list) && argv$region_list != "." && file.exists(argv$region_list)) {
  rl <- readTableNoHeader(argv$region_list)
  regionIds <- unique(as.character(rl[[ncol(rl)]]))
}
rn <- splitC(argv$region_name)
if (length(rn) > 0L) regionIds <- union(regionIds, rn)
if (length(regionIds) > 0L)
  xqtl <- xqtl[xqtl$region_id %in% regionIds, , drop = FALSE]

# ---- condition-based expansion: one row per (condition, region) --------------
rows <- list()
for (i in seq_len(nrow(xqtl))) {
  r <- xqtl[i, ]
  for (cond in splitC(r[[tlcol]]))
    rows[[length(rows) + 1L]] <- data.frame(
      condition = cond, region_id = r$region_id,
      QTL = r$original_data, GWAS = r$block_data, stringsAsFactors = FALSE)
}
if (length(rows) == 0L) stop("enloc_manifest: no (condition, region) pairs produced.")
cb <- do.call(rbind, rows)

# ---- context routing: keep each condition's own study's QTL file -------------
useCtx <- nzchar(argv$context_meta) && argv$context_meta != "." && file.exists(argv$context_meta)
singleFile <- all(vapply(cb$QTL, function(q) length(splitC(q)) <= 1L, logical(1)))
if (useCtx || !singleFile) {
  if (!useCtx)
    stop("enloc_manifest: multiple QTL files per region require --context-meta to route conditions.")
  ctx <- readMeta(argv$context_meta)
  ctxLong <- do.call(rbind, lapply(seq_len(nrow(ctx)), function(k)
    data.frame(context = splitC(ctx$context[k]), analysis_name = ctx$analysis_name[k],
               stringsAsFactors = FALSE)))
  kept <- list()
  for (i in seq_len(nrow(cb))) {
    row <- cb[i, ]
    ans <- unique(ctxLong$analysis_name[vapply(ctxLong$context,
             function(c) grepl(c, row$condition, fixed = TRUE), logical(1))])
    files <- splitC(row$QTL)
    keep <- files[vapply(files, function(f)
      any(vapply(ans, function(a) grepl(tolower(a), tolower(f), fixed = TRUE), logical(1))),
      logical(1))]
    if (length(keep) > 0L)
      kept[[length(kept) + 1L]] <- data.frame(condition = row$condition, region_id = row$region_id,
        QTL = joinC(keep), GWAS = makeUnique(row$GWAS), stringsAsFactors = FALSE)
  }
  if (length(kept) == 0L) stop("enloc_manifest: context routing dropped every unit.")
  cb <- do.call(rbind, kept)
} else {
  cb$GWAS <- vapply(cb$GWAS, makeUnique, character(1))
}

# ---- path prefixing ---------------------------------------------------------
cb$QTL  <- vapply(cb$QTL,  prefixPaths, character(1), pre = argv$qtl_path)
cb$GWAS <- vapply(cb$GWAS, prefixPaths, character(1), pre = argv$gwas_path)

# ---- emit units -------------------------------------------------------------
if (mode == "enrichment") {
  out <- do.call(rbind, lapply(unique(cb$condition), function(cond) {
    s <- cb[cb$condition == cond, , drop = FALSE]
    data.frame(unit_id = cond,
               qtl_files  = joinC(splitC(joinC(s$QTL))),
               gwas_files = makeUnique(joinC(s$GWAS)), stringsAsFactors = FALSE)
  }))
} else {
  out <- data.frame(unit_id = paste0(cb$condition, "@", cb$region_id),
                    qtl_files = cb$QTL, gwas_files = cb$GWAS, stringsAsFactors = FALSE)
}

writeManifest(out, argv$output)
cat(sprintf("Wrote enloc %s manifest with %d unit(s) to %s\n", mode, nrow(out), argv$output))
