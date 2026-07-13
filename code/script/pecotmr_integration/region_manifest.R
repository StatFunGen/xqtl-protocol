#!/usr/bin/env Rscript
# region_manifest.R
#
# Resolve mnm_regression's [mnm_genes] / [fsusie] per-locus analysis units into
# a manifest TSV, so the downstream step can fan out region-mode
# fineMappingPipeline over its rows with inline csv.DictReader (no notebook-local
# Python parsing). One row per region (locus).
#
# A "region" here is a genomic window that a cross-trait (multi-gene) mvSuSiE or
# a functional fSuSiE fit runs over; fineMappingPipeline auto-selects the traits
# whose coordinates overlap the window. Region source priority:
#   1. --customized-association-windows BED (#chr start end region_id): each row
#      is a locus. Filtered to --region-name when given.
#   2. --region-list: a file whose LAST column lists region IDs; resolved against
#      the customized windows (else treated as gene IDs, see 3).
#   3. A selected ID present only in --pheno-manifest (a gene): that gene's
#      coordinates +/- --cis-window (single-gene fallback).
#
# Inputs:
#   --pheno-manifest   QtlDataset phenotype manifest TSV (ID, #chr, start, end);
#                      supplies gene coordinates for the single-gene fallback and
#                      the default region set when no windows/list are given.
#   --customized-association-windows  Optional BED (#chr start end region_id).
#   --region-name      Optional comma-separated region/gene IDs to restrict to.
#   --region-list      Optional file whose LAST column lists region IDs.
#   --cis-window       bp added on each side of a gene for the single-gene
#                      fallback window. Default 1000000.
#   --output           Output manifest TSV path.
#
# Output TSV columns (one row per region): region_id, chr, start, end, ld_block.

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Resolve per-locus regions into a manifest TSV for mnm_genes / fsusie")
parser <- add_argument(parser, "--pheno-manifest",
                       help = "QtlDataset phenotype manifest TSV (ID, #chr, start, end)",
                       type = "character")
parser <- add_argument(parser, "--customized-association-windows",
                       help = "Optional BED (#chr start end region_id) of loci",
                       type = "character", default = "")
parser <- add_argument(parser, "--region-name",
                       help = "Optional comma-separated region/gene IDs to restrict to",
                       type = "character", default = "")
parser <- add_argument(parser, "--region-list",
                       help = "Optional file whose last column lists region IDs",
                       type = "character", default = "")
parser <- add_argument(parser, "--cis-window",
                       help = "bp added on each side of a gene for the single-gene fallback window",
                       type = "numeric", default = 1000000)
parser <- add_argument(parser, "--output",
                       help = "Output manifest TSV path", type = "character")
argv <- parse_args(parser)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "manifest_common.R"))


# ----- Gene coordinates from the phenotype manifest -------------------------
if (is.null(argv$pheno_manifest) || !nzchar(argv$pheno_manifest) ||
    !file.exists(argv$pheno_manifest))
  stop("--pheno-manifest is required and must exist: ", argv$pheno_manifest)
pm <- readMeta(argv$pheno_manifest)
id_col    <- intersect(c("ID", "gene_id", "phenotype_id"), names(pm))[1L]
chr_col   <- intersect(c("#chr", "chrom", "chr"),          names(pm))[1L]
start_col <- intersect(c("start", "Start"),                names(pm))[1L]
end_col   <- intersect(c("end", "End"),                    names(pm))[1L]
if (any(is.na(c(id_col, chr_col, start_col, end_col))))
  stop("--pheno-manifest needs ID, #chr/chrom, start, end columns (got: ",
       paste(names(pm), collapse = ", "), ").")
genes  <- as.character(pm[[id_col]])
gchr   <- chromAdd(pm[[chr_col]])
gstart <- suppressWarnings(as.integer(pm[[start_col]]))
gend   <- suppressWarnings(as.integer(pm[[end_col]]))
uniqGenes <- unique(genes)
geneCoord <- lapply(uniqGenes, function(g) {
  idx <- which(genes == g)
  list(chr = gchr[idx][[1L]], start = min(gstart[idx], na.rm = TRUE),
       end = max(gend[idx], na.rm = TRUE))
})
names(geneCoord) <- uniqGenes

# ----- Customized association windows (loci) --------------------------------
windows <- list()
windowOrder <- character(0)
if (nzchar(argv$customized_association_windows) &&
    argv$customized_association_windows != "." &&
    file.exists(argv$customized_association_windows)) {
  caw <- readTableNoHeader(argv$customized_association_windows)
  for (i in seq_len(nrow(caw))) {
    rid <- as.character(caw[[4L]][[i]])
    windows[[rid]] <- list(chr = chromAdd(caw[[1L]][[i]]),
                           start = as.integer(caw[[2L]][[i]]),
                           end   = as.integer(caw[[3L]][[i]]))
    windowOrder <- c(windowOrder, rid)
  }
}

# ----- Resolve the selected region IDs --------------------------------------
selected <- character(0)
rn <- trimws(strsplit(argv$region_name, ",")[[1L]])
rn <- rn[nzchar(rn)]
if (length(rn) > 0L) {
  selected <- rn
} else if (nzchar(argv$region_list) && argv$region_list != "." &&
           file.exists(argv$region_list)) {
  for (line in readLines(argv$region_list)) {
    line <- trimws(line)
    if (!nzchar(line) || startsWith(line, "#")) next
    parts <- strsplit(line, "\\s+")[[1L]]
    selected <- c(selected, parts[[length(parts)]])
  }
  selected <- unique(selected)
} else if (length(windowOrder) > 0L) {
  selected <- windowOrder            # all loci from the windows file
} else {
  selected <- uniqGenes              # degenerate: per-gene (single-trait) fallback
  message("NOTE: no --customized-association-windows / --region-list / ",
          "--region-name given; emitting one region per gene from the ",
          "phenotype manifest (single-trait regions are degenerate for ",
          "cross-trait / fSuSiE fits).")
}

cis <- as.integer(argv$cis_window)
resolve <- function(id) {
  if (!is.null(windows[[id]])) {
    w <- windows[[id]]
    return(list(chr = w$chr, start = max(w$start, 0L), end = w$end))
  }
  if (!is.null(geneCoord[[id]])) {
    c0 <- geneCoord[[id]]
    return(list(chr = c0$chr, start = max(c0$start - cis, 0L), end = c0$end + cis))
  }
  stop("Region/gene '", id, "' not found in --customized-association-windows ",
       "or --pheno-manifest; cannot determine its window.")
}

rows <- list()
for (id in selected) {
  r <- resolve(id)
  rows[[length(rows) + 1L]] <- data.frame(
    region_id = gsub("[^A-Za-z0-9_]", "_", id),
    chr       = r$chr, start = r$start, end = r$end,
    ld_block  = sprintf("%s:%d-%d", r$chr, r$start, r$end),
    stringsAsFactors = FALSE)
}
if (length(rows) == 0L)
  stop("No regions selected; check --region-name / --region-list / ",
       "--customized-association-windows against the phenotype manifest.")
out <- do.call(rbind, rows)

writeManifest(out, argv$output)
cat(sprintf("Wrote region manifest with %d region(s) to %s\n",
            nrow(out), argv$output))
