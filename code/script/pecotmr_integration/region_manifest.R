#!/usr/bin/env Rscript
# region_manifest.R
#
# Resolve mnm_regression's analysis units into a manifest TSV, so the downstream
# step can fan out over its rows with inline csv.DictReader (no notebook-local
# Python parsing). One row per unit: a locus for [mnm_genes] / [fsusie], which
# run region-mode fineMappingPipeline over its ld_block, or a gene for
# [susie_twas] / [mnm], which run gene mode on its region_name.
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
#   --pheno-manifest   QtlDataset phenotype manifest TSV (ID, #chr, start, end;
#                      only ID with --gene-mode); supplies gene coordinates for
#                      the single-gene fallback and the default region set when
#                      no windows/list are given.
#   --gene-mode        List genes for a gene-mode fit ([susie_twas] / [mnm]):
#                      each selected ID must be a --pheno-manifest gene, whose
#                      coordinates are left empty when the manifest has none;
#                      --customized-association-windows does not apply.
#   --customized-association-windows  Optional BED (#chr start end region_id).
#   --region-name      Optional comma-separated region/gene IDs to restrict to;
#                      --region-list is then ignored.
#   --region-name-list Optional file of those IDs instead, read like --region-list
#                      (a genome-wide set is too long for the command line).
#   --region-list      Optional file whose LAST column lists region IDs; lines
#                      starting with "#" are comments.
#   --cis-window       bp added on each side of a gene for the single-gene
#                      fallback window. Default 1000000.
#   --output           Output manifest TSV path.
# A windows or list file that is given must exist.
#
# Output TSV columns (one row per region): region_id, chr, start, end, ld_block,
# region_name. region_name is the ID as selected, which gene mode needs verbatim
# (e.g. a versioned ENSG ID); region_id is that ID made safe for file names.

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Resolve mnm_regression's loci or genes into a manifest TSV")
parser <- add_argument(parser, "--pheno-manifest",
                       help = "QtlDataset phenotype manifest TSV (ID, #chr, start, end)",
                       type = "character")
parser <- add_argument(parser, "--gene-mode", flag = TRUE,
                       help = "List genes for a gene-mode fit: IDs must be --pheno-manifest genes, coordinates optional, no windows")
parser <- add_argument(parser, "--customized-association-windows",
                       help = "Optional BED (#chr start end region_id) of loci",
                       type = "character", default = "")
parser <- add_argument(parser, "--region-name",
                       help = "Optional comma-separated region/gene IDs to restrict to",
                       type = "character", default = "")
parser <- add_argument(parser, "--region-name-list",
                       help = "Optional file of --region-name IDs, read like --region-list (instead of --region-name)",
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
# Gene mode fits on the gene ID alone, so it needs no coordinates.
hasCoords <- !anyNA(c(chr_col, start_col, end_col))
if (is.na(id_col) || !(hasCoords || argv$gene_mode))
  stop("--pheno-manifest needs ",
       if (argv$gene_mode) "an ID column" else "ID, #chr/chrom, start, end columns",
       " (got: ", paste(names(pm), collapse = ", "), ").")
genes  <- as.character(pm[[id_col]])
uniqGenes <- unique(genes)
geneCoord <- list()
if (hasCoords) {
  gchr   <- chromAdd(pm[[chr_col]])
  gstart <- suppressWarnings(as.integer(pm[[start_col]]))
  gend   <- suppressWarnings(as.integer(pm[[end_col]]))
  geneCoord <- lapply(uniqGenes, function(g) {
    idx <- which(genes == g)
    list(chr = gchr[idx][[1L]], start = min(gstart[idx], na.rm = TRUE),
         end = max(gend[idx], na.rm = TRUE))
  })
  names(geneCoord) <- uniqGenes
}

# ----- Customized association windows (loci) --------------------------------
windows <- list()
windowOrder <- character(0)
if (nzchar(argv$customized_association_windows) &&
    argv$customized_association_windows != ".") {
  if (argv$gene_mode)
    stop("--customized-association-windows does not apply with --gene-mode.")
  if (!file.exists(argv$customized_association_windows))
    stop("--customized-association-windows not found: ", argv$customized_association_windows)
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
# A list file holds one ID per line, in its last whitespace-separated column.
readRegionIds <- function(path) {
  if (!file.exists(path)) stop("Region list not found: ", path)
  lines <- trimws(vroom::vroom_lines(path))
  lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
  unique(vapply(strsplit(lines, "\\s+"), function(parts) parts[[length(parts)]], ""))
}
rn <- if (nzchar(argv$region_name_list) && argv$region_name_list != ".")
  readRegionIds(argv$region_name_list) else splitC(argv$region_name)
if (length(rn) > 0L) {
  selected <- rn
} else if (nzchar(argv$region_list) && argv$region_list != ".") {
  selected <- readRegionIds(argv$region_list)
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
  if (argv$gene_mode && id %in% uniqGenes)
    return(list(chr = NA_character_, start = NA_integer_, end = NA_integer_))
  if (argv$gene_mode)
    stop("Gene '", id, "' not found in --pheno-manifest.")
  stop("Region/gene '", id, "' not found in --customized-association-windows ",
       "or --pheno-manifest; cannot determine its window.")
}

rows <- list()
for (id in selected) {
  r <- resolve(id)
  rows[[length(rows) + 1L]] <- data.frame(
    region_id = gsub("[^A-Za-z0-9_]", "_", id),
    chr       = r$chr, start = r$start, end = r$end,
    ld_block  = if (is.na(r$chr)) NA_character_ else sprintf("%s:%d-%d", r$chr, r$start, r$end),
    region_name = id,
    stringsAsFactors = FALSE)
}
if (length(rows) == 0L)
  stop("No regions selected; check --region-name / --region-list / ",
       "--customized-association-windows against the phenotype manifest.")
out <- do.call(rbind, rows)

writeManifest(out, argv$output)
cat(sprintf("Wrote region manifest with %d region(s) to %s\n",
            nrow(out), argv$output))
