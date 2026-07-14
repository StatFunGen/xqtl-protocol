#!/usr/bin/env Rscript
# mash_manifest.R
#
# Resolve `mash_preprocessing.ipynb`'s --region-file + --sum-files + --conditions
# into a single per-region manifest TSV. Each row carries the chrom-matched
# per-condition tensorqtl paths so the downstream `[mash_inputs_from_tensorqtl]`
# step can fan out one task per region without any Python parsing.
#
# Inputs:
#   --region-file <TSV>          Region list with columns `chr start end gene_id`.
#                                Header rows (start/end not numeric) are skipped.
#   --sum-files <txt> [<txt>...] One text file per condition; each line lists
#                                one tensorqtl summary-stats path. The file
#                                names embed the chromosome (legacy `.CHR.`
#                                convention) so we can pick the right path
#                                per region.
#   --conditions c1,c2,...       Condition labels matching --sum-files order.
#   --output <TSV>               Output manifest TSV path.
#
# Output TSV columns:
#   gene_id, chr, start, end, region, tensorqtl_paths
# where `tensorqtl_paths` is a space-separated list (one per condition, same
# order as --conditions) of the per-condition tensorqtl files for this region.

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Build a per-region manifest TSV for mash_preprocessing.ipynb")
parser <- add_argument(parser, "--region-file",
                       help = "Region list TSV (chr/start/end/gene_id)",
                       type = "character")
parser <- add_argument(parser, "--sum-files",
                       help = "One text file per condition",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--conditions",
                       help = "Comma-separated condition labels (one per --sum-files)",
                       type = "character")
parser <- add_argument(parser, "--output",
                       help = "Output manifest TSV path",
                       type = "character")
argv <- parse_args(parser)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "manifest_common.R"))

if (!file.exists(argv$region_file))
  stop("--region-file not found: ", argv$region_file)
sumFiles <- as.character(argv$sum_files)
conditions <- trimws(strsplit(argv$conditions, ",", fixed = TRUE)[[1L]])
if (length(sumFiles) != length(conditions))
  stop("--sum-files length (", length(sumFiles), ") must match --conditions ",
       "length (", length(conditions), ").")
for (sf in sumFiles) if (!file.exists(sf))
  stop("--sum-files entry not found: ", sf)

# Parse the region file (skip headers + non-numeric rows).
parseRegions <- function(path) {
  lines <- readLines(path)
  out <- list()
  for (l in lines) {
    l <- trimws(l)
    if (!nzchar(l) || startsWith(l, "#")) next
    parts <- strsplit(l, "\\s+")[[1L]]
    if (length(parts) < 4L) next
    sNum <- suppressWarnings(as.integer(parts[[2L]]))
    eNum <- suppressWarnings(as.integer(parts[[3L]]))
    if (is.na(sNum) || is.na(eNum)) next
    chr <- if (startsWith(parts[[1L]], "chr")) parts[[1L]] else paste0("chr", parts[[1L]])
    out[[length(out) + 1L]] <- data.frame(
      chr = chr, start = sNum, end = eNum,
      gene_id = parts[[4L]], stringsAsFactors = FALSE)
  }
  if (length(out) == 0L)
    stop("No data rows parsed from --region-file ", path)
  do.call(rbind, out)
}
regions <- parseRegions(argv$region_file)

# For each sum_files entry, read every line once and keep a per-chrom lookup
# (legacy `.CHR.` naming convention).
matchPerChrom <- function(sumFile) {
  lines <- readLines(sumFile)
  perChrom <- list()
  for (line in lines) {
    line <- trimws(line)
    if (!nzchar(line)) next
    m <- regmatches(line, regexec("\\.([0-9XYM]+)\\.", line))[[1L]]
    if (length(m) < 2L) next
    chrNum <- m[[2L]]
    perChrom[[chrNum]] <- c(perChrom[[chrNum]], line)
  }
  perChrom
}
perCondLookup <- lapply(sumFiles, matchPerChrom)

rows <- list()
for (i in seq_len(nrow(regions))) {
  chrNum <- sub("^chr", "", regions$chr[[i]])
  paths <- character(length(conditions))
  for (j in seq_along(conditions)) {
    hits <- perCondLookup[[j]][[chrNum]]
    if (is.null(hits) || length(hits) == 0L)
      stop("No path matching .", chrNum, ". in ", sumFiles[[j]],
           " (needed for condition '", conditions[[j]],
           "', region ", regions$chr[[i]], ":", regions$start[[i]],
           "-", regions$end[[i]], ").")
    paths[[j]] <- hits[[1L]]
  }
  rows[[length(rows) + 1L]] <- data.frame(
    gene_id         = regions$gene_id[[i]],
    chr             = regions$chr[[i]],
    start           = regions$start[[i]],
    end             = regions$end[[i]],
    region          = sprintf("%s:%d-%d", regions$chr[[i]],
                               regions$start[[i]], regions$end[[i]]),
    tensorqtl_paths = paste(paths, collapse = " "),
    stringsAsFactors = FALSE)
}
out <- do.call(rbind, rows)
writeManifest(out, argv$output)
cat(sprintf("Wrote MASH manifest with %d row(s) (%d regions x %d conditions) to %s\n",
            nrow(out), nrow(regions), length(conditions), argv$output))
