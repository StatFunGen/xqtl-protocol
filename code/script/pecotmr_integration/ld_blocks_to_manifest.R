#!/usr/bin/env Rscript
# ld_blocks_to_manifest.R
#
# Parse a BED-like LD-block file (`#chr`/`chrom` + `start` + `stop`/`end`
# columns; arbitrary leading-`#` header line) into a 2-column manifest
# TSV (`region`, `region_id`). The `region` is the canonical
# `chr:start-end` string; `region_id` is the SoS-safe sanitised form
# (`:` and `-` replaced with `_`) used as the per-task ID downstream.
#
# Inputs:
#   --ld-blocks <BED>   Path to the LD-blocks BED.
#   --output <TSV>      Output manifest TSV path.

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Convert an LD-blocks BED into a region/region_id manifest")
parser <- add_argument(parser, "--ld-blocks",
                       help = "BED-like LD-block file",
                       type = "character")
parser <- add_argument(parser, "--output",
                       help = "Output manifest TSV path",
                       type = "character")
argv <- parse_args(parser)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "manifest_common.R"))

if (!file.exists(argv$ld_blocks))
  stop("--ld-blocks file not found: ", argv$ld_blocks)

lines <- readLines(argv$ld_blocks)
if (length(lines) == 0L)
  stop("Empty --ld-blocks file: ", argv$ld_blocks)

# Header: strip a leading '#' if present, split on whitespace.
hdr <- strsplit(sub("^#", "", lines[[1L]]), "\\s+", perl = TRUE)[[1L]]
chrCol  <- intersect(c("chr", "chrom"), hdr)[1L]
startCol <- intersect(c("start", "Start"), hdr)[1L]
endCol  <- intersect(c("stop", "end", "End"), hdr)[1L]
if (is.na(chrCol) || is.na(startCol) || is.na(endCol))
  stop("--ld-blocks header missing chr / start / stop|end columns; got: ",
       paste(hdr, collapse = ", "))
chrIx   <- match(chrCol,  hdr)
startIx <- match(startCol, hdr)
endIx   <- match(endCol,   hdr)

rows <- list()
for (l in lines[-1L]) {
  l <- trimws(l)
  if (!nzchar(l) || startsWith(l, "#")) next
  parts <- strsplit(l, "\\s+", perl = TRUE)[[1L]]
  if (length(parts) < max(chrIx, startIx, endIx)) next
  chrom <- parts[[chrIx]]
  start <- suppressWarnings(as.integer(parts[[startIx]]))
  end   <- suppressWarnings(as.integer(parts[[endIx]]))
  if (is.na(start) || is.na(end)) next
  region <- sprintf("%s:%d-%d", chrom, start, end)
  region_id <- gsub("[:\\-]", "_", region)
  rows[[length(rows) + 1L]] <- data.frame(
    region = region, region_id = region_id,
    stringsAsFactors = FALSE)
}
if (length(rows) == 0L)
  stop("No data rows parsed from --ld-blocks ", argv$ld_blocks)

out <- do.call(rbind, rows)
writeManifest(out, argv$output)
cat(sprintf("Wrote LD-block manifest with %d row(s) to %s\n",
            nrow(out), argv$output))
