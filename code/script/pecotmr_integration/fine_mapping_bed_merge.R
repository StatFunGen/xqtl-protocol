#!/usr/bin/env Rscript
# fine_mapping_bed_merge.R
#
# Concatenate the per-region published tables written by fine_mapping.R /
# fine_mapping_top_loci_bed.R (pecotmr::writeBed) into one sorted,
# compressed table for a study or context.
#
# Reads the per-region tables rather than the per-region FineMappingResult
# RDSes on purpose: writeBed() can fold many results into one call,
# but that holds every fit in memory at once, and a genome-wide run is
# thousands of regions. The tables are small and already carry the published
# schema, so merging them keeps the peak footprint flat.
#
# Every table pecotmr publishes is a half-open BED now -- #chr / start / end
# for the top-loci schema as much as for the fSuSiE one, which is the point of
# the two sharing one vocabulary -- so both are tabix-indexable. The header is
# still sniffed, because a hand-assembled position table (chr / pos / ...) can
# reach this step from outside the pipeline, and indexing one with the tabix
# "bed" preset would read its third column, an allele, as an end coordinate.
#
# Inputs:
#   --input <TABLE> [...]  Per-region tables (all sharing one header).
#   --input-list <PATH>    File listing those tables, one per line, instead (a
#                          genome-wide list is too long for the command line).
#   --output <PATH>        Merged output; bgzipped when it ends in .gz/.bgz.

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
  library(Rsamtools)
  library(pecotmr)
})

parser <- arg_parser("Merge per-region fine-mapping tables into one sorted table")
parser <- add_argument(parser, "--input", help = "Per-region table paths",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--input-list",
                       help = "File listing the per-region table paths, one per line (instead of --input)",
                       type = "character", default = "")
parser <- add_argument(parser, "--output", help = "Merged output path",
                       type = "character")
argv <- parse_args(parser)

readListFile <- function(f) { x <- trimws(readLines(f)); x[nzchar(x)] }
inputs <- if (nzchar(argv$input_list)) readListFile(argv$input_list) else as.character(argv$input)
if (length(inputs) == 0L) stop("--input requires at least one table path.")

# col_types = "c" throughout: the packed ";"-joined columns must survive
# untouched and the table is re-emitted verbatim. One eager (altrep = FALSE)
# read per file: a multi-file vroom() call, or lazy columns, keeps every file
# open, and a genome-wide merge of thousands of tables then hits the
# open-file limit.
tables <- lapply(inputs, vroom, delim = "\t", col_types = vroom::cols(.default = "c"),
                 progress = FALSE, altrep = FALSE)
header <- names(tables[[1L]])
differs <- !vapply(tables, function(t) identical(names(t), header), logical(1))
if (any(differs))
  stop("All tables must share the header of ", inputs[[1L]], "; these differ: ",
       paste(inputs[differs], collapse = ", "))
merged <- bind_rows(tables)

cols <- names(merged)
# A half-open BED starts #chr/start/end; anything else came from elsewhere and
# is treated as a position table.
is_bed <- length(cols) >= 3L &&
  grepl("^#?chr", cols[[1L]], ignore.case = TRUE) &&
  identical(tolower(cols[[2L]]), "start") &&
  identical(tolower(cols[[3L]]), "end")

if (nrow(merged) > 0L) {
  # genomicOrder() (pecotmr) is the single definition of genomic order, shared
  # with the writers: chromosomes naturally (2 before 10), positions
  # numerically. `end` only breaks ties, and is ignored when it is not a
  # position -- which is what makes one call serve both schemas.
  merged <- merged[genomicOrder(merged[[1L]], merged[[2L]],
                                if (is_bed) merged[[3L]] else NULL), ,
                   drop = FALSE]
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
compress <- grepl("\\.(gz|bgz)$", argv$output)
plain <- if (compress) sub("\\.(gz|bgz)$", "", argv$output) else argv$output
vroom_write(merged, plain, delim = "\t", quote = "none", na = "NA",
            progress = FALSE)
if (compress) {
  bgzip(plain, dest = argv$output, overwrite = TRUE)
  file.remove(plain)
  # Only the real BED is tabix-indexable, and an empty table has no records.
  if (is_bed && nrow(merged) > 0L) indexTabix(argv$output, format = "bed")
}
cat(sprintf("Merged %d table(s), %d rows, into %s%s\n",
            length(inputs), nrow(merged), argv$output,
            if (is_bed && compress && nrow(merged) > 0L) " (tabix-indexed)" else ""))
