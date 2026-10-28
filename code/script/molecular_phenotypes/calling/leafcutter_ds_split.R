#!/usr/bin/env Rscript
# ============================================================
# leafcutter_ds_split.R
# Split a LeafCutter intron count table into one table per chromosome, for the
# differential_splicing_1 step of splicing_calling_leafcutter2.ipynb.
#
# leafcutter_ds.R tests every cluster on its own, so the clusters can be tested in
# separate jobs and the results joined afterwards. Only the multiple-testing
# correction needs every cluster at once; leafcutter_ds_merge.R recomputes it.
#
# Rows go to parts 1-22 and X by their chromosome (a leading "chr" is ignored) and
# everything else (Y, MT, unplaced contigs) to "other". All 24 parts are always
# written, with the header, so the set of output files is fixed in advance; a part
# with no clusters holds the header only.
#
# With --groups, the groups file is checked against the count table first, so a
# naming mismatch stops the run here rather than in every per-part job.
#
# Writes <output-prefix>.part_<name>.counts.gz for each name in PARTS.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("Split a LeafCutter intron count table by chromosome")
parser <- add_argument(parser, "--counts", type = "character", help = "<prefix>_perind_numers.counts.gz")
parser <- add_argument(parser, "--output-prefix", type = "character", help = "<dir>/<prefix> for the part files")
parser <- add_argument(parser, "--groups", type = "character", default = NA,
                       help = "optional groups file (sample, group, covariates) to check against the table")
argv <- parse_args(parser)
if (is.na(argv$counts)) stop("--counts is required")
if (is.na(argv$output_prefix)) stop("--output-prefix is required")

die <- function(...) { message("ERROR: ", ...); quit(status = 1) }

PARTS <- c(as.character(1:22), "X", "other")

part_of <- function(intron_id) {
  chrom <- sub("^chr", "", sub(":.*$", "", intron_id))
  ifelse(chrom %in% PARTS[-length(PARTS)], chrom, "other")
}

# Fail early on the mistakes leafcutter_ds.R would otherwise hit in every part.
check_groups <- function(groups, header) {
  columns <- strsplit(trimws(header), "\\s+")[[1]]
  lines <- vroom::vroom_lines(groups)
  rows <- strsplit(trimws(lines[nzchar(trimws(lines))]), "\\s+")
  if (any(lengths(rows) < 2L))
    die(groups, " needs at least two columns: sample and group")
  sample <- vapply(rows, `[[`, "", 1L)
  group <- vapply(rows, `[[`, "", 2L)
  used <- sample %in% columns
  if (!any(used))
    die("no sample in ", groups, " matches a column of the count table. ",
        "Column names look like: ", paste(head(sort(columns), 3L), collapse = " "))
  labels <- sort(unique(group[used]))
  if (length(labels) != 2L)
    die(groups, " gives ", length(labels), " groups (", paste(labels, collapse = ", "),
        ") among the matched samples; leafcutter_ds compares exactly two")
  n <- table(factor(group[used], labels))
  cat("groups: ", sum(used), " of ", length(rows), " samples matched; ",
      paste0(labels, "=", n, collapse = ", "), "\n", sep = "")
}

lines <- vroom::vroom_lines(argv$counts)
if (!length(lines) || !nzchar(trimws(lines[[1L]]))) die(argv$counts, " is empty")
header <- lines[[1L]]
body <- lines[-1L]
body <- body[nzchar(trimws(body))]
if (!is.na(argv$groups)) check_groups(argv$groups, header)

d <- dirname(argv$output_prefix)
if (nzchar(d)) dir.create(d, showWarnings = FALSE, recursive = TRUE)
part <- factor(part_of(body), PARTS)
for (p in PARTS)
  vroom::vroom_write_lines(c(header, body[part == p]),
                           paste0(argv$output_prefix, ".part_", p, ".counts.gz"))
n <- table(part)
cat("introns per part: ", paste0(PARTS, "=", n[PARTS], collapse = " "), "\n", sep = "")
cat("total ", length(body), "\n", sep = "")
