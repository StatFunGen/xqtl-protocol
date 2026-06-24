#!/usr/bin/env Rscript
# gwas_rss_manifest.R
#
# Resolve `rss_analysis.ipynb`'s per-study GWAS sources (a
# --gwas-meta TSV and/or --gwas-tsv-list STUDY=PATH items) together with
# its region inputs (a --region-list BED-ish file and/or --regions
# chr:start-end strings) into a single per-job manifest TSV. One row per
# (study x region) cross-product. Downstream SoS steps fan out over this
# TSV's rows without needing any custom Python parsing.
#
# Inputs (zero or more of each input pair; at least one of each pair):
#   --gwas-meta <TSV>           Optional. Columns: study_id, path,
#                                column_mapping (optional). Relative
#                                paths resolve against the meta file's
#                                own directory.
#   --gwas-tsv-list STUDY=PATH  Optional. One or more positional items.
#                                (zero or more)              The notebook may pass each STUDY=PATH item
#                                                            as a separate positional arg.
#   --region-list <BED>         Optional. Whitespace-separated #chr start
#                                end ...; lines starting with '#' or with
#                                non-numeric start/end are skipped.
#   --regions chr:start-end     Optional. One or more positional items.
#                                (zero or more)
#   --output <TSV>              Output manifest path.
#
# Output TSV columns:
#   study_id, gwas_tsv, column_mapping, chr, start, end, region_id,
#   gwas_tsv_basename
#
# `region_id` is the SoS-safe sanitised region label (`:` and `-` replaced
# with `_`); the downstream notebook uses it as the per-task ID.
# `gwas_tsv_basename` is the basename of the TSV without extension — the
# notebook uses it in the per-task output filename for traceability.

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Resolve GWAS sources x regions into a per-task manifest TSV")
parser <- add_argument(parser, "--gwas-meta",
                       help = "Optional per-study meta TSV",
                       type = "character", default = "")
parser <- add_argument(parser, "--gwas-tsv-list",
                       help = "Zero or more STUDY=PATH items",
                       type = "character", nargs = Inf, default = character(0))
parser <- add_argument(parser, "--region-list",
                       help = "Optional BED-like region file",
                       type = "character", default = "")
parser <- add_argument(parser, "--regions",
                       help = "Zero or more chr:start-end items",
                       type = "character", nargs = Inf, default = character(0))
parser <- add_argument(parser, "--output",
                       help = "Output manifest TSV path",
                       type = "character")
argv <- parse_args(parser)

# ----- Resolve per-study sources ----------------------------------------
studies <- data.frame(study_id = character(0),
                      gwas_tsv = character(0),
                      column_mapping = character(0),
                      stringsAsFactors = FALSE)
seenStudies <- character(0)

if (nzchar(argv$gwas_meta) && argv$gwas_meta != ".") {
  if (!file.exists(argv$gwas_meta))
    stop("--gwas-meta file not found: ", argv$gwas_meta)
  meta <- read.table(argv$gwas_meta, header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE, check.names = FALSE,
                     comment.char = "")
  required <- c("study_id", "path")
  missing <- setdiff(required, names(meta))
  if (length(missing) > 0L)
    stop("--gwas-meta missing required column(s): ",
         paste(missing, collapse = ", "), " (got: ",
         paste(names(meta), collapse = ", "), ").")
  metaDir <- dirname(normalizePath(argv$gwas_meta))
  hasCm <- "column_mapping" %in% names(meta)
  for (i in seq_len(nrow(meta))) {
    sid <- as.character(meta$study_id[[i]])
    tsv <- as.character(meta$path[[i]])
    if (!startsWith(tsv, "/")) tsv <- file.path(metaDir, tsv)
    cm  <- if (hasCm) as.character(meta$column_mapping[[i]]) else ""
    if (!is.na(cm) && nzchar(cm) && !startsWith(cm, "/"))
      cm <- file.path(metaDir, cm)
    if (sid %in% seenStudies)
      stop("Duplicate study_id in --gwas-meta: ", sid)
    seenStudies <- c(seenStudies, sid)
    studies <- rbind(studies,
      data.frame(study_id = sid, gwas_tsv = tsv,
                 column_mapping = if (is.na(cm)) "" else cm,
                 stringsAsFactors = FALSE))
  }
}

tsvItems <- as.character(argv$gwas_tsv_list)
tsvItems <- tsvItems[nzchar(tsvItems)]
for (item in tsvItems) {
  if (!grepl("=", item, fixed = TRUE))
    stop("--gwas-tsv-list expects STUDY=PATH items (got: ", item, ").")
  parts <- regmatches(item, regexec("^([^=]+)=(.+)$", item))[[1L]]
  if (length(parts) != 3L)
    stop("Cannot parse --gwas-tsv-list item: ", item)
  sid <- parts[[2L]]; tsv <- parts[[3L]]
  if (sid %in% seenStudies)
    stop("Study '", sid, "' appears in both --gwas-meta and ",
         "--gwas-tsv-list.")
  seenStudies <- c(seenStudies, sid)
  studies <- rbind(studies,
    data.frame(study_id = sid, gwas_tsv = tsv,
               column_mapping = "", stringsAsFactors = FALSE))
}
if (nrow(studies) == 0L)
  stop("No GWAS inputs supplied (give --gwas-meta and/or --gwas-tsv-list).")

# ----- Resolve regions --------------------------------------------------
regions <- data.frame(chr = character(0), start = integer(0),
                      end = integer(0), stringsAsFactors = FALSE)
pushRegion <- function(chr, start, end) {
  if (!startsWith(chr, "chr")) chr <- paste0("chr", chr)
  start <- as.integer(start); end <- as.integer(end)
  if (is.na(start) || is.na(end)) return(invisible(NULL))
  key <- paste(chr, start, end, sep = "|")
  prior <- paste(regions$chr, regions$start, regions$end, sep = "|")
  if (key %in% prior) return(invisible(NULL))
  regions <<- rbind(regions,
    data.frame(chr = chr, start = start, end = end,
               stringsAsFactors = FALSE))
}

if (nzchar(argv$region_list) && argv$region_list != ".") {
  if (!file.exists(argv$region_list))
    stop("--region-list file not found: ", argv$region_list)
  rl <- readLines(argv$region_list)
  for (line in rl) {
    line <- trimws(line)
    if (!nzchar(line) || startsWith(line, "#")) next
    parts <- strsplit(line, "\\s+")[[1L]]
    if (length(parts) < 3L) next
    if (is.na(suppressWarnings(as.integer(parts[[2L]])))) next  # header
    pushRegion(parts[[1L]], parts[[2L]], parts[[3L]])
  }
}

regionItems <- as.character(argv$regions)
regionItems <- regionItems[nzchar(regionItems)]
for (r in regionItems) {
  m <- regmatches(r, regexec("^([^:]+):([0-9]+)-([0-9]+)$", r))[[1L]]
  if (length(m) != 4L)
    stop("--regions item must be chr:start-end (got: ", r, ").")
  pushRegion(m[[2L]], m[[3L]], m[[4L]])
}
if (nrow(regions) == 0L)
  stop("No regions supplied (give --region-list and/or --regions).")

# ----- Build cross-product manifest -------------------------------------
manifest <- list()
for (i in seq_len(nrow(studies))) {
  for (j in seq_len(nrow(regions))) {
    chr <- regions$chr[[j]]; s <- regions$start[[j]]; e <- regions$end[[j]]
    region_id <- gsub("[:\\-]", "_", sprintf("%s_%d_%d", chr, s, e))
    manifest[[length(manifest) + 1L]] <- data.frame(
      study_id          = studies$study_id[[i]],
      gwas_tsv          = studies$gwas_tsv[[i]],
      column_mapping    = studies$column_mapping[[i]],
      chr               = chr, start = s, end = e,
      region_id         = region_id,
      gwas_tsv_basename = tools::file_path_sans_ext(
                            tools::file_path_sans_ext(basename(studies$gwas_tsv[[i]]))),
      stringsAsFactors  = FALSE)
  }
}
out <- do.call(rbind, manifest)
dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
write.table(out, file = argv$output, sep = "\t", quote = FALSE,
            row.names = FALSE, na = "")
cat(sprintf("Wrote manifest with %d row(s) (%d studies x %d regions) to %s\n",
            nrow(out), nrow(studies), nrow(regions), argv$output))
