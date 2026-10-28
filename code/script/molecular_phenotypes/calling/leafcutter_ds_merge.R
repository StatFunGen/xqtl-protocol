#!/usr/bin/env Rscript
# ============================================================
# leafcutter_ds_merge.R
# Join per-part leafcutter_ds.R results and recompute the FDR over all clusters,
# for the differential_splicing_3 step of splicing_calling_leafcutter2.ipynb.
#
# Each part was tested separately, so its p.adjust column was corrected only within
# that part. The p-values themselves do not depend on the split, so the column is
# recomputed here with p.adjust(p, method = "fdr") over every cluster that has a
# p-value, as leafcutter_ds.R does in an unsplit run. Clusters without a p-value
# keep NA.
#
# Each <part>_cluster_significance.txt is read with its <part>_effect_sizes.txt
# sibling. Other paths are ignored, so a step's whole output list can be passed.
# Empty files (parts with no clusters) are skipped. Writes
# <output-prefix>_cluster_significance.txt and <output-prefix>_effect_sizes.txt.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("Join per-part leafcutter_ds.R results and recompute the FDR")
parser <- add_argument(parser, "--output-prefix", type = "character", help = "<dir>/<prefix> for the joined tables")
parser <- add_argument(parser, "--files", nargs = Inf, help = "per-part *_cluster_significance.txt files")
argv <- parse_args(parser)
if (is.na(argv$output_prefix)) stop("--output-prefix is required")
if (all(is.na(argv$files))) stop("--files is required")

die <- function(...) { message("ERROR: ", ...); quit(status = 1) }

SIG <- "_cluster_significance.txt"
EFF <- "_effect_sizes.txt"

# Header and body lines of a table; NULL header for an empty file.
read_table <- function(path) {
  lines <- vroom::vroom_lines(path)
  lines <- lines[nzchar(trimws(lines))]
  if (!length(lines)) list(header = NULL, body = character(0))
  else list(header = lines[[1L]], body = lines[-1L])
}

sig_files <- argv$files[endsWith(argv$files, SIG)]
if (!length(sig_files)) die("no *", SIG, " files given")

sig_hdr <- eff_hdr <- NULL
rows <- list()
eff_rows <- character(0)
used <- 0L
for (s in sig_files) {
  e <- paste0(substr(s, 1L, nchar(s) - nchar(SIG)), EFF)
  if (!file.exists(e)) die(e, " is missing")
  sig <- read_table(s)
  eff <- read_table(e)
  if (is.null(sig$header)) next
  if (!is.null(sig_hdr) && sig$header != sig_hdr) die(s, " has a different header")
  if (!is.null(eff$header) && !is.null(eff_hdr) && eff$header != eff_hdr)
    die(e, " has a different header (were the groups the same?)")
  sig_hdr <- sig$header
  if (!is.null(eff$header)) eff_hdr <- eff$header
  rows <- c(rows, strsplit(sig$body, "\t", fixed = TRUE))
  eff_rows <- c(eff_rows, eff$body)
  used <- used + 1L
}
if (is.null(sig_hdr)) die("every part was empty; nothing was tested")

cols <- strsplit(sig_hdr, "\t", fixed = TRUE)[[1]]
ip <- match("p", cols)
iq <- match("p.adjust", cols)
p <- vapply(rows, `[[`, "", ip)
tested <- !p %in% c("NA", "")
q <- rep("NA", length(rows))
adj <- p.adjust(as.numeric(p[tested]), method = "fdr")
q[tested] <- as.character(adj)
for (i in seq_along(rows)) rows[[i]][[iq]] <- q[[i]]

vroom::vroom_write_lines(c(sig_hdr, vapply(rows, paste, "", collapse = "\t")),
                         paste0(argv$output_prefix, SIG))
vroom::vroom_write_lines(c(if (is.null(eff_hdr)) "" else eff_hdr, eff_rows),
                         paste0(argv$output_prefix, EFF))
cat("parts with clusters ", used, " of ", length(sig_files), "; clusters ", length(rows),
    "; tested ", sum(tested), "; FDR < 0.05 ", sum(adj < 0.05), "\n", sep = "")
