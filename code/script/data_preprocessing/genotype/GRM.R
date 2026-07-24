#!/usr/bin/env Rscript
# GRM.R — GRM.ipynb workers (APEX formatting of the GCTA LOCO-GRM).
#
# Steps:
#   format_apex  Map the GCTA gzipped GRM (integer sample indices) to sample IDs
#                and emit the APEX-format table (#id1, id2, kinship).
#   apex_list    Assemble the per-chromosome APEX GRM file list for APEX.
#
# gcta itself (grm_2) stays a thin inline call in the notebook; only the R
# reformatting logic lives here (vroom/dplyr, per repo convention).

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
})

do_format_apex <- function(argv) {
  # GCTA .grm.gz columns: idx1, idx2, n_snps, kinship  (lower triangle)
  grm <- vroom(argv$grm, col_names = c("i1", "i2", "n", "kinship"),
               col_types = "iidd", delim = "\t")
  # .grm.id is FID, IID — sample IDs are column 2, indexed by the GRM indices
  ids <- vroom(argv$id, col_names = c("fid", "iid"), col_types = "cc", delim = "\t")$iid
  out <- tibble(`#id1` = ids[grm$i1], id2 = ids[grm$i2], kinship = grm$kinship)
  vroom_write(out, argv$output, delim = "\t", quote = "none")
}

do_apex_list <- function(argv) {
  files <- argv$input[!is.na(argv$input)]
  chroms <- argv$chroms[!is.na(argv$chroms)]
  # matches the original: a single concatenated chr label, one row per GRM file
  chr_label <- paste0("chr", chroms, collapse = "")
  out <- tibble(`#chr` = chr_label, dir = files) %>% arrange(.data$`#chr`)
  vroom_write(out, argv$output, delim = "\t", quote = "none")
}

p <- arg_parser("GRM APEX formatting (format_apex / apex_list)")
p <- add_argument(p, "--step", help = "format_apex | apex_list")
p <- add_argument(p, "--grm", help = "format_apex: GCTA .grm.gz")
p <- add_argument(p, "--id", help = "format_apex: GCTA .grm.id")
p <- add_argument(p, "--input", help = "apex_list: per-chrom .apex.grm files", nargs = Inf)
p <- add_argument(p, "--chroms", help = "apex_list: chromosome labels", nargs = Inf)
p <- add_argument(p, "--output", help = "output file")
argv <- parse_args(p)

if (identical(argv$step, "format_apex")) {
  do_format_apex(argv)
} else if (identical(argv$step, "apex_list")) {
  do_apex_list(argv)
} else {
  stop("--step must be 'format_apex' or 'apex_list'")
}
