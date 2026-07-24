#!/usr/bin/env Rscript
# genotype_formatting.R — R port of genotype_formatting.py (the non-shell helpers).
#
# Steps:
#   ld_by_region_plink_1  plink `--r square0 --make-just-bim` over a region, then
#                         save the LD matrix + variant IDs as an RDS (was a numpy
#                         .npz; switched to RDS since nothing in-repo reads the .npz).
#   write_data_list       Filter a "::"-joined file list to non-empty entries and
#                         write a two-column (#id, #path) manifest.
#   vcf_gz_summary        Print row/column/header/preview stats for VCF(.gz) files.
#   file_size_summary     Print file path + human-readable size.

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
})

human_readable_size <- function(path) {
  value <- as.numeric(file.info(path)$size)
  for (u in c("B", "K", "M", "G", "T", "P")) {
    if (value < 1024 || u == "P") {
      return(if (u == "B") sprintf("%d%s", as.integer(value), u) else sprintf("%.1f%s", value, u))
    }
    value <- value / 1024
  }
}

read_lines_any <- function(path) {
  con <- if (grepl("\\.gz$", path)) gzfile(path) else file(path)
  on.exit(close(con)); readLines(con)
}

do_ld_by_region <- function(argv) {
  stopifnot(nzchar(argv$genoFile), nzchar(argv$output))
  for (n in c("region_chrom", "region_start", "region_end")) {
    if (!nzchar(argv[[n]])) stop(sprintf("--%s is required", gsub("_", "-", n)))
  }
  out <- argv$output
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  out_prefix <- sub("\\.[^.]*\\.[^.]*$", "", out)          # strip .floatN.rds -> prefix
  geno_prefix <- sub("\\.[^.]*$", "", argv$genoFile)       # strip .bed

  system2("plink", c("--bfile", geno_prefix, "--out", out_prefix,
                     "--chr", argv$region_chrom, "--from-bp", argv$region_start,
                     "--to-bp", argv$region_end, "--r", "square0",
                     "--make-just-bim", "--threads", argv$numThreads))

  ld <- as.matrix(vroom(paste0(out_prefix, ".ld"), col_names = FALSE,
                        delim = "\t", show_col_types = FALSE))
  dimnames(ld) <- NULL
  bim <- vroom(paste0(out_prefix, ".bim"), col_names = FALSE, delim = "\t", show_col_types = FALSE)
  saveRDS(list(ld = ld, variant_ids = bim[[2]]), out)      # RDS replaces the numpy .npz
  cat(sprintf("output_info: %s\noutput_size: %s\noutput_variants: %d\n",
              out, human_readable_size(out), length(bim[[2]])))
}

do_write_data_list <- function(argv) {
  stopifnot(nzchar(argv$output))
  files <- Filter(nzchar, strsplit(argv$data_files, "::", fixed = TRUE)[[1]])
  if (!length(files)) stop("--data-files is required")
  n <- length(strsplit(argv$ext, ".", fixed = TRUE)[[1]]) + 1
  ids <- character(0); keep <- character(0)
  for (f in files) {
    parts <- strsplit(f, ".", fixed = TRUE)[[1]]
    fid <- parts[length(parts) - n + 1]
    sz <- file.info(f)$size
    if (!is.na(sz) && sz > 0) { keep <- c(keep, f); ids <- c(ids, fid) }
    else message(sprintf("Empty file found: %s", f))
  }
  if (!length(keep)) stop("No non-empty files found. Exiting.")
  dir.create(dirname(argv$output), recursive = TRUE, showWarnings = FALSE)
  vroom_write(data.frame(`#id` = ids, `#path` = keep, check.names = FALSE),
              argv$output, delim = "\t", quote = "none")
}

do_vcf_gz_summary <- function(files) {
  for (f in files) {
    lines <- read_lines_any(f)
    body <- lines[!startsWith(lines, "##")]
    ncols <- if (length(body)) length(strsplit(body[1], "[ \t]+")[[1]]) else 0
    preview <- vapply(utils::head(body, 10),
                      function(l) paste(utils::head(strsplit(l, "\t", fixed = TRUE)[[1]], 11), collapse = "\t"),
                      character(1))
    cat(sprintf("output_info: %s\noutput_size: %s\noutput_rows: %d\noutput_column: %d\noutput_header_row: %d\noutput_preview:\n%s\n",
                f, human_readable_size(f), length(lines), ncols, sum(startsWith(lines, "##")),
                paste(preview, collapse = "\n")))
  }
}

do_file_size_summary <- function(files) {
  for (f in files) cat(sprintf("output_info: %s \noutput_size: %s\n", f, human_readable_size(f)))
}

p <- arg_parser("genotype_formatting helpers (R port)")
p <- add_argument(p, "--step", help = "ld_by_region_plink_1 | write_data_list | vcf_gz_summary | file_size_summary")
p <- add_argument(p, "--genoFile", help = "ld_by_region: PLINK bed", default = "")
p <- add_argument(p, "--region-chrom", help = "ld_by_region: chromosome", default = "")
p <- add_argument(p, "--region-start", help = "ld_by_region: region start bp", default = "")
p <- add_argument(p, "--region-end", help = "ld_by_region: region end bp", default = "")
p <- add_argument(p, "--float-type", help = "filename precision tag", default = "16")
p <- add_argument(p, "--output", help = "output path", default = "")
p <- add_argument(p, "--data-files", help = "\"::\"-joined file list", default = "")
p <- add_argument(p, "--ext", help = "write_data_list: file extension", default = "")
p <- add_argument(p, "--numThreads", help = "plink threads", default = "8")
argv <- parse_args(p)

files_or_output <- function() {
  f <- Filter(nzchar, strsplit(argv$data_files, "::", fixed = TRUE)[[1]])
  if (!length(f) && nzchar(argv$output)) f <- argv$output
  if (!length(f)) stop("--data-files or --output is required")
  f
}

if (identical(argv$step, "ld_by_region_plink_1")) {
  do_ld_by_region(argv)
} else if (identical(argv$step, "write_data_list")) {
  do_write_data_list(argv)
} else if (identical(argv$step, "vcf_gz_summary")) {
  do_vcf_gz_summary(files_or_output())
} else if (identical(argv$step, "file_size_summary")) {
  do_file_size_summary(files_or_output())
} else {
  stop(sprintf("Unknown step: %s", argv$step))
}
