#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("usage: check_rss_mwe_output.R <rds> <png> <expected-region-key>")
}

rds_path <- args[[1]]
png_path <- args[[2]]
expected_key <- args[[3]]

fail <- function(...) {
  stop(paste0(...), call. = FALSE)
}

if (!file.exists(rds_path)) {
  fail("Missing RSS RDS output: ", rds_path)
}
if (!file.exists(png_path)) {
  fail("Missing RSS plot output: ", png_path)
}

result <- readRDS(rds_path)
if (!identical(names(result), expected_key)) {
  fail(
    "Expected top-level RDS key ", expected_key,
    "; observed ", paste(names(result), collapse = ",")
  )
}

region_result <- result[[expected_key]]
if (!is.list(region_result) || length(region_result) == 0) {
  fail("Region result is empty for key ", expected_key)
}

study_result <- region_result[[1]]
if (is.null(study_result$rss_data_analyzed)) {
  fail("Missing rss_data_analyzed in first study result")
}
if (!is.data.frame(study_result$rss_data_analyzed) ||
    nrow(study_result$rss_data_analyzed) == 0) {
  fail("rss_data_analyzed is empty")
}

png_size <- file.info(png_path)$size
if (is.na(png_size) || png_size == 0) {
  fail("RSS plot output is empty: ", png_path)
}

cat(
  "rss_mwe_assertions key=", expected_key,
  " rss_rows=", nrow(study_result$rss_data_analyzed),
  " rss_cols=", ncol(study_result$rss_data_analyzed),
  " png_bytes=", png_size,
  "\n",
  sep = ""
)
