#!/usr/bin/env Rscript
# ============================================================
# apa_impute.R
# Standalone CLI worker for apa_impute.ipynb steps.
#
# Steps (selected via --step):
#   impute       — merge per-chr DaPars2 results, KNN-impute + qnorm, write BEDs
#                  ([APAimpute]); handles the empty-result case gracefully
#   rename       — remap ProjID sample columns via a match table ([APArename_1/_3];
#                  --update-end recomputes end = start + 1 as in _1)
#   bgzip_index  — bgzip + tabix a BED, keeping the plain .bed ([APArename_2/_4])
#
# Conventions: read with vroom, manipulate with dplyr/tidyr, bgzip/tabix via
# Rsamtools (no data.table). CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(tidyr)
  library(stringr)
  library(dplyr)   # last
})

parser <- arg_parser("apa_impute worker (see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "impute | rename | bgzip_index")
parser <- add_argument(parser, "--cwd", type = "character", default = "",
                       help = "[impute] work dir holding apa_<chr>/ subdirs")
parser <- add_argument(parser, "--chrlist", type = "character", default = "",
                       help = "[impute] chromosomes, e.g. 'chr22'")
parser <- add_argument(parser, "--input", type = "character", default = "",
                       help = "[rename/bgzip_index] input BED")
parser <- add_argument(parser, "--match", type = "character", default = "",
                       help = "[rename] ProjID -> SampleID match table")
parser <- add_argument(parser, "--output", type = "character", default = "",
                       help = "[rename/bgzip_index] output path")
parser <- add_argument(parser, "--update-end", flag = TRUE,
                       help = "[rename] recompute end = start + 1")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# ---------------------------------------------------------------------------
impute_step <- function(argv) {
  suppressPackageStartupMessages(library(impute))
  chr <- stringr::str_extract_all(argv$chrlist, "chr\\d+")[[1]]
  apalist <- file.path(argv$cwd, paste0("apa_", chr),
                       paste0("Dapars_result_result_temp.", chr, ".txt"))
  merge_apa <- function(file) {
    df <- vroom::vroom(file, delim = "\t", col_types = cols(.default = "c"), show_col_types = FALSE)
    df %>% select(1, 2, 3, 4, 4 + order(colnames(df)[5:ncol(df)]))
  }
  dapars_result <- do.call("rbind", lapply(apalist, merge_apa))
  allchrom <- file.path(argv$cwd, "Dapars_allchrom.bed")

  if (is.null(dapars_result) || nrow(dapars_result) == 0) {
    message("APAimpute: DaPars2 result is empty (no APA events called); writing empty outputs.")
    empty_df <- setNames(data.frame(matrix(ncol = 4, nrow = 0)), c("#chr", "start", "end", "Gene"))
    vroom::vroom_write(empty_df, allchrom, delim = "\t")
    for (i in chr) {
      out <- file.path(argv$cwd, paste0("apa_", i), paste0("Dapars_result_impute_", i, ".bed"))
      dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
      write.table(empty_df, out, quote = FALSE, row.names = FALSE, sep = "\t")
    }
    return(invisible())
  }

  tmp <- dapars_result[, 1:4] %>%
    separate(Loci, into = c("#chr", "coords"), sep = ":", remove = FALSE) %>%
    separate(coords, into = c("start", "end"), sep = "-") %>%
    select(`#chr`, start, end, Gene)
  dapars_result <- dapars_result[, -c(2:4)]
  tmp_vec <- pull(dapars_result, Gene)
  dapars_result <- dapars_result[, -1]
  colnames(dapars_result) <- sub("_.*", "", basename(colnames(dapars_result)))
  dapars_result <- as.matrix(dapars_result)
  rownames(dapars_result) <- tmp_vec
  dapars_result <- dapars_result[, colMeans(is.na(dapars_result)) <= 0.8]
  dapars_result <- dapars_result[rowMeans(is.na(dapars_result)) < 0.5, ]
  class(dapars_result) <- "numeric"
  df <- as.data.frame(impute::impute.knn(dapars_result)$data)
  for (gene in seq_len(nrow(df))) {                       # per-gene rank -> qnorm
    mat <- apply(df[gene, ], 1, rank, ties.method = "average")
    df[gene, ] <- qnorm(mat / (ncol(df) + 1))
  }
  df$Gene <- rownames(dapars_result)
  final_data <- inner_join(tmp, df)
  vroom::vroom_write(final_data, allchrom, delim = "\t")
  for (i in chr) {
    out <- file.path(argv$cwd, paste0("apa_", i), paste0("Dapars_result_impute_", i, ".bed"))
    write.table(final_data %>% filter(`#chr` == i), out, quote = FALSE, row.names = FALSE)
  }
}

# ---------------------------------------------------------------------------
rename_step <- function(argv) {
  df  <- vroom::vroom(argv$input, delim = "\t", col_types = cols(.default = "c"), show_col_types = FALSE)
  ref <- vroom::vroom(argv$match, delim = "\t", col_types = cols(.default = "c"), show_col_types = FALSE)
  if (ncol(df) >= 5) {
    cols <- colnames(df)[5:ncol(df)]
    colnames(df)[5:ncol(df)] <- vapply(cols, function(i) as.character(ref[[2]][which(ref$ProjID == i)]),
                                       character(1))
  }
  df <- df %>% group_by(`#chr`) %>% arrange(start, .by_group = TRUE)
  if (isTRUE(argv$update_end)) df <- df %>% mutate(end = as.integer(start) + 1)
  vroom::vroom_write(ungroup(df), argv$output, delim = "\t")
}

# ---------------------------------------------------------------------------
bgzip_index_step <- function(argv) {
  suppressPackageStartupMessages(library(Rsamtools))
  Rsamtools::bgzip(argv$input, dest = argv$output, overwrite = TRUE)   # keeps the plain .bed
  Rsamtools::indexTabix(argv$output, format = "bed")
}

# ---------------------------------------------------------------------------
switch(argv$step,
  impute      = impute_step(argv),
  rename      = rename_step(argv),
  bgzip_index = bgzip_index_step(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
