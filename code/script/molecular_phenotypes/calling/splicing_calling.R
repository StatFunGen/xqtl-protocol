#!/usr/bin/env Rscript
# ============================================================
# splicing_calling.R
# Standalone CLI worker for splicing_calling.ipynb psichomics steps.
#
# Steps (selected via --step):
#   junction_quant  — psichomics::prepareJunctionQuant over STAR SJ.out.tab files
#                     -> merged junction quantification table  ([psichomics_1])
#   quantify_psi    — psichomics::quantifySplicing with a splicing annotation
#                     -> PSI raw-data table                    ([psichomics_2])
#
# Off-the-shelf psichomics; heavy libs load inside each step so --help works.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("splicing_calling worker (psichomics; see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "junction_quant | quantify_psi")
parser <- add_argument(parser, "--inputs", type = "character", nargs = Inf,
                       help = "[junction_quant] STAR SJ.out.tab files")
parser <- add_argument(parser, "--junctions", type = "character", default = "",
                       help = "[quantify_psi] junction quantification table (junction_quant output)")
parser <- add_argument(parser, "--splicing-annotation", type = "character", default = "",
                       help = "[quantify_psi] psichomics annotation RDS")
parser <- add_argument(parser, "--output", type = "character", help = "output file")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# ---------------------------------------------------------------------------
# junction_quant ([psichomics_1])
# ---------------------------------------------------------------------------
junction_quant <- function(argv) {
  suppressPackageStartupMessages({ library(psichomics); library(dplyr); library(tidyr); library(purrr) })
  # preserve which NAs were present originally vs. those introduced by the 0-pad merge
  df_replace <- function(df, old, new) {
    if (is.na(old)) df[is.na(df)] <- new else df[df == old] <- new
    df
  }
  options(scipen = 15)                                   # avoid scientific notation in coordinates

  # group input files by their directory (prepareJunctionQuant reads bare filenames)
  files <- list()
  for (f in argv$inputs) {
    filename  <- gsub("^.*/", "", f)
    directory <- gsub(filename, "", f)
    if (length(files[[directory]]) == 0) files[[directory]] <- filename
    else files[[directory]] <- append(files[[directory]], filename)
  }

  if (length(files) == 1) {
    setwd(names(files)[1])
    prepareJunctionQuant(files[[1]], output = argv$output)
  } else {
    stem <- sub("\\.txt$", "", argv$output)
    batch_junction_list <- list()
    for (d in names(files)) {
      output_name <- sprintf("%s_%s.txt", stem, tail(strsplit(d, "/")[[1]], 1))
      setwd(d)
      prepareJunctionQuant(files[[d]], output = output_name)
      batch_junction_list <- append(batch_junction_list, output_name)
    }
    res <- list()
    for (file_name in batch_junction_list) res[[file_name]] <- read.table(file_name, sep = "\t", header = TRUE)
    res <- lapply(res, df_replace, old = NA, new = "original_na")   # mark real NAs
    res <- res %>% reduce(full_join, by = "Junction.ID")
    res[is.na(res)] <- 0                                            # pad merge-introduced NAs with 0
    res <- df_replace(res, "original_na", NA)                      # restore real NAs
    write.table(res, file = argv$output, quote = FALSE, sep = "\t", row.names = FALSE)
  }
  options(scipen = 0)
}

# ---------------------------------------------------------------------------
# quantify_psi ([psichomics_2])
# ---------------------------------------------------------------------------
quantify_psi <- function(argv) {
  suppressPackageStartupMessages({ library(psichomics); library(dplyr); library(tidyr); library(purrr) })
  data <- read.table(argv$junctions, sep = "\t", header = TRUE)
  names(data) <- sub("X", "", names(data))               # read.table prefixes numeric sample names with X
  # drop mitochondrial / unplaced / random-contig junctions. Use grepl-negation:
  # `data[-grep(...), ]` wipes ALL rows when a pattern has zero matches
  # (`-integer(0)` indexes nothing), which breaks inputs lacking those contigs.
  data <- data[!grepl("chrM|chrUn|random", data$Junction.ID), ]
  junctionQuant <- data[, -1]
  rownames(junctionQuant) <- data[, 1]

  annotation <- readRDS(argv$splicing_annotation)
  psi <- quantifySplicing(annotation, junctionQuant)
  write.table(psi, file = argv$output, quote = FALSE, sep = "\t")
}

# ---------------------------------------------------------------------------
switch(argv$step,
  junction_quant = junction_quant(argv),
  quantify_psi   = quantify_psi(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
