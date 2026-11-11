#!/usr/bin/env Rscript
# ============================================================
# torino_factorize.R
# Torino factorization of one gene's depth matrix, for the torino_3 step of
# isoform_calling_torino.ipynb. Replaces run_torino_factorization.R from the
# Torino package (cfbuenabadn/torino, inst/scripts), whose positional arguments
# and TORINO_FILTER_SAMPLES variable do not fit the pipeline's named-flag wrappers.
#
# A thin call to torino::run_torino_gene: samples whose summed depth is below
# --min-counts are dropped, consecutive bases are summed in groups of --merge-n to
# bound memory, and the matrix is factorized into --K coverage components by
# empirical Bayes Poisson matrix factorization.
#
# --merge-n auto applies the package's rule for the matrix width: 3 above 15 kb,
# 9 above 60 kb, 30 above 1 Mb, otherwise 1.
#
# Writes --output, an RDS list with $gene, $torino_K<K> and $coords_all (see
# ?torino::run_torino_gene). Torino returns NULL when the factorization fails,
# for instance when fewer samples than --K pass --min-counts; that NULL is saved
# with a warning rather than stopping the run, and torino_usage.R skips it.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("Torino factorization of one gene")
parser <- add_argument(parser, "--counts", type = "character", help = "<gene_id>.csv.gz from torino_count_matrix.R")
parser <- add_argument(parser, "--output", type = "character", help = "RDS to write")
parser <- add_argument(parser, "--gene", type = "character", default = NA,
                       help = "gene ID stored in the RDS [default: --counts file name]")
parser <- add_argument(parser, "--K", type = "integer", default = 3L, help = "number of coverage components")
parser <- add_argument(parser, "--min-counts", type = "integer", default = 100L,
                       help = "summed depth below which a sample is dropped")
parser <- add_argument(parser, "--merge-n", type = "character", default = "auto",
                       help = "bases summed per column, or auto")
parser <- add_argument(parser, "--seed", type = "integer", default = 1L, help = "random seed")
argv <- parse_args(parser)
for (a in c("counts", "output")) if (is.na(argv[[a]])) stop("--", a, " is required")

die <- function(...) { message("ERROR: ", ...); quit(status = 1) }
if (is.na(argv$K) || argv$K < 1) die("--K must be a positive integer")
merge_n <- argv$merge_n
if (merge_n != "auto" && !grepl("^[1-9][0-9]*$", merge_n)) die("--merge-n must be a positive integer or auto")
if (!requireNamespace("torino", quietly = TRUE))
    die("the torino R package is not installed (https://github.com/cfbuenabadn/torino)")

gene <- if (is.na(argv$gene)) sub("\\.csv(\\.gz)?$", "", basename(argv$counts)) else argv$gene
fit <- torino::run_torino_gene(gene_name = gene, K = argv$K, counts_path = argv$counts,
                               out_path = argv$output, min_counts = argv$min_counts,
                               n = if (merge_n == "auto") "auto" else as.integer(merge_n),
                               seed = argv$seed)
