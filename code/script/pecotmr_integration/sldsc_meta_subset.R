#!/usr/bin/env Rscript
# sldsc_meta_subset.R
#
# Re-run the DerSimonian-Laird random-effects meta on a user-defined subset
# of traits, reusing the cached per-trait standardized results from
# sldsc_postprocess.R. No regression rerun -- this only re-meta's the
# already-standardized per-trait tables via pecotmr::metaSldscRandom().
#
# Inputs:
#   --postprocess-rds <RDS>      Output of sldsc_postprocess.R.
#   --subset-traits-file <txt>   One trait id per line; a subset of the traits
#                                passed to sldsc_postprocess.R.
#   --target-categories c1,c2    Target annotation names to meta on. When empty,
#                                uses params$target_categories from the RDS.
#                                (If sldsc_postprocess.R ran with display
#                                labels, that field already holds the labels.)
#   --output <RDS>               Subset meta result RDS.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Re-run sLDSC random-effects meta on a subset of traits")
parser <- add_argument(parser, "--postprocess-rds",
                       help = "Output RDS of sldsc_postprocess.R",
                       type = "character")
parser <- add_argument(parser, "--subset-traits-file",
                       help = "Text file: one trait id per line",
                       type = "character")
parser <- add_argument(parser, "--target-categories",
                       help = "Comma-separated target annotation names",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output subset-meta RDS path",
                       type = "character")
argv <- parse_args(parser)

res <- readRDS(argv$postprocess_rds)

subsetTraits <- readLines(argv$subset_traits_file)
subsetTraits <- subsetTraits[nzchar(trimws(subsetTraits))]

# When --target-categories is empty, sldscSubsetMeta falls back to
# params$target_categories. The view-building, the per-category metaSldscRandom
# grid, and the missing-trait check all live in pecotmr::sldscSubsetMeta.
targetCats <- if (nzchar(argv$target_categories)) {
  trimws(strsplit(argv$target_categories, ",", fixed = TRUE)[[1L]])
} else NULL

out <- sldscSubsetMeta(res, subsetTraits, targetCategories = targetCats)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(out, argv$output)
message("Subset meta complete; results written to ", argv$output)
