#!/usr/bin/env Rscript
# ctwas_assemble.R
#
# cTWAS step 1: per-region manifest → assembled cTWAS inputs.
# Reads the per-region manifest, loads each block's GwasSumStats /
# TwasWeights RDS, and calls pecotmr::assembleCtwasInputs() to build
# the named list of ctwas-shape inputs (z_snp, weights, region_info,
# snp_map, LD_map, LD/snpInfo loader closures). The result is saved
# to a single RDS that ctwas_est.R consumes downstream.
#
# Inputs:
#   --manifest             Per-region manifest TSV with columns:
#                            region_id              (string, unique)
#                            gwas_sumstats_rds      (per-block GwasSumStats RDS)
#                            twas_weights_rds       (comma-sep per-gene TwasWeights RDS; may be empty)
#                            fine_mapping_result_rds (optional, comma-sep)
#   --method               Which TWAS method to feed into ctwas
#                          (default: NULL — resolves to 'ensemble' if
#                          present, or sole method, or errors)
#   --twas-z               Optional TWAS-Z GRanges RDS
#   --twas-weight-cutoff   Pass-through (default 0)
#   --cs-min-cor           Pass-through (default 0.8)
#   --min-pip-cutoff       Pass-through (default 0)
#   --max-num-variants     Pass-through (default Inf)
#   --output               Output RDS path (the assembled `inputs` list)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(S4Vectors)
})

parser <- arg_parser("cTWAS step 1: assemble inputs from per-region manifest")
parser <- add_argument(parser, "--manifest",
                       help = "Per-region manifest TSV (region_id, gwas_sumstats_rds, twas_weights_rds[, fine_mapping_result_rds])",
                       type = "character")
parser <- add_argument(parser, "--method",
                       help = "Which TWAS method to feed into ctwas (defaults to 'ensemble' if present, or sole method)",
                       type = "character", default = "")
parser <- add_argument(parser, "--twas-z",
                       help = "Optional TWAS-Z GRanges RDS",
                       type = "character", default = "")
parser <- add_argument(parser, "--twas-weight-cutoff",
                       help = "Drop weight variants with |w| below this",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--cs-min-cor",
                       help = "CS purity floor for must-keep rescue",
                       type = "numeric", default = 0.8)
parser <- add_argument(parser, "--min-pip-cutoff",
                       help = "PIP rescue threshold",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--max-num-variants",
                       help = "Per-gene variant cap",
                       type = "numeric", default = Inf)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path (assembled cTWAS inputs)",
                       type = "character")
argv <- parse_args(parser)

split_paths <- function(s) {
  if (is.na(s) || !nzchar(s)) return(character(0))
  trimws(strsplit(s, ",", fixed = TRUE)[[1L]])
}

manifest <- read.table(argv$manifest, header = TRUE, sep = "\t",
                       stringsAsFactors = FALSE, check.names = FALSE,
                       comment.char = "")
required <- c("region_id", "gwas_sumstats_rds", "twas_weights_rds")
missing <- setdiff(required, names(manifest))
if (length(missing) > 0L)
  stop("Manifest missing required column(s): ",
       paste(missing, collapse = ", "))
if (anyDuplicated(manifest$region_id))
  stop("Manifest has duplicate region_id values.")
if (nrow(manifest) < 2L)
  stop("Manifest must list at least two LD blocks. cTWAS's EM cannot ",
       "converge on a single region.")

gwasSumStatsByRegion <- list()
twasWeightsByRegion  <- list()
fmrByRegion          <- list()
for (i in seq_len(nrow(manifest))) {
  rid <- manifest$region_id[[i]]
  gwasSumStatsByRegion[[rid]] <- readRDS(manifest$gwas_sumstats_rds[[i]])
  tw_paths <- split_paths(manifest$twas_weights_rds[[i]])
  if (length(tw_paths) > 0L) {
    tw_list <- lapply(tw_paths, readRDS)
    twasWeightsByRegion[[rid]] <- if (length(tw_list) == 1L) tw_list[[1L]]
                                  else Reduce(function(a, b)
                                                 pecotmr:::.rbindTwasWeights(a, b),
                                               tw_list)
  }
  if ("fine_mapping_result_rds" %in% names(manifest)) {
    fmr_paths <- split_paths(manifest$fine_mapping_result_rds[[i]])
    if (length(fmr_paths) > 0L) {
      fmr_list <- lapply(fmr_paths, readRDS)
      fmrByRegion[[rid]] <- if (length(fmr_list) == 1L) fmr_list[[1L]]
                            else Reduce(function(a, b)
                                          pecotmr:::.rbindFineMappingResult(a, b),
                                        fmr_list)
    }
  }
}
if (length(twasWeightsByRegion) == 0L)
  stop("Manifest yielded zero TwasWeights across all blocks.")

# Optional precomputed TWAS-Z and fineMappingResult (single-object).
tz <- if (nzchar(argv$twas_z) && argv$twas_z != "." && file.exists(argv$twas_z))
        readRDS(argv$twas_z) else NULL
fmr <- if (length(fmrByRegion) > 0L) {
  if (length(fmrByRegion) == 1L) fmrByRegion[[1L]]
  else Reduce(function(a, b) pecotmr:::.rbindFineMappingResult(a, b),
              unname(fmrByRegion))
} else NULL

inputs <- assembleCtwasInputs(
  gwasSumStats     = gwasSumStatsByRegion,
  twasWeights      = twasWeightsByRegion,
  twasZ            = tz,
  fineMappingResult = fmr,
  method           = if (nzchar(argv$method)) argv$method else NULL,
  twasWeightCutoff = argv$twas_weight_cutoff,
  csMinCor         = argv$cs_min_cor,
  minPipCutoff     = argv$min_pip_cutoff,
  maxNumVariants   = argv$max_num_variants)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(inputs, argv$output)
cat(sprintf("Wrote assembled cTWAS inputs (%d regions, %d weights) to %s\n",
            nrow(inputs$region_info), length(inputs$weights), argv$output))
