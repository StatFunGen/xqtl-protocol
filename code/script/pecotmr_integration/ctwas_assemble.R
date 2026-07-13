#!/usr/bin/env Rscript
# ctwas_assemble.R
#
# cTWAS step 1: LD-block grid + flat weights -> assembled cTWAS inputs.
# Reads the block-grid manifest, loads each block's GwasSumStats RDS, loads the
# FLAT set of per-gene weight RDS, and calls pecotmr::assembleCtwasInputs(),
# which places each gene into its home LD block internally from the `region`
# provenance (matching cTWAS's p0 rule) and builds the ctwas-shape input set
# (z_snp, weights, region_info, snp_map, LD_map, LD/snpInfo loader closures).
# The result is saved to a single RDS that ctwas_est.R consumes downstream.
#
# The weights are NO LONGER bucketed per block in the manifest: they are handed
# in flat and placed by pecotmr. The weight source may be TwasWeights or
# QtlFineMappingResult objects (the latter uses each gene's topLoci posterior
# effect as its weight).
#
# Inputs:
#   --manifest             Block-grid manifest TSV with columns:
#                            region_id          (string, unique)
#                            gwas_sumstats_rds  (per-block GwasSumStats RDS)
#   --twas-weights         Comma-separated FLAT per-gene weight RDS
#                          (TwasWeights or QtlFineMappingResult; each carries
#                          `region` provenance for placement)
#   --fine-mapping-results Optional comma-separated FineMappingResult RDS used
#                          only as the CS / PIP rescue-filter source
#                          (NOT the weight source)
#   --method               Which TWAS method to feed into ctwas
#                          (default: NULL — resolves to 'ensemble' if present,
#                          or the sole method, or errors)
#   --twas-z               Optional TWAS-Z GRanges RDS
#   --twas-weight-cutoff   Pass-through (default 0)
#   --cs-min-cor           Pass-through (default 0.8)
#   --min-pip-cutoff       Pass-through (default 0)
#   --max-num-variants     Pass-through (default Inf)
#   --output               Output RDS path (the assembled `inputs` list)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("cTWAS step 1: assemble inputs from block grid + flat weights")
parser <- add_argument(parser, "--manifest",
                       help = "Block-grid manifest TSV (region_id, gwas_sumstats_rds)",
                       type = "character")
parser <- add_argument(parser, "--twas-weights",
                       help = "Comma-separated FLAT per-gene weight RDS (TwasWeights or QtlFineMappingResult)",
                       type = "character")
parser <- add_argument(parser, "--fine-mapping-results",
                       help = "Optional comma-separated FineMappingResult RDS (CS/PIP filter source)",
                       type = "character", default = "")
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
required <- c("region_id", "gwas_sumstats_rds")
missing <- setdiff(required, names(manifest))
if (length(missing) > 0L)
  stop("Manifest missing required column(s): ", paste(missing, collapse = ", "))
if (anyDuplicated(manifest$region_id))
  stop("Manifest has duplicate region_id values.")
if (nrow(manifest) < 2L)
  stop("Manifest must list at least two LD blocks. cTWAS's EM cannot ",
       "converge on a single region.")

# Per-block GWAS sum-stats keyed by region_id (the LD-block grid).
gwasSumStatsByRegion <- list()
for (i in seq_len(nrow(manifest))) {
  rid <- manifest$region_id[[i]]
  gwasSumStatsByRegion[[rid]] <- readRDS(manifest$gwas_sumstats_rds[[i]])
}

# FLAT weight source: an unnamed list of per-gene weight objects. assembleCtwasInputs
# combines them and places each gene into its home block by `region`.
weightPaths <- split_paths(argv$twas_weights)
if (length(weightPaths) == 0L)
  stop("--twas-weights lists no weight RDS paths.")
weightObjs <- lapply(weightPaths, readRDS)

# Optional precomputed TWAS-Z and the CS/PIP-filter FineMappingResult.
tz <- if (nzchar(argv$twas_z) && argv$twas_z != "." && file.exists(argv$twas_z))
        readRDS(argv$twas_z) else NULL
fmrPaths <- split_paths(argv$fine_mapping_results)
fmr <- if (length(fmrPaths) > 0L)
         combineFineMappingResults(lapply(fmrPaths, readRDS)) else NULL

inputs <- assembleCtwasInputs(
  gwasSumStats      = gwasSumStatsByRegion,
  twasWeights       = weightObjs,
  twasZ             = tz,
  fineMappingResult = fmr,
  method            = if (nzchar(argv$method)) argv$method else NULL,
  twasWeightCutoff  = argv$twas_weight_cutoff,
  csMinCor          = argv$cs_min_cor,
  minPipCutoff      = argv$min_pip_cutoff,
  maxNumVariants    = argv$max_num_variants)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(inputs, argv$output)
cat(sprintf("Wrote assembled cTWAS inputs (%d regions, %d weights) to %s\n",
            nrow(inputs$region_info), length(inputs$weights), argv$output))
