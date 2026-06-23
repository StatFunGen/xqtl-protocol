#!/usr/bin/env Rscript
# twas_weights.R
#
# Per-region TWAS-weights worker. Loads a pre-built pecotmr::QtlDataset
# RDS and calls twasWeightsPipeline() for either a single trait (gene
# mode) or a single genomic region (region mode). Designed to be
# invoked once per fan-out unit by the SoS step in twas_weights.ipynb.
#
# Modes (mutually exclusive):
#   gene   : --gene-id ENSG... --cis-window 1000000
#   region : --region chr22:15000000-16000000
#
# Inputs:
#   --qtl-dataset           Path to a QtlDataset RDS
#   --gene-id               (gene mode) trait identifier
#   --cis-window            (gene mode) cis-window in bp
#   --region                (region mode) chr:start-end string
#   --fine-mapping-result   Optional pre-fit FineMappingResult RDS;
#                           SuSiE-family methods reuse the cached fits
#                           via the fineMappingResult cache. Pass "" or
#                           "." to skip.
#   --output                Output RDS path (one TwasWeights)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
})

parser <- arg_parser("Per-gene or per-region default-preset TWAS weights over a pre-built QtlDataset")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS",
                       type = "character")
parser <- add_argument(parser, "--gene-id",
                       help = "Trait identifier (gene mode); mutually exclusive with --region",
                       type = "character", default = "")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the trait's TSS (gene mode)",
                       type = "integer", default = 1000000L)
parser <- add_argument(parser, "--region",
                       help = "Genomic region as chr:start-end (region mode); mutually exclusive with --gene-id",
                       type = "character", default = "")
parser <- add_argument(parser, "--fine-mapping-result",
                       help = "Optional pre-fit FineMappingResult RDS",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

parse_region <- function(s) {
  m <- regmatches(s, regexec("^([^:]+):([0-9]+)-([0-9]+)$", s))[[1L]]
  if (length(m) != 4L)
    stop("--region must be in chr:start-end format (got: ", s, ")")
  GRanges(seqnames = m[[2L]],
          ranges   = IRanges(start = as.integer(m[[3L]]),
                             end   = as.integer(m[[4L]])))
}

has_gene   <- nzchar(argv$gene_id)
has_region <- nzchar(argv$region)
if (has_gene && has_region)
  stop("--gene-id and --region are mutually exclusive; pass exactly one.")
if (!has_gene && !has_region)
  stop("Specify either --gene-id (with --cis-window) or --region.")

qd <- readRDS(argv$qtl_dataset)

fmr_path <- argv$fine_mapping_result
fmr <- if (nzchar(fmr_path) && fmr_path != "." && file.exists(fmr_path)) {
  readRDS(fmr_path)
} else {
  NULL
}

res <- if (has_region) {
  twasWeightsPipeline(
    qd,
    methods           = NULL,
    region            = parse_region(argv$region),
    cisWindow         = argv$cis_window,
    fineMappingResult = fmr)
} else {
  twasWeightsPipeline(
    qd,
    methods           = NULL,
    traitId           = argv$gene_id,
    cisWindow         = argv$cis_window,
    fineMappingResult = fmr)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote TWAS weights for %s (%d row(s)) to %s\n",
            if (has_region) paste0("region '", argv$region, "'")
            else paste0("gene '", argv$gene_id, "'"),
            nrow(res), argv$output))
