#!/usr/bin/env Rscript
# fine_mapping.R
#
# SuSiE fine-mapping worker for either QTL (per-gene / per-region over a
# pre-built QtlDataset) or GWAS (per-block over a GwasSumStats RDS).
# Designed to be invoked once per fan-out unit by the SoS step in
# fine_mapping.ipynb. `pecotmr::fineMappingPipeline` dispatches on the
# input class.
#
# Input modes (exactly one of --qtl-dataset / --gwas-sumstats):
#
# QTL — fan-out per gene or per region inside a single QtlDataset:
#   --qtl-dataset <RDS>             pecotmr::QtlDataset (from qtl_dataset.ipynb)
#   --gene-id ENSG... --cis-window 1000000   (gene mode)
#   --region chr22:15000000-16000000          (region mode)
#
# GWAS — one call per per-block GwasSumStats RDS (each carrying its own
# z-scores + LD sketch; no gene/region concept):
#   --gwas-sumstats <RDS>           pecotmr::GwasSumStats (per LD block,
#                                   typically from gwas_sumstats_construct.R)
#
# Shared:
#   --methods       Comma-separated method tokens. Default "susie".
#   --coverage      SuSiE credible-set coverage. Default 0.95.
#   --output        Output RDS path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
})

parser <- arg_parser("SuSiE fine-mapping over a pecotmr S4 input (QtlDataset or GwasSumStats)")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS (QTL mode)",
                       type = "character", default = "")
parser <- add_argument(parser, "--gwas-sumstats",
                       help = "Path to a GwasSumStats RDS (GWAS mode)",
                       type = "character", default = "")
parser <- add_argument(parser, "--gene-id",
                       help = "Trait identifier (QTL gene mode); mutually exclusive with --region",
                       type = "character", default = "")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the trait's TSS (QTL gene mode)",
                       type = "integer", default = 1000000L)
parser <- add_argument(parser, "--region",
                       help = "Genomic region as chr:start-end (QTL region mode)",
                       type = "character", default = "")
parser <- add_argument(parser, "--methods",
                       help = "Comma-separated fine-mapping method tokens",
                       type = "character", default = "susie")
parser <- add_argument(parser, "--coverage",
                       help = "SuSiE credible-set coverage",
                       type = "numeric", default = 0.95)
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

has_qtl  <- nzchar(argv$qtl_dataset)
has_gwas <- nzchar(argv$gwas_sumstats)
if (has_qtl && has_gwas)
  stop("--qtl-dataset and --gwas-sumstats are mutually exclusive; pass exactly one.")
if (!has_qtl && !has_gwas)
  stop("Specify either --qtl-dataset (QTL mode) or --gwas-sumstats (GWAS mode).")

methods <- trimws(strsplit(argv$methods, ",", fixed = TRUE)[[1L]])

if (has_gwas) {
  # ----- GWAS mode -------------------------------------------------------
  gss <- readRDS(argv$gwas_sumstats)
  res <- fineMappingPipeline(
    gss,
    methods  = methods,
    coverage = argv$coverage)
  label <- paste0("GwasSumStats '", basename(argv$gwas_sumstats), "'")
} else {
  # ----- QTL mode --------------------------------------------------------
  has_gene   <- nzchar(argv$gene_id)
  has_region <- nzchar(argv$region)
  if (has_gene && has_region)
    stop("--gene-id and --region are mutually exclusive (QTL mode); pass exactly one.")
  if (!has_gene && !has_region)
    stop("QTL mode requires --gene-id (with --cis-window) or --region.")
  qd <- readRDS(argv$qtl_dataset)
  res <- if (has_region) {
    fineMappingPipeline(
      qd,
      methods   = methods,
      region    = parse_region(argv$region),
      cisWindow = argv$cis_window,
      coverage  = argv$coverage)
  } else {
    fineMappingPipeline(
      qd,
      methods    = methods,
      traitId    = argv$gene_id,
      cisWindow  = argv$cis_window,
      coverage   = argv$coverage)
  }
  label <- if (has_region) paste0("region '", argv$region, "'")
           else paste0("gene '", argv$gene_id, "'")
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote fineMapping result for %s (%d row(s)) to %s\n",
            label, nrow(res), argv$output))
