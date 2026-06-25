#!/usr/bin/env Rscript
# colocboost.R
#
# Per-region colocboost worker. Loads a pre-built pecotmr::QtlDataset RDS
# (and optionally a QC'd GwasSumStats RDS) and calls colocboostPipeline()
# for either a single focal trait (gene mode) or a single genomic region
# (region mode). Designed to be invoked once per fan-out unit by the SoS
# step in colocboost.ipynb.
#
# Modes (mutually exclusive):
#   gene   : --gene-id ENSG... --cis-window 1000000
#            traitId = gene-id, focalTrait = gene-id.
#   region : --region chr22:15000000-16000000
#            Pipeline receives region = GRanges(...). No focal trait.
#
# Pipeline variants (any combination; at least one must be active):
#   --xqtl-coloc      ON by default; within-study cross-context QTL coloc.
#                     Pass --no-xqtl-coloc to disable.
#   --joint-gwas      Off by default; xQTL+GWAS joint coloc. Requires
#                     --gwas-sumstats.
#   --separate-gwas   Off by default; one focal-GWAS coloc per GWAS study.
#                     Requires --gwas-sumstats.
#
# Inputs:
#   --qtl-dataset    Path to a QtlDataset RDS
#   --gene-id        (gene mode) focal trait identifier
#   --cis-window     cis-window in bp around the trait's TSS
#   --region         (region mode) chr:start-end string
#   --gwas-sumstats  Optional path to a QC'd GwasSumStats RDS
#   --no-xqtl-coloc  Flag: disable xQTL-only cross-context coloc
#   --joint-gwas     Flag: run xQTL+GWAS joint coloc
#   --separate-gwas  Flag: run per-GWAS-study separate-focal coloc
#   --pip-cutoff-to-skip
#                    Optional per-context single-effect skip cutoff passed to
#                    colocboostPipeline(pipCutoffToSkip = ). Either a single
#                    number applied to every context, or comma-separated
#                    context=value pairs (e.g. "context1=0.1,context2=0").
#                    A negative value selects the data-driven 3/ncol(X)
#                    default. Empty (default) disables the skip (0).
#   --output         Output RDS path (one colocboost-pipeline list)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
})

parser <- arg_parser("Per-gene or per-region colocboost over a pre-built QtlDataset")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS",
                       type = "character")
parser <- add_argument(parser, "--gene-id",
                       help = "Focal trait identifier (gene mode); mutually exclusive with --region",
                       type = "character", default = "")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the focal trait's TSS",
                       type = "integer", default = 1000000L)
parser <- add_argument(parser, "--region",
                       help = "Genomic region as chr:start-end (region mode); mutually exclusive with --gene-id",
                       type = "character", default = "")
parser <- add_argument(parser, "--gwas-sumstats",
                       help = "Path to a QC'd GwasSumStats RDS (required when --joint-gwas or --separate-gwas is set)",
                       type = "character", default = "")
parser <- add_argument(parser, "--no-xqtl-coloc",
                       help = "Disable the within-study cross-context xQTL coloc",
                       flag = TRUE)
parser <- add_argument(parser, "--joint-gwas",
                       help = "Run xQTL+GWAS joint colocboost (requires --gwas-sumstats)",
                       flag = TRUE)
parser <- add_argument(parser, "--separate-gwas",
                       help = "Run per-GWAS-study separate-focal coloc (requires --gwas-sumstats)",
                       flag = TRUE)
parser <- add_argument(parser, "--pip-cutoff-to-skip",
                       help = "Per-context single-effect skip cutoff: a scalar or comma-separated context=value pairs (negative = data-driven default; empty = 0)",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Parse --pip-cutoff-to-skip into the scalar / context-named-vector shape
# colocboostPipeline(pipCutoffToSkip = ) accepts. Mirrors susie_twas.R.
parse_pip_cutoff <- function(s) {
  if (is.null(s) || !nzchar(s)) return(0)
  pairs <- strsplit(s, ",")[[1L]]
  if (any(grepl("=", pairs, fixed = TRUE))) {
    setNames(as.numeric(sub(".*=", "", pairs)),
             gsub("'", "", sub("=.*", "", pairs)))
  } else {
    as.numeric(pairs[[1L]])
  }
}
pip_cutoff_to_skip <- parse_pip_cutoff(argv$pip_cutoff_to_skip)

parse_region <- function(s) {
  m <- regmatches(s, regexec("^([^:]+):([0-9]+)-([0-9]+)$", s))[[1L]]
  if (length(m) != 4L)
    stop("--region must be in chr:start-end format (got: ", s, ")")
  GRanges(seqnames = m[[2L]],
          ranges   = IRanges(start = as.integer(m[[3L]]),
                             end   = as.integer(m[[4L]])))
}

# Mode validation
has_gene   <- nzchar(argv$gene_id)
has_region <- nzchar(argv$region)
if (has_gene && has_region)
  stop("--gene-id and --region are mutually exclusive; pass exactly one.")
if (!has_gene && !has_region)
  stop("Specify either --gene-id (with --cis-window) or --region.")

xqtl_coloc    <- !argv$no_xqtl_coloc
joint_gwas    <- argv$joint_gwas
separate_gwas <- argv$separate_gwas

if (!xqtl_coloc && !joint_gwas && !separate_gwas)
  stop("All colocboost variants disabled. Enable at least one of xQTL coloc, --joint-gwas, --separate-gwas.")

gwas_path <- argv$gwas_sumstats
has_gwas  <- nzchar(gwas_path) && gwas_path != "." && file.exists(gwas_path)
if ((joint_gwas || separate_gwas) && !has_gwas)
  stop("--joint-gwas / --separate-gwas require --gwas-sumstats pointing at a QC'd GwasSumStats RDS.")

qd  <- readRDS(argv$qtl_dataset)
gss <- if (has_gwas) readRDS(gwas_path) else NULL

res <- if (has_region) {
  colocboostPipeline(
    qd,
    gwasSumStats    = gss,
    region          = parse_region(argv$region),
    cisWindow       = argv$cis_window,
    xqtlColoc       = xqtl_coloc,
    jointGwas       = joint_gwas,
    separateGwas    = separate_gwas,
    pipCutoffToSkip = pip_cutoff_to_skip)
} else {
  colocboostPipeline(
    qd,
    gwasSumStats    = gss,
    traitId         = argv$gene_id,
    cisWindow       = argv$cis_window,
    focalTrait      = argv$gene_id,
    xqtlColoc       = xqtl_coloc,
    jointGwas       = joint_gwas,
    separateGwas    = separate_gwas,
    pipCutoffToSkip = pip_cutoff_to_skip)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)

# Brief per-variant summary
variants <- c(
  if (xqtl_coloc) "xqtl-coloc" else NULL,
  if (joint_gwas) "joint-gwas" else NULL,
  if (separate_gwas) "separate-gwas" else NULL)
cat(sprintf("Wrote colocboost result for %s [variants: %s] to %s\n",
            if (has_region) paste0("region '", argv$region, "'")
            else paste0("gene '", argv$gene_id, "'"),
            paste(variants, collapse = ", "),
            argv$output))
