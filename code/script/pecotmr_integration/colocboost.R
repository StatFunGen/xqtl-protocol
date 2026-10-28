#!/usr/bin/env Rscript
# colocboost.R
#
# Per-region colocboost worker. Loads a pre-built pecotmr::QtlDataset RDS
# (and optionally a QC'd GwasSumStats RDS) and calls colocboostPipeline()
# for either a single gene (gene mode) or a single genomic region
# (region mode). Designed to be invoked once per fan-out unit by the SoS
# step in colocboost.ipynb.
#
# Modes (mutually exclusive; they choose the window, not the outcomes):
#   gene   : --gene-id ENSG... --cis-window 1000000
#            traitId = gene-id: the gene in every context that holds it.
#   region : --region chr22:15000000-16000000
#            Pipeline receives region = GRanges(...).
#
# Pipeline variants (any combination; at least one must be active). Focality
# follows colocboost's own pipeline in either mode:
#   --xqtl-coloc      ON by default; within-study cross-context QTL coloc,
#                     non-focal. Pass --no-xqtl-coloc to disable.
#   --joint-gwas      Off by default; one non-focal xQTL+GWAS coloc over every
#                     GWAS study. Requires --gwas-sumstats.
#   --separate-gwas   Off by default; one coloc per GWAS study, focal on that
#                     GWAS. Requires --gwas-sumstats.
#
# Inputs:
#   --qtl-dataset    Path to a QtlDataset RDS
#   --gene-id        (gene mode) trait identifier
#   --cis-window     cis-window in bp around the trait's TSS
#   --region         (region mode) chr:start-end string
#   --gwas-sumstats  Optional path to a QC'd GwasSumStats RDS
#   --no-xqtl-coloc  Flag: disable xQTL-only cross-context coloc
#   --joint-gwas     Flag: run one non-focal xQTL+GWAS coloc
#   --separate-gwas  Flag: run one GWAS-focal coloc per GWAS study
#   --pip-cutoff-to-skip
#                    Optional per-context single-effect skip cutoff passed to
#                    colocboostPipeline's SER pre-screen. Either a single
#                    number applied to every context, or comma-separated
#                    context=value pairs (e.g. "context1=0.1,context2=0").
#                    A negative value selects the data-driven 3/ncol(X)
#                    default. Empty (default) disables the skip (0).
#   --output         Output RDS path (one ColocBoostResult collection)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "pecotmr_aliases.R"))

parser <- arg_parser("Per-gene or per-region colocboost over a pre-built QtlDataset")
parser <- add_argument(parser, "--qtl-dataset",
                       help = "Path to a QtlDataset RDS",
                       type = "character")
parser <- add_argument(parser, "--gene-id",
                       help = "Trait (gene) identifier (gene mode); mutually exclusive with --region",
                       type = "character", default = "")
parser <- add_argument(parser, "--cis-window",
                       help = "cis-window in bp around the trait's TSS",
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
                       help = "Run one non-focal xQTL+GWAS colocboost over every GWAS study (requires --gwas-sumstats)",
                       flag = TRUE)
parser <- add_argument(parser, "--separate-gwas",
                       help = "Run one colocboost per GWAS study, focal on that GWAS (requires --gwas-sumstats)",
                       flag = TRUE)
parser <- add_argument(parser, "--pip-cutoff-to-skip",
                       help = "Per-context single-effect skip cutoff: a scalar or comma-separated context=value pairs (negative = data-driven default; empty = 0)",
                       type = "character", default = "")
# --- Analysis-time genotype filters and residualization. Each is opt-in: an
# unset flag leaves the pipeline's own default in place. A QtlDataset stores
# neither, so these belong here rather than on qtl_dataset_construct.R.
# Mirrors fine_mapping.R / twas_weights.R.
parser <- add_argument(parser, "--maf-cutoff",
                       help = "Override mafCutoff for this analysis (GenotypeFilterParam)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--mac-cutoff",
                       help = "Override macCutoff (GenotypeFilterParam)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--xvar-cutoff",
                       help = "Override xvarCutoff (GenotypeFilterParam)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--imiss-cutoff",
                       help = "Override imissCutoff (GenotypeFilterParam)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--drop-indel",
                       help = "Drop indels (keepIndel = FALSE); omit to keep the pipeline default",
                       flag = TRUE)
parser <- add_argument(parser, "--keep-samples",
                       help = "Path to a whitespace-delimited file of sample IDs to restrict to",
                       type = "character", default = "")
parser <- add_argument(parser, "--keep-variants",
                       help = "Path to a whitespace-delimited file of variant IDs to restrict to",
                       type = "character", default = "")
parser <- add_argument(parser, "--no-scale-residuals",
                       help = "Fit on unscaled residuals (ResidualizationParam scaleResiduals = FALSE)",
                       flag = TRUE)
parser <- add_argument(parser, "--seed",
                       help = "Integer RNG seed set before the colocboost fit (reproducibility); unset = no seeding",
                       type = "integer", default = NA)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Parse --pip-cutoff-to-skip into the scalar / context-named-vector shape
# the SER pre-screen accepts. Mirrors susie_twas.R. It travels as
# SignalScreenParam(pip = ), where 0 is off -- this flag's documented 0 = off
# -- and a negative value selects the adaptive 3/nVariants threshold.
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

# Read a whitespace-delimited ID file into a unique character vector; NULL when
# the path is empty / "." / missing.
read_ids <- function(p) if (nzchar(p) && p != "." && file.exists(p))
  unique(trimws(unlist(strsplit(readLines(p), "[[:space:]]+")))) else NULL

# Collect the opt-in overrides into the two bundles colocboostPipeline takes.
# An unset field is dropped rather than passed, which is what leaves the
# pipeline default in place.
analysis_params <- function(argv) {
  ov <- list(
    mafCutoff    = if (!is.na(argv$maf_cutoff))   argv$maf_cutoff   else NULL,
    macCutoff    = if (!is.na(argv$mac_cutoff))   argv$mac_cutoff   else NULL,
    xvarCutoff   = if (!is.na(argv$xvar_cutoff))  argv$xvar_cutoff  else NULL,
    imissCutoff  = if (!is.na(argv$imiss_cutoff)) argv$imiss_cutoff else NULL,
    keepIndel    = if (isTRUE(argv$drop_indel))   FALSE             else NULL,
    keepSamples  = read_ids(argv$keep_samples),
    keepVariants = read_ids(argv$keep_variants))
  ov <- ov[!vapply(ov, is.null, logical(1))]
  out <- list(genotypeFilterParam = do.call(GenotypeFilterParam, ov))
  if (isTRUE(argv$no_scale_residuals))
    out$residualizationParam <- ResidualizationParam(scaleResiduals = FALSE)
  out
}
extra <- analysis_params(argv)

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

# Seed up front for reproducible colocboost fits (colocboost is stochastic:
# gradient boosting). Mirrors fine_mapping.R.
if (length(argv$seed) == 1L && !is.na(argv$seed)) set.seed(as.integer(argv$seed))

res <- if (has_region) {
  do.call(colocboostPipeline, c(list(
    qd,
    gwasSumStats    = gss,
    region          = as_region_granges(argv$region),
    cisWindow       = argv$cis_window,
    xqtlColoc       = xqtl_coloc,
    jointGwas       = joint_gwas,
    separateGwas    = separate_gwas,
    signalScreenParam = SignalScreenParam(pip = pip_cutoff_to_skip)), extra))
} else {
  do.call(colocboostPipeline, c(list(
    qd,
    gwasSumStats    = gss,
    traitId         = argv$gene_id,
    cisWindow       = argv$cis_window,
    xqtlColoc       = xqtl_coloc,
    jointGwas       = joint_gwas,
    separateGwas    = separate_gwas,
    signalScreenParam = SignalScreenParam(pip = pip_cutoff_to_skip)), extra))
}

# Production output is emitted verbatim — including colocboostPipeline()'s wall-clock
# res$computing_time (difftime leaves). Those timings vary run-to-run but are metadata,
# not analysis output, so the regression comparator strips time-like leaves rather than
# the worker mutating its output (see tests/helpers/rds_compare.R). --seed above governs
# the actual stochastic fit, which is what makes the result reproducible.
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
