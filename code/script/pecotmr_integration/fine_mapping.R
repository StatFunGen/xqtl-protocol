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
#   --method-args   Optional JSON object of per-method kwargs spliced
#                   into fineMappingPipeline() via its named-list
#                   methods= argument. Keys are method tokens (must be
#                   a subset of --methods); values are kwargs lists
#                   forwarded to the underlying fitter (susieR::susie,
#                   susieR::susie_rss, mvsusieR::mvsusie, fsusieR::susiF,
#                   etc.). Example: '{"susie":{"L":1,"refine":false}}'.
#   --output        Output RDS path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
  library(jsonlite)
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
                       help = "SuSiE primary credible-set coverage",
                       type = "numeric", default = 0.95)
parser <- add_argument(parser, "--secondary-coverage",
                       help = "Comma-separated secondary credible-set coverages (fineMappingPipeline secondaryCoverage)",
                       type = "character", default = "0.7,0.5")
parser <- add_argument(parser, "--min-abs-corr",
                       help = "Credible-set purity threshold (fineMappingPipeline minAbsCorr)",
                       type = "numeric", default = 0.8)
parser <- add_argument(parser, "--median-abs-corr",
                       help = "Optional median-abs-corr purity, OR-logic with --min-abs-corr (fineMappingPipeline medianAbsCorr; requires pecotmr step-1). Omit to leave off.",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--pip-cutoff",
                       help = "PIP signal cutoff (fineMappingPipeline signalCutoff)",
                       type = "numeric", default = 0.025)
parser <- add_argument(parser, "--L",
                       help = "SuSiE number of single effects (susie L); pipeline default 20",
                       type = "integer", default = 20L)
parser <- add_argument(parser, "--L-greedy",
                       help = "SuSiE greedy init count (susie L_greedy); pipeline default 5",
                       type = "integer", default = 5L)
parser <- add_argument(parser, "--method-args",
                       help = "JSON object {token: {kwarg: value, ...}, ...} for fineMappingPipeline()",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Parse --method-args into a nested named list of per-method kwargs.
# Empty / '.' / '{}' all yield NULL (which makes us pass the char-vector
# form of methods= to fineMappingPipeline).
parsed_method_args <- if (nzchar(argv$method_args) && argv$method_args != "." &&
                          argv$method_args != "{}") {
  parsed <- tryCatch(jsonlite::fromJSON(argv$method_args, simplifyVector = FALSE),
                     error = function(e) stop(
                       "--method-args must be a JSON object string (got: ",
                       argv$method_args, "). Error: ", conditionMessage(e)))
  if (!is.list(parsed) || is.null(names(parsed)) || any(names(parsed) == ""))
    stop("--method-args must be a JSON object whose keys are method ",
         "tokens, e.g. '{\"susie\":{\"L\":1}}'.")
  nonObject <- vapply(parsed, function(x) !is.list(x), logical(1))
  if (any(nonObject))
    stop("--method-args: each value must itself be an object of kwargs ",
         "(got non-object for: ",
         paste(names(parsed)[nonObject], collapse = ", "),
         "). Did you mean '{\"susie\":", argv$method_args, "}'?")
  parsed
} else {
  NULL
}

# Secondary coverages: comma-separated -> numeric vector. medianAbsCorr: NA
# (unset) -> NULL so we omit it from the call (off, and back-compatible with a
# pecotmr that predates the medianAbsCorr argument).
secondary_cov <- as.numeric(trimws(strsplit(argv$secondary_coverage, ",", fixed = TRUE)[[1L]]))
median_abs_corr <- if (length(argv$median_abs_corr) != 1L || is.na(argv$median_abs_corr))
                     NULL else argv$median_abs_corr

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

# Build the final `methods` argument for fineMappingPipeline as the named-list
# form {token: kwargs}. The pipeline's SuSiE fit defaults (L = --L,
# L_greedy = --L-greedy) seed every SuSiE-family token; any matching
# --method-args entry overrides them per key (explicit > default). Keys in
# --method-args must be among the --methods tokens (no silent typos).
if (!is.null(parsed_method_args)) {
  unknown <- setdiff(names(parsed_method_args), methods)
  if (length(unknown) > 0L)
    stop("--method-args has keys not listed in --methods (got '",
         paste(unknown, collapse = ", "),
         "'; --methods = '", paste(methods, collapse = ", "), "').")
}
.susie_family <- c("susie", "susieInf", "susieAsh")
.fit_defaults <- list(L = argv$L, L_greedy = argv[["L_greedy"]])
methods_arg <- setNames(lapply(methods, function(tk) {
  base <- if (tk %in% .susie_family) .fit_defaults else list()
  user <- if (!is.null(parsed_method_args) && tk %in% names(parsed_method_args))
            parsed_method_args[[tk]] else list()
  modifyList(base, user)
}), methods)

# Credible-set / coverage knobs common to both modes. medianAbsCorr is added
# only when set (NULL -> omitted), so the call also works against a pecotmr
# that predates that argument.
cs_args <- list(methods           = methods_arg,
                coverage          = argv$coverage,
                secondaryCoverage = secondary_cov,
                signalCutoff      = argv$pip_cutoff,
                minAbsCorr        = argv$min_abs_corr)
if (!is.null(median_abs_corr)) cs_args$medianAbsCorr <- median_abs_corr

if (has_gwas) {
  # ----- GWAS mode -------------------------------------------------------
  gss <- readRDS(argv$gwas_sumstats)
  res <- do.call(fineMappingPipeline, c(list(gss), cs_args))
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
  qtl_args <- c(list(qd), cs_args, list(cisWindow = argv$cis_window))
  res <- if (has_region) {
    do.call(fineMappingPipeline, c(qtl_args, list(region = parse_region(argv$region))))
  } else {
    do.call(fineMappingPipeline, c(qtl_args, list(traitId = argv$gene_id)))
  }
  label <- if (has_region) paste0("region '", argv$region, "'")
           else paste0("gene '", argv$gene_id, "'")
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote fineMapping result for %s (%d row(s)) to %s\n",
            label, nrow(res), argv$output))
