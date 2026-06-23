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
#   --methods               Comma-separated method tokens (default
#                           "default"; "default" expands inside pecotmr).
#                           Pass e.g. "mrmash,mvsusie" for multivariate
#                           TWAS weights.
#   --fine-mapping-result   Optional pre-fit FineMappingResult RDS;
#                           SuSiE-family methods reuse the cached fits
#                           via the fineMappingResult cache. Pass "" or
#                           "." to skip.
#   --method-args           Optional JSON object of per-method kwargs
#                           spliced into twasWeightsPipeline() via its
#                           named-list methods= argument. Keys are
#                           method tokens (must be a subset of --methods);
#                           values are kwargs lists forwarded to the
#                           underlying per-method learner. Example:
#                           '{"lasso":{"nfolds":10}}'.
#   --output                Output RDS path (one TwasWeights)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
  library(jsonlite)
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
parser <- add_argument(parser, "--methods",
                       help = "Comma-separated TWAS method tokens (default 'default')",
                       type = "character", default = "default")
parser <- add_argument(parser, "--fine-mapping-result",
                       help = "Optional pre-fit FineMappingResult RDS",
                       type = "character", default = "")
parser <- add_argument(parser, "--method-args",
                       help = "JSON object {token: {kwarg: value, ...}, ...} for twasWeightsPipeline()",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Parse --method-args into a nested named list of per-method kwargs.
parsed_method_args <- if (nzchar(argv$method_args) && argv$method_args != "." &&
                          argv$method_args != "{}") {
  parsed <- tryCatch(jsonlite::fromJSON(argv$method_args, simplifyVector = FALSE),
                     error = function(e) stop(
                       "--method-args must be a JSON object string (got: ",
                       argv$method_args, "). Error: ", conditionMessage(e)))
  if (!is.list(parsed) || is.null(names(parsed)) || any(names(parsed) == ""))
    stop("--method-args must be a JSON object whose keys are method ",
         "tokens, e.g. '{\"lasso\":{\"nfolds\":10}}'.")
  nonObject <- vapply(parsed, function(x) !is.list(x), logical(1))
  if (any(nonObject))
    stop("--method-args: each value must itself be an object of kwargs ",
         "(got non-object for: ",
         paste(names(parsed)[nonObject], collapse = ", "), ").")
  parsed
} else {
  NULL
}

# Normalize --methods into the form twasWeightsPipeline expects: a
# character vector of method tokens, or "default" (the preset). When
# --method-args is supplied, build the named-list form {token: kwargs}
# that twasWeightsPipeline accepts directly. The "default" preset can't
# carry per-method kwargs (it expands inside pecotmr to a fixed token
# set we can't see here) — explicit --methods is required in that case.
methods <- trimws(strsplit(argv$methods, ",", fixed = TRUE)[[1L]])
methods_arg <- if (is.null(parsed_method_args)) {
  if (length(methods) == 1L && methods == "default") "default" else methods
} else {
  if (length(methods) == 1L && methods == "default")
    stop("--method-args is only supported with explicit --methods (got ",
         "--methods 'default'); list the method tokens you want to tune ",
         "explicitly, e.g. --methods lasso,enet --method-args '{\"lasso\":{\"nfolds\":10}}'.")
  unknown <- setdiff(names(parsed_method_args), methods)
  if (length(unknown) > 0L)
    stop("--method-args has keys not listed in --methods (got '",
         paste(unknown, collapse = ", "),
         "'; --methods = '", paste(methods, collapse = ", "), "').")
  setNames(lapply(methods, function(tk) {
    if (tk %in% names(parsed_method_args)) parsed_method_args[[tk]]
    else list()
  }), methods)
}

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
  twasWeightsPipeline(qd, methods = methods_arg,
                      region    = parse_region(argv$region),
                      cisWindow = argv$cis_window,
                      fineMappingResult = fmr)
} else {
  twasWeightsPipeline(qd, methods = methods_arg,
                      traitId   = argv$gene_id,
                      cisWindow = argv$cis_window,
                      fineMappingResult = fmr)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote TWAS weights for %s (%d row(s)) to %s\n",
            if (has_region) paste0("region '", argv$region, "'")
            else paste0("gene '", argv$gene_id, "'"),
            nrow(res), argv$output))
