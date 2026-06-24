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
parser <- add_argument(parser, "--contexts",
                       help = "Comma-separated context names to restrict to; empty = all contexts",
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
parser <- add_argument(parser, "--min-twas-maf",
                       help = "Minimum MAF for the variants used to learn TWAS weights (twasWeightsPipeline minTwasMaf), applied on top of the dataset's construct-time mafCutoff",
                       type = "numeric", default = 0.01)
parser <- add_argument(parser, "--min-twas-xvar",
                       help = "Minimum per-variant genotype variance for TWAS weight learning (twasWeightsPipeline minTwasXvar)",
                       type = "numeric", default = 0.01)
parser <- add_argument(parser, "--max-cv-variants",
                       help = "Cap on the number of variants used in cross-validation (twasWeightsPipeline maxCvVariants); -1 = no cap",
                       type = "integer", default = 5000L)
parser <- add_argument(parser, "--cv-folds",
                       help = "Cross-validation folds for TWAS weight evaluation (twasWeightsPipeline cvFolds)",
                       type = "integer", default = 5L)
parser <- add_argument(parser, "--cv-threads",
                       help = "Threads for the cross-validation refits (twasWeightsPipeline cvThreads)",
                       type = "integer", default = 1L)
parser <- add_argument(parser, "--seed",
                       help = "Integer RNG seed set before fitting (reproducibility); unset = no seeding",
                       type = "integer", default = NA)
# --- Per-analysis overrides of the QtlDataset's construct-time filters. Each is
# opt-in (unset leaves the dataset's stored value); applied to the loaded object
# before the pipeline runs.
parser <- add_argument(parser, "--maf-cutoff",
                       help = "Override QtlDataset mafCutoff for this analysis",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--mac-cutoff",
                       help = "Override QtlDataset macCutoff",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--xvar-cutoff",
                       help = "Override QtlDataset xvarCutoff",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--imiss-cutoff",
                       help = "Override QtlDataset imissCutoff",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--drop-indel",
                       help = "Drop indels (sets keepIndel = FALSE); omit to use the dataset's stored value",
                       flag = TRUE)
parser <- add_argument(parser, "--keep-samples",
                       help = "Path to a whitespace-delimited file of sample IDs to restrict to (overrides keepSamples)",
                       type = "character", default = "")
parser <- add_argument(parser, "--keep-variants",
                       help = "Path to a whitespace-delimited file of variant IDs to restrict to (overrides keepVariants)",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Opt-in overrides of a loaded QtlDataset's construct-time filter slots. Filters
# apply lazily at extraction, so mutating the slot re-filters without rebuilding
# the RDS. Mirrors fine_mapping.R.
apply_qd_filter_overrides <- function(qd, argv) {
  if (!is.na(argv$maf_cutoff))   qd@mafCutoff   <- argv$maf_cutoff
  if (!is.na(argv$mac_cutoff))   qd@macCutoff   <- argv$mac_cutoff
  if (!is.na(argv$xvar_cutoff))  qd@xvarCutoff  <- argv$xvar_cutoff
  if (!is.na(argv$imiss_cutoff)) qd@imissCutoff <- argv$imiss_cutoff
  if (isTRUE(argv$drop_indel))   qd@keepIndel   <- FALSE
  read_ids <- function(p) if (nzchar(p) && p != "." && file.exists(p))
    unique(trimws(unlist(strsplit(readLines(p), "[[:space:]]+")))) else NULL
  ks <- read_ids(argv$keep_samples);  if (!is.null(ks)) qd@keepSamples  <- ks
  kv <- read_ids(argv$keep_variants); if (!is.null(kv)) qd@keepVariants <- kv
  qd
}

# Seed up front for reproducible fits (mirrors the legacy susie_twas set.seed).
if (length(argv$seed) == 1L && !is.na(argv$seed)) set.seed(as.integer(argv$seed))

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
# Optional context restriction: NULL = all contexts in the dataset.
contexts_arg <- if (nzchar(argv$contexts) && argv$contexts != ".")
  trimws(strsplit(argv$contexts, ",", fixed = TRUE)[[1L]]) else NULL

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
qd <- apply_qd_filter_overrides(qd, argv)

fmr_path <- argv$fine_mapping_result
fmr <- if (nzchar(fmr_path) && fmr_path != "." && file.exists(fmr_path)) {
  readRDS(fmr_path)
} else {
  NULL
}

# Shared args for both modes. minTwas* tighten the variant set used to learn
# the weights (on top of the QtlDataset's construct-time cutoffs); the cv* knobs
# control the cross-validated predictive-performance refits.
tw_args <- list(methods           = methods_arg,
                cisWindow         = argv$cis_window,
                contexts          = contexts_arg,
                minTwasMaf        = argv$min_twas_maf,
                minTwasXvar       = argv$min_twas_xvar,
                maxCvVariants     = argv$max_cv_variants,
                cvFolds           = argv$cv_folds,
                cvThreads         = argv$cv_threads,
                fineMappingResult = fmr)
res <- if (has_region) {
  do.call(twasWeightsPipeline,
          c(list(qd), tw_args, list(region = parse_region(argv$region))))
} else {
  do.call(twasWeightsPipeline,
          c(list(qd), tw_args, list(traitId = argv$gene_id)))
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote TWAS weights for %s (%d row(s)) to %s\n",
            if (has_region) paste0("region '", argv$region, "'")
            else paste0("gene '", argv$gene_id, "'"),
            nrow(res), argv$output))
