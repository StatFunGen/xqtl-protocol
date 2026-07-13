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
#   --mixture-prior         Optional RDS of mr.mash data-driven prior
#                           covariance matrices (dataDrivenPriorMatrices)
#                           for the 'mrmash' method. Omitted -> canonical
#                           prior (canonicalPriorMatrices = TRUE). mr.mash
#                           requires one of these; this wrapper supplies a
#                           default so 'mrmash' runs out of the box, and is
#                           also the producer of the mvSuSiE data-driven
#                           prior consumed by fine_mapping.R --twas-weights.
#   --output                Output RDS path (one TwasWeights)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
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
parser <- add_argument(parser, "--joint-specification",
                       help = "Comma-separated joint-analysis axis spec (twasWeightsPipeline jointSpecification), e.g. 'context' for cross-context joint weight learning; empty = per-(context, trait)",
                       type = "character", default = "")
parser <- add_argument(parser, "--twas-weights",
                       help = "Optional existing TwasWeights RDS to resume from (twasWeightsPipeline twasWeights); already-computed (context, trait, method) weights are reused for checkpointing/resumption",
                       type = "character", default = "")
parser <- add_argument(parser, "--method-args",
                       help = "JSON object {token: {kwarg: value, ...}, ...} for twasWeightsPipeline()",
                       type = "character", default = "")
parser <- add_argument(parser, "--mixture-prior",
                       help = paste0("Path to an RDS of mr.mash data-driven prior covariance matrices ",
                                     "(dataDrivenPriorMatrices) for the 'mrmash' method. When omitted, ",
                                     "mr.mash uses the canonical prior set (canonicalPriorMatrices = TRUE). ",
                                     "Only consulted when 'mrmash' is in --methods and the prior is not ",
                                     "already set via --method-args."),
                       type = "character", default = "")
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

# Read a whitespace-delimited ID file into a unique character vector; NULL when
# the path is empty / "." / missing.
read_ids <- function(p) if (nzchar(p) && p != "." && file.exists(p))
  unique(trimws(unlist(strsplit(readLines(p), "[[:space:]]+")))) else NULL

# Build the per-call genotype-filter overrides as pipeline ARGUMENTS (NULL =
# leave the QtlDataset's construct-time slot untouched). The pipeline applies
# these to a validated copy; the wrapper no longer mutates @slots directly.
# Mirrors fine_mapping.R.
qd_filter_overrides <- function(argv) {
  ov <- list(
    mafCutoff    = if (!is.na(argv$maf_cutoff))   argv$maf_cutoff   else NULL,
    macCutoff    = if (!is.na(argv$mac_cutoff))   argv$mac_cutoff   else NULL,
    xvarCutoff   = if (!is.na(argv$xvar_cutoff))  argv$xvar_cutoff  else NULL,
    imissCutoff  = if (!is.na(argv$imiss_cutoff)) argv$imiss_cutoff else NULL,
    keepIndel    = if (isTRUE(argv$drop_indel))   FALSE             else NULL,
    keepSamples  = read_ids(argv$keep_samples),
    keepVariants = read_ids(argv$keep_variants))
  ov[!vapply(ov, is.null, logical(1))]
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

# mr.mash data-driven prior override. The wrapper stays thin: it hands the prior
# covariance set to pecotmr as a MashPrior object; twasWeightsPipeline unpacks it
# (.unpackMashPrior) and injects it into the mr.mash fit. pecotmr's mrmash method
# default supplies the canonical set, so the common case needs no prior here.
mash_prior <- if (nzchar(argv$mixture_prior) && argv$mixture_prior != "." &&
                  file.exists(argv$mixture_prior)) {
  MashPrior(fullFit = readRDS(argv$mixture_prior))
} else NULL

# Joint-analysis axis (mirrors fine_mapping.R) and an existing TwasWeights to
# resume from (checkpointing): already-computed (context, trait, method) weights
# are reused so a re-run only fills in the missing methods.
joint_spec <- if (nzchar(argv$joint_specification) && argv$joint_specification != ".")
  trimws(strsplit(argv$joint_specification, ",", fixed = TRUE)[[1L]]) else NULL
twas_weights_obj <- if (nzchar(argv$twas_weights) && argv$twas_weights != "." &&
                        file.exists(argv$twas_weights)) readRDS(argv$twas_weights) else NULL


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

# Shared args for both modes. Genotype-filter overrides ride on the same
# --maf-cutoff/--xvar-cutoff/... args fine-mapping uses (variant QC is a data
# property, applied identically); the cv* knobs control the cross-validated
# predictive-performance refits.
tw_args <- c(list(methods           = methods_arg,
                  mashPrior         = mash_prior,
                  cisWindow         = argv$cis_window,
                  contexts          = contexts_arg,
                  maxCvVariants     = argv$max_cv_variants,
                  cvFolds           = argv$cv_folds,
                  cvThreads         = argv$cv_threads,
                  fineMappingResult = fmr),
             qd_filter_overrides(argv))
# Opt-in joint axis / resume-from-checkpoint, added only when supplied (keeps
# the call compatible with a pecotmr that predates the argument).
if (!is.null(joint_spec))       tw_args$jointSpecification <- joint_spec
if (!is.null(twas_weights_obj)) tw_args$twasWeights        <- twas_weights_obj
res <- if (has_region) {
  do.call(twasWeightsPipeline,
          c(list(qd), tw_args, list(region = argv$region)))
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
