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
#                   For mvsusie / fsusie this is how you pass their kwargs,
#                   e.g. '{"mvsusie":{"max_iter":200}}' or
#                   '{"fsusie":{"prior":"mixture_normal","max_scale":10}}'.
#
# Multivariate / joint fits (QTL mode only; mvsusie / fsusie tokens):
#   --joint-specification  Comma-separated joint-fit axes (subset of
#                   context,trait,study). "context" joins a gene's contexts
#                   (cross-context mvSuSiE); "trait" joins the genes in the
#                   region (cross-trait / multi-gene). Empty (default) leaves
#                   the implicit per-(context,trait) branch.
#   --twas-weights  Optional TwasWeights RDS from a preceding mr.mash run
#                   (twas_weights.R --methods mrmash) supplying the mvSuSiE
#                   data-driven reweighted prior. Omit for the canonical
#                   prior. Requires a pecotmr build with twasWeights support.
#   --data-driven-prior-weights-cutoff  Prior-component weight floor for the
#                   reweighted mvSuSiE prior (fineMappingPipeline
#                   dataDrivenPriorWeightsCutoff). Only used with --twas-weights.
#   --use-pca / --n-pcs  Fine-map each multi-trait context's top principal
#                   components with univariate SuSiE (ports fsusie.R
#                   susie_on_top_pc). Requires a pecotmr build with usePCA.
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
parser <- add_argument(parser, "--contexts",
                       help = "Comma-separated context names to restrict to (QTL mode); empty = all contexts",
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
parser <- add_argument(parser, "--seed",
                       help = "Integer RNG seed set before fitting (reproducibility); unset = no seeding",
                       type = "integer", default = NA)
parser <- add_argument(parser, "--pip-cutoff-to-skip",
                       help = "Single-effect (SER) pre-screen cutoff (fineMappingPipeline pipCutoffToSkip), QTL mode; 0 = off, <0 = adaptive 3/nVariants",
                       type = "numeric", default = 0)
# --- Multivariate / joint-fit knobs (QTL mode; mvsusie / fsusie). Each is
# opt-in and omitted from the pipeline call when left at its default, so this
# wrapper also runs against a pecotmr build that predates twasWeights / usePCA.
parser <- add_argument(parser, "--joint-specification",
                       help = "Comma-separated joint-fit axes (context,trait,study) for mvsusie/fsusie; empty = implicit per-(context,trait) branch",
                       type = "character", default = "")
parser <- add_argument(parser, "--twas-weights",
                       help = "Optional TwasWeights RDS (preceding mr.mash run) supplying the mvSuSiE data-driven prior; omit for the canonical prior",
                       type = "character", default = "")
parser <- add_argument(parser, "--data-driven-prior-weights-cutoff",
                       help = "Prior-component weight floor for the reweighted mvSuSiE prior (only used with --twas-weights)",
                       type = "numeric", default = 1e-10)
parser <- add_argument(parser, "--use-pca",
                       help = "Fine-map each multi-trait context's top PCs with univariate SuSiE (usePCA; ports fsusie.R susie_on_top_pc)",
                       flag = TRUE)
parser <- add_argument(parser, "--n-pcs",
                       help = "Cap on top principal components per context when --use-pca (nPCs)",
                       type = "integer", default = 10L)
# --- Per-analysis overrides of the QtlDataset's construct-time filters. Each is
# opt-in (unset leaves the dataset's stored value); applied to the loaded object
# before the pipeline runs (QTL mode only).
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
# the RDS. Shared by both QTL workers (see twas_weights.R).
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

# Seed up front for reproducible fits (mirrors the legacy susie_twas set.seed).
if (length(argv$seed) == 1L && !is.na(argv$seed)) set.seed(as.integer(argv$seed))

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

# Optional context restriction (QTL mode): NULL = all contexts in the dataset.
contexts_arg <- if (nzchar(argv$contexts) && argv$contexts != ".")
  trimws(strsplit(argv$contexts, ",", fixed = TRUE)[[1L]]) else NULL

methods <- trimws(strsplit(argv$methods, ",", fixed = TRUE)[[1L]])

# Joint-fit axes for mvsusie/fsusie: "" -> NULL (implicit per-tuple branch).
joint_spec <- if (nzchar(argv$joint_specification) && argv$joint_specification != ".")
  trimws(strsplit(argv$joint_specification, ",", fixed = TRUE)[[1L]]) else NULL
# Optional mr.mash data-driven prior (TwasWeights from a preceding mrmash run).
twas_weights_obj <- if (nzchar(argv$twas_weights) && argv$twas_weights != "." &&
                        file.exists(argv$twas_weights)) readRDS(argv$twas_weights) else NULL

# Build the `methods` argument: the bare tokens, or the named-list {token: kwargs}
# form when --method-args is given. SuSiE L / L_greedy defaults + susie-family
# routing live in pecotmr (.fmNormalizeMethods); the wrapper only forwards the
# --L / --L-greedy values as top-level args (see cs_args). --method-args keys
# must be among the --methods tokens (no silent typos).
if (!is.null(parsed_method_args)) {
  unknown <- setdiff(names(parsed_method_args), methods)
  if (length(unknown) > 0L)
    stop("--method-args has keys not listed in --methods (got '",
         paste(unknown, collapse = ", "),
         "'; --methods = '", paste(methods, collapse = ", "), "').")
}
methods_arg <- if (is.null(parsed_method_args)) methods else
  setNames(lapply(methods, function(tk)
    if (tk %in% names(parsed_method_args)) parsed_method_args[[tk]] else list()),
    methods)

# Credible-set / coverage knobs common to both modes. medianAbsCorr is added
# only when set (NULL -> omitted), so the call also works against a pecotmr
# that predates that argument. L / L_greedy are forwarded top-level; pecotmr
# seeds the susie-family tokens and applies explicit --method-args overrides.
cs_args <- list(methods           = methods_arg,
                L                 = argv$L,
                Lgreedy           = argv[["L_greedy"]],
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
  qd <- apply_qd_filter_overrides(qd, argv)
  # `contexts` and `pipCutoffToSkip` are QTL-mode only, so they ride on qtl_args
  # rather than the mode-shared cs_args.
  # cisWindow is a gene-mode (traitId) knob — it expands each trait's own
  # coordinates. In region mode the literal variant window comes from --region,
  # and passing both is an error, so cisWindow rides on the traitId branch only.
  qtl_args <- c(list(qd), cs_args,
                list(contexts        = contexts_arg,
                     pipCutoffToSkip = argv$pip_cutoff_to_skip))
  # Opt-in multivariate / joint knobs. Each is added only when the user set it,
  # so the call stays compatible with a pecotmr that predates the argument
  # (mirrors the medianAbsCorr handling above).
  if (!is.null(joint_spec)) qtl_args$jointSpecification <- joint_spec
  if (!is.null(twas_weights_obj)) {
    qtl_args$twasWeights <- twas_weights_obj
    qtl_args$dataDrivenPriorWeightsCutoff <- argv$data_driven_prior_weights_cutoff
  }
  if (isTRUE(argv$use_pca)) {
    qtl_args$usePCA <- TRUE
    qtl_args$nPCs   <- argv$n_pcs
  }
  label <- if (has_region) paste0("region '", argv$region, "'")
           else paste0("gene '", argv$gene_id, "'")
  run_fm <- function() if (has_region) {
    do.call(fineMappingPipeline, c(qtl_args, list(region = parse_region(argv$region))))
  } else {
    do.call(fineMappingPipeline, c(qtl_args,
                                   list(traitId = argv$gene_id,
                                        cisWindow = argv$cis_window)))
  }
  # A multivariate fan-out unit can legitimately have nothing to jointly fit:
  # a locus with < 2 genes overlapping it (cross-trait / mnm_genes / fsusie), a
  # single-context trait (cross-context), etc. fineMappingPipeline signals this
  # in two ways that only the multivariate paths (explicit jointSpecification AND
  # auto-detected mvsusie / fsusie) raise -- the univariate SuSiE family never
  # does: the early multivariate guard ("mvsusie/fsusie requires multi-trait
  # [or multi-context] input ...") and the late dispatch check ("no joint fits
  # produced"). For a per-locus batch each is the honest NULL result, not a
  # failure, so write NULL rather than aborting the substep; every other error
  # propagates unchanged.
  res <- tryCatch(run_fm(), error = function(e) {
    if (grepl("no joint fits produced|requires multi-trait|requires multi-context",
              conditionMessage(e))) {
      message("fine_mapping.R: no multivariate fit for ", label,
              " (scope yields < 2 joinable conditions); writing NULL.")
      NULL
    } else stop(e)
  })
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
n_rows <- if (is.null(res)) 0L else nrow(res)
cat(sprintf("Wrote fineMapping result for %s (%d row(s)) to %s\n",
            label, n_rows, argv$output))
