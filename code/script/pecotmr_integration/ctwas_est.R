#!/usr/bin/env Rscript
# ctwas_est.R
#
# cTWAS step 2: estimate group prior + prior variance.
# Reads the assembled inputs from ctwas_assemble.R and calls
# pecotmr::estCtwasGroupPriors() (which wraps ctwas::assemble_region_data +
# ctwas::est_param). Saves the augmented state — `inputs` + `region_data`
# + `boundary_genes` + `z_gene` + `param` — for ctwas_finemap.R to
# consume downstream.
#
# Inputs:
#   --inputs               RDS produced by ctwas_assemble.R
#   --thin                 Pass-through (default 0.1)
#   --niter-prefit         Pass-through (default 3)
#   --niter                Pass-through (default 30)
#   --group-prior-var-structure  Pass-through (default 'shared_type')
#   --min-group-size       Pass-through (default 100)
#   --min-p-single-effect  Pass-through (default 0.8)
#   --fallback-to-prefit   Flag: on accurate-EM NaN, recover via prefit-only
#   --ncore                Pass-through (default 1)
#   --output               Output RDS path (the est state list)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("cTWAS step 2: estimate group prior + prior variance")
parser <- add_argument(parser, "--inputs",
                       help = "RDS produced by ctwas_assemble.R",
                       type = "character")
parser <- add_argument(parser, "--thin",
                       help = "Thin", type = "numeric", default = 0.1)
parser <- add_argument(parser, "--niter-prefit",
                       help = "Prefit EM iterations",
                       type = "integer", default = 3L)
parser <- add_argument(parser, "--niter",
                       help = "Accurate EM iterations",
                       type = "integer", default = 30L)
parser <- add_argument(parser, "--group-prior-var-structure",
                       help = "Pass-through (default 'shared_type')",
                       type = "character", default = "shared_type")
parser <- add_argument(parser, "--min-group-size",
                       help = "Minimum genes per ctwas group",
                       type = "integer", default = 100L)
parser <- add_argument(parser, "--min-p-single-effect",
                       help = "Minimum p(single effect) for accurate EM region retention",
                       type = "numeric", default = 0.8)
parser <- add_argument(parser, "--fallback-to-prefit",
                       help = "On accurate-EM NaN, fall back to prefit estimates",
                       flag = TRUE)
parser <- add_argument(parser, "--ncore",
                       help = "Number of cores",
                       type = "integer", default = 1L)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path (est state)",
                       type = "character")
argv <- parse_args(parser)

inputs <- readRDS(argv$inputs)
# pecotmr 0.9.0: estCtwasParam() -> estCtwasGroupPriors(), with the EM prior
# knobs in CtwasPriorParam() (groupPriorVarStructure -> varStructure), ncore ->
# numThreads, and the two ctwas::est_param pass-throughs in the est-param
# options bundle under ctwas's own names.
est <- estCtwasGroupPriors(
  inputs,
  ctwasPriorArgs = CtwasPriorParam(
    thin             = argv$thin,
    niterPrefit      = argv$niter_prefit,
    niter            = argv$niter,
    varStructure     = argv$group_prior_var_structure,
    fallbackToPrefit = argv$fallback_to_prefit),
  numThreads = argv$ncore,
  # min_group_size is a formal of BOTH ctwas::assemble_region_data and
  # ctwas::est_param. Pre-0.9.0 a single `...` was splatted into every step, so
  # one value reached both; the per-step bundles must each carry it or
  # assemble_region_data silently keeps ctwas's own default (100) and drops
  # every group in a small run. min_p_single_effect is est_param's alone.
  methodArgs = CtwasOptions(
    regionData = CtwasRegionDataOptions(
      min_group_size      = argv$min_group_size),
    estParam = CtwasEstParamOptions(
      min_group_size      = argv$min_group_size,
      min_p_single_effect = argv$min_p_single_effect)))

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(est, argv$output)
cat(sprintf("Wrote estCtwasGroupPriors result to %s\n", argv$output))
cat("group_prior:    ",
    paste(format(est$param$group_prior, digits = 4), collapse = " | "), "\n")
cat("group_prior_var:",
    paste(format(est$param$group_prior_var, digits = 4), collapse = " | "), "\n")
