#!/usr/bin/env Rscript
# ctwas_est.R
#
# cTWAS step 2: estimate group prior + prior variance.
# Reads the assembled inputs from ctwas_assemble.R and calls
# pecotmr::estCtwasParam() (which wraps ctwas::assemble_region_data +
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
est <- estCtwasParam(
  inputs,
  thin                   = argv$thin,
  niterPrefit            = argv$niter_prefit,
  niter                  = argv$niter,
  groupPriorVarStructure = argv$group_prior_var_structure,
  ncore                  = argv$ncore,
  fallbackToPrefit       = argv$fallback_to_prefit,
  min_group_size         = argv$min_group_size,
  min_p_single_effect    = argv$min_p_single_effect)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(est, argv$output)
cat(sprintf("Wrote estCtwasParam result to %s\n", argv$output))
cat("group_prior:    ",
    paste(format(est$param$group_prior, digits = 4), collapse = " | "), "\n")
cat("group_prior_var:",
    paste(format(est$param$group_prior_var, digits = 4), collapse = " | "), "\n")
