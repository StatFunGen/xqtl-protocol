#!/usr/bin/env Rscript
# ctwas_finemap.R
#
# cTWAS step 3: screen regions + fine-map.
# Reads the est state from ctwas_est.R and calls
# pecotmr::screenCtwasRegions() then pecotmr::finemapCtwasRegions().
# Optionally accepts a user-supplied param override RDS so the caller
# can swap in hand-tuned priors when the accurate EM diverged and the
# defaults aren't usable. Saves the final ctwas_sumstats-shape result.
#
# Inputs:
#   --est                  RDS produced by ctwas_est.R
#   --param-override       Optional RDS with $group_prior and
#                          $group_prior_var to substitute before
#                          screen/finemap. Mirrors the legacy ctwas_3
#                          escape hatch for NaN-on-iter-2 EM divergence.
#   --L                    Pass-through (default 5)
#   --min-nonsnp-pip       Pass-through (default 0.5)
#   --merge-regions        Flag: after fine-mapping, merge boundary genes'
#                          adjacent LD blocks and re-fine-map the merged
#                          regions (legacy ctwas_3 merge_regions; default-off)
#   --merge-pip-cutoff     PIP threshold for selecting which boundary genes to
#                          merge (default 0.5)
#   --merge-filter-cs      Flag: require a boundary gene to be in a credible
#                          set to be selected for merging
#   --max-snp              Per-merged-region SNP cap (default Inf)
#   --keep-snps            Flag: retain the SNP background as a dedicated
#                          study=context="SNP" row in the CtwasResult
#   --ncore                Pass-through (default 1)
#   --output               Output RDS path (a CtwasResult: one row per
#                          (gwasStudy, study, context, method), carrying the
#                          per-gene fine-mapping posteriors + susie alphas)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("cTWAS step 3: screen regions + fine-map")
parser <- add_argument(parser, "--est",
                       help = "RDS produced by ctwas_est.R",
                       type = "character")
parser <- add_argument(parser, "--param-override",
                       help = "Optional param-override RDS (overrides $param before screen/finemap)",
                       type = "character", default = "")
parser <- add_argument(parser, "--L",
                       help = "Max number of credible sets per region",
                       type = "integer", default = 5L)
parser <- add_argument(parser, "--min-nonsnp-pip",
                       help = "min_nonSNP_PIP threshold for screen_regions",
                       type = "numeric", default = 0.5)
parser <- add_argument(parser, "--merge-regions",
                       help = "Flag: merge boundary genes' adjacent regions and re-fine-map (legacy ctwas_3 merge_regions)",
                       flag = TRUE)
parser <- add_argument(parser, "--merge-pip-cutoff",
                       help = "PIP threshold for selecting boundary genes to merge",
                       type = "numeric", default = 0.5)
parser <- add_argument(parser, "--merge-filter-cs",
                       help = "Flag: require a boundary gene to be in a credible set to be merged",
                       flag = TRUE)
parser <- add_argument(parser, "--max-snp",
                       help = "Per-merged-region SNP cap",
                       type = "numeric", default = Inf)
parser <- add_argument(parser, "--keep-snps",
                       help = "Flag: retain the SNP background as a dedicated CtwasResult row",
                       flag = TRUE)
parser <- add_argument(parser, "--ncore",
                       help = "Number of cores",
                       type = "integer", default = 1L)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path (ctwas_sumstats-shape result)",
                       type = "character")
argv <- parse_args(parser)

est <- readRDS(argv$est)

# Optional caller-supplied param override (NaN escape hatch).
if (nzchar(argv$param_override) && argv$param_override != "." &&
    file.exists(argv$param_override)) {
  override <- readRDS(argv$param_override)
  if (!is.null(override$group_prior))
    est$param$group_prior <- override$group_prior
  if (!is.null(override$group_prior_var))
    est$param$group_prior_var <- override$group_prior_var
  cat("Applied param override from ", argv$param_override, "\n", sep = "")
}

# pecotmr 0.9.0: ncore -> numThreads everywhere, and L splits by step. The
# screening step has no L formal of its own -- its three knobs are
# ctwas::screen_regions() arguments, which travel under ctwas's own names in
# CtwasScreenOptions() -- while the fine-mapping and merge steps take L as the
# named `maxNumSingleEffects` formal.
screened <- screenCtwasRegions(
  est,
  numThreads = argv$ncore,
  # Only min_nonSNP_PIP: the installed ctwas's screen_regions() has neither an
  # `L` nor a `filter_L` argument. Pre-0.9.0 both were accepted here and then
  # silently discarded (pecotmr filtered pass-throughs to the callee's
  # formals); 0.9.0's options bundle validates against those formals instead,
  # which is what surfaced it. L still matters for the fine-map / merge steps
  # below, where it is a real argument.
  methodArgs = CtwasOptions(
    screen = CtwasScreenOptions(
      min_nonSNP_PIP = argv$min_nonsnp_pip)))

final <- finemapCtwasRegions(
  screened,
  maxNumSingleEffects = argv$L,
  numThreads          = argv$ncore)

# Optional step 4: boundary-gene region merging + re-fine-map.
if (argv$merge_regions) {
  final <- mergeCtwasBoundaryRegions(
    final,
    pipThresh           = argv$merge_pip_cutoff,
    filterCs            = argv$merge_filter_cs,
    maxSNP              = argv$max_snp,
    maxNumSingleEffects = argv$L,
    numThreads          = argv$ncore)
  cat("Applied boundary-region merging (merge_regions).\n")
}

# Structure the granular result as a CtwasResult (one row per
# (gwasStudy, study, context, method); GWAS study read from z_snp, method from
# the gene ids). This is the per-method deliverable downstream consumes.
result <- asCtwasResult(final, keepSnps = argv$keep_snps)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(result, argv$output)
cat(sprintf("Wrote CtwasResult (%d row(s)) to %s\n", nrow(result), argv$output))
fm <- ctwasPosteriors(result)
if (!is.null(fm) && nrow(fm) > 0L) {
  g <- fm[fm$type != "SNP", , drop = FALSE]
  cat(sprintf("  fine-mapped rows: %d (%d gene-level, %d SNP-level)\n",
              nrow(fm), nrow(g), nrow(fm) - nrow(g)))
} else {
  cat("  no fine-mapped genes (no regions survived screening)\n")
}
