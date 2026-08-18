#!/usr/bin/env Rscript
# qtl_enrichment.R
#
# xQTL-GWAS enrichment worker. Loads an S4 `QtlFineMappingResult` and
# an S4 `GwasFineMappingResult`, calls
# `pecotmr::qtlEnrichmentPipeline()`, and saves the resulting
# per-(gwasStudy, qtlContext) enrichment tibble. The output feeds
# downstream into `coloc.R` (via colocPipeline's `enrichment` arg) as
# the prior-adjustment factor for colocalization.
#
# Inputs:
#   --qtl-fine-mapping     Path to S4 QtlFineMappingResult RDS
#                          (output of fine_mapping.ipynb / fineMappingPipeline)
#   --gwas-fine-mapping    Path to S4 GwasFineMappingResult RDS
#   --num-gwas             Number of GWAS variants per study (pass-through)
#   --pi-qtl               Optional QTL pi value (pass-through)
#   --lambda               Pass-through (default 1)
#   --imp-n                Pass-through (default 25)
#   --ncore                Number of threads (default 1)
#   --seed                 Integer RNG seed for the multiple-imputation sampler
#                          (reproducibility); unset = nondeterministic
#   --output               Output RDS path (per-pair enrichment tibble)

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("xQTL-GWAS enrichment via qtlEnrichmentPipeline()")
parser <- add_argument(parser, "--qtl-fine-mapping",
                       help = "Path(s) to S4 QtlFineMappingResult RDS (combined if >1)",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--gwas-fine-mapping",
                       help = "Path(s) to S4 GwasFineMappingResult RDS (combined if >1)",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--num-gwas",
                       help = "Number of GWAS variants (per study; pass-through)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--pi-qtl",
                       help = "Optional QTL pi value (pass-through)",
                       type = "numeric", default = NA)
parser <- add_argument(parser, "--lambda",
                       help = "Pass-through lambda",
                       type = "numeric", default = 1)
parser <- add_argument(parser, "--imp-n",
                       help = "Pass-through impN",
                       type = "integer", default = 25L)
parser <- add_argument(parser, "--ncore",
                       help = "Pass-through ncore",
                       type = "integer", default = 1L)
parser <- add_argument(parser, "--seed",
                       help = paste("Integer RNG seed for the multiple-imputation sampler",
                                    "(reproducibility); unset = nondeterministic"),
                       type = "integer", default = NA)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Load one-or-more FineMappingResult RDS(es) and combine (the modern pipeline
# supplies per-study/per-block FMRs directly, replacing the legacy converter's
# multi --rds-files combine step).
.loadCombine <- function(paths) {
  fmrs <- lapply(as.character(paths), readRDS)
  if (length(fmrs) == 1L) return(fmrs[[1L]])
  # Preserve the (shared) ldSketch: combineFineMappingResults defaults it to
  # NULL, but gwas enrichment/coloc require the RSS-derived ldSketch.
  ld <- tryCatch(fmrs[[1L]]@ldSketch, error = function(e) NULL)
  do.call(combineFineMappingResults, c(fmrs, list(ldSketch = ld)))
}
qtlFmr  <- .loadCombine(argv$qtl_fine_mapping)
gwasFmr <- .loadCombine(argv$gwas_fine_mapping)

# The impN imputation rounds are drawn in C++ (qtl_enrichment.h), so a
# main-process set.seed() cannot reach them -- the base seed has to be forwarded,
# and each round derives its own seed from it so the result is independent of
# which OpenMP thread runs which round. Unset --seed keeps the historical
# nondeterministic std::random_device behaviour. Added only when supplied, for
# compatibility with a pecotmr that predates the argument.
seed_val <- if (length(argv$seed) == 1L && !is.na(argv$seed))
              as.integer(argv$seed) else NULL
enr_args <- list(
  gwasFineMappingResult = gwasFmr,
  qtlFineMappingResult  = qtlFmr,
  numGwas               = if (is.na(argv$num_gwas)) NULL else argv$num_gwas,
  piQtl                 = if (is.na(argv$pi_qtl))   NULL else argv$pi_qtl,
  lambda                = argv$lambda,
  impN                  = argv$imp_n,
  numThreads            = argv$ncore)
if (!is.null(seed_val)) enr_args$seed <- seed_val
res <- do.call(qtlEnrichmentPipeline, enr_args)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote qtlEnrichmentPipeline result (%d (gwasStudy, qtlContext) pair(s)) to %s\n",
            nrow(res), argv$output))
