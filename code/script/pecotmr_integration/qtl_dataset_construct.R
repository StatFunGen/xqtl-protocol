#!/usr/bin/env Rscript
# qtl_dataset_construct.R
#
# Build one pecotmr::QtlDataset from a single study's data and serialize
# to RDS. The resulting RDS is the upstream dependency for every per-gene
# fineMappingPipeline / twasWeightsPipeline / colocboostPipeline task;
# gene-level parallelization happens against this single object.
#
# Inputs:
#   --study                Study identifier (length-1 character).
#   --genotype-prefix      PLINK1 bed/bim/fam prefix (no extension).
#   --phenotype-manifest   Path to a TSV manifest. Required columns:
#                            ID       trait identifier
#                            path     bgzipped BED phenotype file
#                            cond     context name
#                          Optional column:
#                            cov_path per-context covariate TSV
#                          One row per (region, context). Phenotype /
#                          covariate paths may be absolute or relative to
#                          the manifest's own directory. Multiple rows can
#                          share the same context; per-context the
#                          phenotype/cov path must agree.
#   --genotype-covariates  Optional TSV of genotype-derived covariates
#                          (e.g. ancestry PCs) applied uniformly across all
#                          contexts. Same shape as the per-context covariate
#                          files.
#   --transpose-covariates When set, transposes every covariate TSV
#                          (phenotype + genotype) before treating it as
#                          samples-as-rows. Use this for QTLtools-format
#                          inputs where rows are covariate names and
#                          columns are samples.
#   --maf-cutoff           Pass-through MAF cutoff for QtlDataset().
#   --xvar-cutoff          Pass-through variance cutoff for QtlDataset().
#   --mac-cutoff           Pass-through minor-allele-count cutoff.
#   --imiss-cutoff         Pass-through per-variant missingness cutoff.
#   --keep-samples         Optional file of sample IDs to keep.
#   --keep-variants        Optional file of variant IDs to keep.
#   --drop-indel           Drop indels (QtlDataset keepIndel = FALSE).
#   --no-scale-residuals   Disable residual scaling (scaleResiduals = FALSE).
#   --output               Output RDS path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Build a pecotmr QtlDataset for one study and save to RDS")
parser <- add_argument(parser, "--study",
                       help = "Study identifier", type = "character")
parser <- add_argument(parser, "--genotype-prefix",
                       help = "PLINK1 bed/bim/fam prefix", type = "character")
parser <- add_argument(parser, "--phenotype-manifest",
                       help = "TSV manifest path (ID, path, cond, cov_path)",
                       type = "character")
parser <- add_argument(parser, "--genotype-covariates",
                       help = "TSV of uniformly-applied genotype PCs",
                       type = "character", default = "")
parser <- add_argument(parser, "--transpose-covariates",
                       help = "Transpose covariate TSVs (QTLtools format)",
                       flag = TRUE)
parser <- add_argument(parser, "--maf-cutoff",
                       help = "QtlDataset() MAF cutoff",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--xvar-cutoff",
                       help = "QtlDataset() variance cutoff",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--mac-cutoff",
                       help = "QtlDataset() minor-allele-count cutoff",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--imiss-cutoff",
                       help = "QtlDataset() per-variant missingness cutoff",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--drop-indel",
                       help = "Drop indel variants (sets keepIndel = FALSE); default keeps them, matching QtlDataset()",
                       flag = TRUE)
parser <- add_argument(parser, "--keep-samples",
                       help = "Path to a whitespace-delimited file of sample IDs to restrict to (QtlDataset keepSamples)",
                       type = "character", default = "")
parser <- add_argument(parser, "--keep-variants",
                       help = "Path to a whitespace-delimited file of variant IDs to restrict to (QtlDataset keepVariants)",
                       type = "character", default = "")
parser <- add_argument(parser, "--no-scale-residuals",
                       help = "Do not scale residuals (QtlDataset scaleResiduals = FALSE; default scales)",
                       flag = TRUE)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# Read a whitespace-delimited ID file into a unique character vector; empty
# (or "." / missing) yields character(0), i.e. "keep all" in QtlDataset().
read_id_file <- function(p) if (nzchar(p) && p != "." && file.exists(p))
  unique(trimws(unlist(strsplit(readLines(p), "[[:space:]]+")))) else character(0)
keep_samples  <- read_id_file(argv$keep_samples)
keep_variants <- read_id_file(argv$keep_variants)

# Absolutise a (possibly non-existent, e.g. a PLINK prefix) path against CWD.
# normalizePath() alone won't absolutise a path with no file on disk, so
# resolve the (existing) parent directory and re-attach the basename.
abs_path <- function(p) file.path(normalizePath(dirname(p), mustWork = FALSE),
                                  basename(p))

# Genotype covariates: pass the path through to the loader (which reads and,
# when --transpose-covariates is set, transposes it); NULL when unset.
# Absolutise so the loader does not re-resolve it against the manifest's own
# directory (the --genotype-* args are CWD-relative, not manifest-relative).
geno_cov_path <- argv$genotype_covariates
genotype_covariates <- if (nzchar(geno_cov_path) && geno_cov_path != "." &&
                           file.exists(geno_cov_path))
  abs_path(geno_cov_path) else NULL

# Absolutise the genotype prefix for the same reason (the loader resolves a
# character genotype spec against the manifest directory).
genotype_prefix <- abs_path(argv$genotype_prefix)

# All manifest ingestion, per-context phenotype/covariate assembly, genotype
# handling, and QtlDataset construction now live in pecotmr's manifest loader.
# The manifest's cond/path/cov_path columns are recognised via the loader's
# snake_case aliases (cond -> context, path -> phenotypePath,
# cov_path -> covariatePath).
qd <- loadQtlDatasetFromManifest(
  manifest            = argv$phenotype_manifest,
  study               = argv$study,
  genotypes           = genotype_prefix,
  genotypeCovariates  = genotype_covariates,
  scaleResiduals      = !isTRUE(argv$no_scale_residuals),
  mafCutoff           = argv$maf_cutoff,
  macCutoff           = argv$mac_cutoff,
  xvarCutoff          = argv$xvar_cutoff,
  imissCutoff         = argv$imiss_cutoff,
  keepSamples         = keep_samples,
  keepVariants        = keep_variants,
  keepIndel           = !isTRUE(argv$drop_indel),
  transposeCovariates = isTRUE(argv$transpose_covariates))

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(qd, argv$output)
contexts <- getContexts(qd)
cat(sprintf("Wrote QtlDataset for study '%s' (%d contexts: %s) to %s\n",
            argv$study, length(contexts),
            paste(contexts, collapse = ", "),
            argv$output))
