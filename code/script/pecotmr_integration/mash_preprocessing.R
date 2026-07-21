#!/usr/bin/env Rscript
# mash_preprocessing.R
#
# Assemble the MASH mixture-model input (`mash_input.rds`) from per-region S4
# objects via pecotmr::mashInput -- the merge step of mash_preprocessing.ipynb.
#
# Each --objects RDS is ONE region, holding either:
#   * a QtlSumStats       -- multi-context summary statistics (e.g. from
#                            mash_sumstats_construct.R over tensorqtl files);
#                            strong = the most significant variant (max|z|)
#                            per context.
#   * a FineMappingResult -- multi-context fine-mapping (e.g. from
#                            fine_mapping.R); strong = the lead variant
#                            (max PIP) of each credible set per condition.
# Random / null background rows are sampled from every object identically.
# mashInput() merges the per-region partitions and appends the strong XtX
# cross-product; the output is the flat list(strong.b/strong.s/strong.z,
# random.*, null.*, XtX) consumed by mixture_prior / mash_fit.
#
# Inputs:
#   --objects f1 [f2 ...]   Per-region RDS files (QtlSumStats or
#                            FineMappingResult). Mixed classes are allowed.
#   --region-ids id1,...    Optional comma-separated region labels (one per
#                            object; default = file basenames).
#   --n-random N            Random rows sampled per object. Default 10.
#   --n-null N              Null (max|z|<2) rows sampled per object. Default 10.
#   --exclude-condition c   Comma-separated condition names to drop. Default none.
#   --coverage X            Credible-set coverage for FineMappingResult strong
#                            selection. Default 0.95.
#   --z-only                Emit only the .z matrices (drop .b/.s).
#   --sig-p-cutoff X        Strong-partition significance cutoff. Default 1e-6.
#   --independent-variant-list <path>  Optional LD-pruned independent-SNP list
#                            (a variant_id column or CHROM/POS; gzip OK).
#                            Restricts the random/null background to matching
#                            variants -- allele-aware via matchVariants (strong
#                            is never filtered).
#   --seed N                RNG seed for random/null sampling. Default 999.
#   --output <RDS>          Output mash_input.rds path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Assemble MASH input from per-region S4 objects via mashInput")
p <- add_argument(p, "--objects", type = "character", nargs = Inf,
                  help = "Per-region RDS files (QtlSumStats or FineMappingResult)")
p <- add_argument(p, "--region-ids", type = "character", default = "",
                  help = "Comma-separated region labels (default: file basenames)")
p <- add_argument(p, "--n-random", type = "integer", default = 10L,
                  help = "Random rows sampled per object")
p <- add_argument(p, "--n-null", type = "integer", default = 10L,
                  help = "Null rows sampled per object")
p <- add_argument(p, "--exclude-condition", type = "character", default = "",
                  help = "Comma-separated conditions to drop")
p <- add_argument(p, "--coverage", type = "numeric", default = 0.95,
                  help = "CS coverage for FineMappingResult strong selection")
p <- add_argument(p, "--z-only", flag = TRUE, help = "Emit only .z matrices")
p <- add_argument(p, "--sig-p-cutoff", type = "numeric", default = 1e-6,
                  help = "Strong-partition significance cutoff")
p <- add_argument(p, "--independent-variant-list", type = "character", default = "",
                  help = "Optional LD-pruned independent-SNP list; restricts the random/null background")
p <- add_argument(p, "--seed", type = "integer", default = 999L, help = "RNG seed")
p <- add_argument(p, "--output", type = "character", help = "Output mash_input.rds")
argv <- parse_args(p)

# Read a variant-id list from --independent-variant-list. Handles a header'd
# table with a variant_id (or CHROM/POS) column and gzipped input; matchVariants
# inside mashInput does the actual allele-aware matching.
readIndependentVariantIds <- function(path) {
  df <- suppressWarnings(as.data.frame(vroom::vroom(
    path, comment = "", show_col_types = FALSE, progress = FALSE)))
  if (ncol(df) == 0L || nrow(df) == 0L)
    stop("--independent-variant-list is empty or unreadable: ", path)
  nm <- tolower(sub("^#", "", names(df)))
  idCol <- which(nm %in% c("variant_id", "id", "snp", "rsid"))[1L]
  if (!is.na(idCol)) return(as.character(df[[idCol]]))
  cCol <- which(nm %in% c("chrom", "chr", "chromosome"))[1L]
  pCol <- which(nm %in% c("pos", "position", "bp"))[1L]
  if (!is.na(cCol) && !is.na(pCol))
    return(paste0(df[[cCol]], ":", df[[pCol]]))
  as.character(df[[1L]])
}

paths <- as.character(argv$objects)
if (length(paths) == 0L) stop("--objects requires at least one RDS file.")
missing <- paths[!file.exists(paths)]
if (length(missing) > 0L)
  stop("--objects file(s) not found: ", paste(missing, collapse = ", "))

splitCsv <- function(x) if (nzchar(x)) trimws(strsplit(x, ",", fixed = TRUE)[[1L]]) else character(0)

regionIds <- splitCsv(argv$region_ids)
if (length(regionIds) == 0L) {
  regionIds <- tools::file_path_sans_ext(basename(paths))
}
if (length(regionIds) != length(paths))
  stop("--region-ids length (", length(regionIds), ") must match --objects (",
       length(paths), ").")

objects <- setNames(lapply(paths, readRDS), regionIds)

independentVariants <- if (nzchar(argv$independent_variant_list))
  readIndependentVariantIds(argv$independent_variant_list) else NULL

result <- mashInput(
  objects,
  nRandom             = argv$n_random,
  nNull               = argv$n_null,
  excludeCondition    = splitCsv(argv$exclude_condition),
  coverage            = argv$coverage,
  zOnly               = argv$z_only,
  sigPCutoff          = argv$sig_p_cutoff,
  independentVariants = independentVariants,
  seed                = argv$seed)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(result, argv$output, compress = "xz")

strongN <- if (is.null(result$strong.z)) 0L else nrow(as.matrix(result$strong.z))
randomN <- if (is.null(result$random.z)) 0L else nrow(as.matrix(result$random.z))
nullN   <- if (is.null(result$null.z))   0L else nrow(as.matrix(result$null.z))
nCond   <- if (is.null(result$strong.z)) NA_integer_ else ncol(as.matrix(result$strong.z))
cat(sprintf("Wrote MASH input (strong=%d, random=%d, null=%d rows x %s conditions) to %s\n",
            strongN, randomN, nullN, as.character(nCond), argv$output))
