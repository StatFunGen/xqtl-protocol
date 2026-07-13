#!/usr/bin/env Rscript
# mash_sumstats_construct.R
#
# Per-region MASH input builder. Reads one or more multi-context tensorqtl
# summary-statistics files (one per condition / trait combination) for a
# single region, inner-joins them on variant_id into a variants x conditions
# z-matrix, and writes a per-region multi-context QtlSumStats RDS (each
# context = one column) via pecotmr::qtlSumStatsFromZMatrix.
#
# The downstream mash_preprocessing.R worker reads the per-region QtlSumStats
# RDSes and calls pecotmr::mashInput, which selects the strong / random / null
# partitions and assembles the mash_input.rds. (ldSketch stays NULL -- mash
# operates across conditions per variant and needs no LD reference.)
#
# Inputs:
#   --tensorqtl-paths file1 [file2 ...]   Per-condition tensorqtl
#                                          summary-stats files for THIS
#                                          region. Each file must carry
#                                          a `variant_id` column and
#                                          either `z` or `tstat`.
#   --conditions c1,c2,...                Condition labels, same order
#                                          and length as --tensorqtl-paths.
#   --region chr:start-end                Genomic interval label (recorded
#                                          in the log; the QtlSumStats keys
#                                          variants by their ids).
#   --study                               Study label for the QtlSumStats.
#                                          Default "mash".
#   --output <RDS>                        Output QtlSumStats RDS path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Build a per-region MASH input RDS from per-condition tensorqtl outputs")
parser <- add_argument(parser, "--tensorqtl-paths",
                       help = "Per-condition tensorqtl sumstat files",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--conditions",
                       help = "Comma-separated condition labels (one per path)",
                       type = "character")
parser <- add_argument(parser, "--region",
                       help = "Genomic interval as chr:start-end (log label)",
                       type = "character")
parser <- add_argument(parser, "--study",
                       help = "Study label for the QtlSumStats", type = "character",
                       default = "mash")
parser <- add_argument(parser, "--output",
                       help = "Output QtlSumStats RDS path", type = "character")
argv <- parse_args(parser)

paths      <- as.character(argv$tensorqtl_paths)
conditions <- trimws(strsplit(argv$conditions, ",", fixed = TRUE)[[1L]])
if (length(paths) == 0L)
  stop("--tensorqtl-paths requires at least one path.")
if (length(paths) != length(conditions))
  stop("--tensorqtl-paths length (", length(paths),
       ") must match --conditions length (", length(conditions), ").")

# Read one file → return data.frame(variant_id, z)
read_one <- function(path) {
  open_fn <- if (grepl("\\.gz$", path)) function(p) gzfile(p)
             else function(p) p
  df <- read.table(open_fn(path), header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE, check.names = FALSE,
                   comment.char = "")
  pick <- function(opts) intersect(opts, names(df))[1L]
  vid <- pick(c("variant_id", "SNP", "snp", "rsid"))
  zcol <- pick(c("z", "Z"))
  tcol <- pick(c("tstat", "tstatistic", "tstat_val"))
  if (is.na(vid))
    stop(path, " is missing a variant_id column (looked for variant_id / SNP).")
  if (is.na(zcol) && is.na(tcol))
    stop(path, " has neither z nor tstat column.")
  z <- if (!is.na(zcol)) as.numeric(df[[zcol]]) else as.numeric(df[[tcol]])
  data.frame(variant_id = as.character(df[[vid]]),
             z = z, stringsAsFactors = FALSE)
}

# Read every condition, then inner-join on variant_id so the z-matrix is
# rectangular (variants × conditions). This mirrors what the legacy
# `load_multitrait_*_sumstat` helpers did before they were retired.
perCondition <- lapply(paths, read_one)
common <- Reduce(intersect, lapply(perCondition, `[[`, "variant_id"))
if (length(common) == 0L)
  stop("No variant overlaps across the supplied --tensorqtl-paths for region ",
       argv$region, ".")
zMat <- matrix(NA_real_, nrow = length(common), ncol = length(conditions),
               dimnames = list(common, conditions))
for (i in seq_along(perCondition)) {
  d <- perCondition[[i]]
  idx <- match(common, d$variant_id)
  zMat[, i] <- d$z[idx]
}
zMat <- zMat[stats::complete.cases(zMat), , drop = FALSE]
if (nrow(zMat) == 0L)
  stop("No variants left after dropping rows with NA in any condition for region ",
       argv$region, ".")

# Wrap the region's z-matrix as a multi-context QtlSumStats (one column per
# condition). qtlSumStatsFromZMatrix stamps a non-empty qcInfo, so mashInput()
# accepts it directly; ldSketch stays NULL (mash needs no LD reference).
qss <- qtlSumStatsFromZMatrix(zMat, study = argv$study, context = conditions,
                              trait = "mash")

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(qss, argv$output, compress = "xz")
cat(sprintf("Wrote QtlSumStats for region %s: %d variants x %d conditions to %s\n",
            argv$region, nrow(zMat), ncol(zMat), argv$output))
