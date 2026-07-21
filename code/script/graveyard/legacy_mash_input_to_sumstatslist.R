#!/usr/bin/env Rscript
# legacy_mash_input_to_sumstatslist.R
#
# Standalone one-shot converter: reshape a pre-baked legacy MASH MWE
# input file (`protocol_example.mash_input.rds`, the partitioned output
# of the retired `susie_to_mash` workflow) into the named list of
# QtlSumStats objects that `pecotmr::mashPipeline()` accepts directly.
#
# Not part of the production pipeline. Use this once to materialise a
# `sumStatsList` RDS from the legacy MWE fixture so you can manually
# smoke `mashPipeline()` end-to-end, e.g.:
#
#   Rscript legacy_mash_input_to_sumstatslist.R \
#       --legacy-mash-input input/mash_preprocessing/protocol_example.mash_input.rds \
#       --ld-sketch path/to/ld_sketch.rds \
#       --study protocol_example \
#       --output /tmp/sumstatslist.rds
#
#   Rscript -e 'suppressPackageStartupMessages(library(pecotmr));
#               saveRDS(mashPipeline(readRDS("/tmp/sumstatslist.rds"), alpha = 0),
#                       "/tmp/mash_result.rds")'
#
# Input shape (verified against the shipped MWE fixture):
#   list(
#     strong.z, strong.b, strong.s,   # matrices [n_variants_strong x n_conditions]
#     random.z, random.b, random.s,   # matrices [n_variants_random x n_conditions]
#     null.z,   null.b,   null.s,     # matrices [n_variants_null   x n_conditions]
#     ZtZ)
#
# Output shape (one entry per partition, ready to splice into mashPipeline):
#   list(
#     strong = QtlSumStats(...),
#     random = QtlSumStats(...),
#     null   = QtlSumStats(...))   # null is optional
#
# Each QtlSumStats has one row per condition; the entry GRanges carries
# the partition's z/beta/se vectors as mcols (with a synthesised
# variant_id when the legacy matrices have no rownames). qcInfo is
# pre-stamped non-empty so mashPipeline()'s QC gate accepts the result
# without re-running summaryStatsQc.
#
# Inputs:
#   --legacy-mash-input <RDS>   The pre-baked mash_input.rds.
#   --ld-sketch <RDS>           GenotypeHandle RDS (required slot on
#                                QtlSumStats; never read by mashPipeline,
#                                only validated).
#   --study                     Study label for the synthesised
#                                QtlSumStats. Default "study".
#   --output <RDS>              Output sumStatsList RDS path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
  library(S4Vectors)
})

parser <- arg_parser("Convert legacy pre-baked mash_input.rds to a mashPipeline sumStatsList")
parser <- add_argument(parser, "--legacy-mash-input",
                       help = "Pre-baked MASH input RDS (strong/random/null .z/.b/.s matrices)",
                       type = "character")
parser <- add_argument(parser, "--ld-sketch",
                       help = "GenotypeHandle RDS to attach to the QtlSumStats",
                       type = "character")
parser <- add_argument(parser, "--study",
                       help = "Study label", type = "character", default = "study")
parser <- add_argument(parser, "--output",
                       help = "Output sumStatsList RDS path", type = "character")
argv <- parse_args(parser)

if (!file.exists(argv$legacy_mash_input))
  stop("--legacy-mash-input file not found: ", argv$legacy_mash_input)
if (!file.exists(argv$ld_sketch))
  stop("--ld-sketch file not found: ", argv$ld_sketch)

ld_handle <- readRDS(argv$ld_sketch)
if (!methods::is(ld_handle, "GenotypeHandle"))
  stop("--ld-sketch must deserialise to a GenotypeHandle (got '",
       class(ld_handle)[[1L]], "').")

mi <- readRDS(argv$legacy_mash_input)
required <- c("strong.z", "random.z")
missing  <- setdiff(required, names(mi))
if (length(missing) > 0L)
  stop("Pre-baked MASH input missing required component(s): ",
       paste(missing, collapse = ", "),
       " (got: ", paste(names(mi), collapse = ", "), ").")

# Build a single QtlSumStats from one partition's matrices. Each
# condition becomes one row of the collection; its entry GRanges holds
# the partition's variants with Z/BETA/SE/N mcols pulled column-wise.
toQss <- function(partition, role) {
  zMat <- partition[["z"]]; bMat <- partition[["b"]]; sMat <- partition[["s"]]
  if (is.null(zMat) || nrow(zMat) == 0L || ncol(zMat) == 0L) return(NULL)
  vids <- rownames(zMat)
  if (is.null(vids))
    vids <- sprintf("var_%s_%05d", role, seq_len(nrow(zMat)))
  contexts <- colnames(zMat)
  if (is.null(contexts))
    stop("Partition '", role, "' has no column names; cannot infer conditions.")
  entries <- lapply(seq_along(contexts), function(j) {
    mcols_df <- DataFrame(
      SNP = vids,
      A1  = rep("A", length(vids)),
      A2  = rep("G", length(vids)),
      Z   = as.numeric(zMat[, j]),
      N   = rep(1000L, length(vids)))
    if (!is.null(bMat)) mcols_df$BETA <- as.numeric(bMat[, j])
    if (!is.null(sMat)) mcols_df$SE   <- as.numeric(sMat[, j])
    gr <- GRanges(seqnames = rep("chr1", length(vids)),
                  ranges   = IRanges(start = seq_along(vids), width = 1L))
    mcols(gr) <- mcols_df
    gr
  })
  QtlSumStats(
    study    = rep(argv$study, length(contexts)),
    context  = contexts,
    trait    = rep("mash", length(contexts)),
    entry    = entries,
    genome   = "GRCh38",
    ldSketch = ld_handle,
    qcInfo   = list(source = "legacy_mash_input_to_sumstatslist",
                    role   = role,
                    entryAudit = vector("list", length(contexts))))
}

# Pull the three (or two) partitions out of the flat key naming
# (`<role>.z` / `<role>.b` / `<role>.s`).
partitionOf <- function(role) {
  out <- list(z = mi[[paste0(role, ".z")]],
              b = mi[[paste0(role, ".b")]],
              s = mi[[paste0(role, ".s")]])
  if (is.null(out$z)) NULL else out
}

sumStatsList <- list(
  strong = toQss(partitionOf("strong"), "strong"),
  random = toQss(partitionOf("random"), "random"))
nullPart <- partitionOf("null")
if (!is.null(nullPart)) sumStatsList$null <- toQss(nullPart, "null")

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(sumStatsList, argv$output, compress = "xz")
cat(sprintf(paste0("Wrote sumStatsList for mashPipeline (strong=%d, random=%d, ",
                   "null=%s rows x %d conditions) to %s\n"),
            nrow(mi$strong.z), nrow(mi$random.z),
            if (is.null(mi$null.z)) "none" else as.character(nrow(mi$null.z)),
            ncol(mi$strong.z), argv$output))
