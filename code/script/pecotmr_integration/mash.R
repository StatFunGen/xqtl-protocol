#!/usr/bin/env Rscript
# mash.R
#
# Estimate the MASH mixture-component covariance + weights via
# pecotmr::mashPipeline(). Consumes the per-region RDSes produced by
# mash_sumstats_construct.R (one per LD-block region), partitions into
# strong / random / null subsets via pecotmr::mashRandNullSample, wraps
# each partition as a per-context QtlSumStats collection (with a
# pass-through QC record), and calls mashPipeline.
#
# Inputs:
#   --mash-inputs f1.rds [f2.rds ...]   Per-region RDSes (each is
#                                        list(region_id = list(z, region))).
#   --study                              Study label for the synthesised
#                                        QtlSumStats. Default "study".
#   --ld-sketch <RDS>                    GenotypeHandle RDS to embed in
#                                        the synthesised QtlSumStats
#                                        (required by mashPipeline).
#   --n-random / --n-null                Random / null subset sizes
#                                        passed to mashRandNullSample.
#                                        Defaults 4000 / 4000.
#   --exclude-condition c1,c2,...        Conditions to drop from the
#                                        per-context QtlSumStats before
#                                        running MASH. Default none.
#   --alpha                              alpha argument to mashPipeline().
#                                        Default 0 (standard scale).
#   --seed                               RNG seed. Default 999.
#   --vhat-rds <RDS>                     Optional pre-computed residual
#                                        correlation matrix (vhat). When
#                                        supplied, replaces
#                                        estimate_null_correlation_simple()
#                                        inside mashPipeline; the wrapper
#                                        readRDS()s the file and passes
#                                        the matrix as an in-memory R
#                                        object (no I/O inside pecotmr).
#   --prior-rds <RDS>                    Optional pre-computed prior
#                                        covariance matrices. The RDS is
#                                        expected to contain either a
#                                        named list of square matrices
#                                        directly, OR a list with a `$U`
#                                        slot holding that list (the
#                                        legacy `protocol_example.EE.prior.rds`
#                                        shape). The wrapper extracts and
#                                        forwards the U list to
#                                        mashPipeline's `priorCovariances`
#                                        argument.
#   --output <RDS>                       MASH result RDS (U + w + meta).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
  library(S4Vectors)
})

parser <- arg_parser("Run pecotmr::mashPipeline on per-region MASH inputs")
parser <- add_argument(parser, "--mash-inputs",
                       help = "Per-region MASH input RDSes",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--study",
                       help = "Study label for the synthesised QtlSumStats",
                       type = "character", default = "study")
parser <- add_argument(parser, "--ld-sketch",
                       help = "GenotypeHandle RDS (required by mashPipeline's QC gate)",
                       type = "character", default = "")
parser <- add_argument(parser, "--n-random",
                       help = "Random subset size for mashRandNullSample",
                       type = "integer", default = 4000L)
parser <- add_argument(parser, "--n-null",
                       help = "Null subset size for mashRandNullSample",
                       type = "integer", default = 4000L)
parser <- add_argument(parser, "--exclude-condition",
                       help = "Comma-separated conditions to drop",
                       type = "character", default = "")
parser <- add_argument(parser, "--alpha",
                       help = "alpha argument to mashPipeline()",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--seed",
                       help = "RNG seed", type = "integer", default = 999L)
parser <- add_argument(parser, "--vhat-rds",
                       help = "Optional pre-computed residual correlation matrix RDS",
                       type = "character", default = "")
parser <- add_argument(parser, "--prior-rds",
                       help = "Optional pre-computed prior-covariance RDS (matrix list or list with $U)",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output MASH-result RDS path",
                       type = "character")
argv <- parse_args(parser)

# Load the optional pre-computed mash artefacts at the I/O boundary so
# the pecotmr-side call stays purely in-memory.
vhat_obj <- NULL
if (nzchar(argv$vhat_rds)) {
  if (!file.exists(argv$vhat_rds))
    stop("--vhat-rds file not found: ", argv$vhat_rds)
  vhat_obj <- readRDS(argv$vhat_rds)
  if (!is.matrix(vhat_obj) || !is.numeric(vhat_obj))
    stop("--vhat-rds must deserialise to a numeric matrix (got '",
         class(vhat_obj)[[1L]], "').")
  if (nrow(vhat_obj) != ncol(vhat_obj))
    stop("--vhat-rds matrix must be square; got ",
         nrow(vhat_obj), " x ", ncol(vhat_obj), ".")
}
prior_U <- NULL
if (nzchar(argv$prior_rds)) {
  if (!file.exists(argv$prior_rds))
    stop("--prior-rds file not found: ", argv$prior_rds)
  raw <- readRDS(argv$prior_rds)
  # Accept either a named list of matrices OR a list with $U.
  prior_U <- if (is.list(raw) && !is.null(raw$U)) raw$U else raw
  if (!is.list(prior_U) || length(prior_U) == 0L ||
      is.null(names(prior_U)) || any(names(prior_U) == ""))
    stop("--prior-rds must hold (or have a $U slot containing) a non-empty ",
         "named list of square matrices.")
}

inputs <- as.character(argv$mash_inputs)
if (length(inputs) == 0L)
  stop("--mash-inputs requires at least one per-region RDS.")
if (!nzchar(argv$ld_sketch) || !file.exists(argv$ld_sketch))
  stop("--ld-sketch is required and must point at a GenotypeHandle RDS ",
       "(mashPipeline gates on QC'd sumstats, which carry the ldSketch).")
ld_handle <- readRDS(argv$ld_sketch)
if (!methods::is(ld_handle, "GenotypeHandle"))
  stop("--ld-sketch must deserialise to a GenotypeHandle (got '",
       class(ld_handle)[[1L]], "').")

exclude <- if (nzchar(argv$exclude_condition)) {
  trimws(strsplit(argv$exclude_condition, ",", fixed = TRUE)[[1L]])
} else character(0)

# Load and concatenate per-region inputs. Each entry is
# list(region_id = list(z = matrix, region = chr:start-end))
dat <- list()
for (path in inputs) {
  x <- readRDS(path)
  if (!is.list(x) || length(x) == 0L)
    stop(path, " is not a non-empty list (expected per-region MASH input).")
  for (rid in names(x)) {
    if (rid %in% names(dat))
      stop("Duplicate region_id '", rid, "' across --mash-inputs (",
           path, "); regions must be unique.")
    dat[[rid]] <- x[[rid]]
  }
}

# Build a single concatenated `dat` for mashRandNullSample. We require
# every region to share the same condition set (so column-binding the
# z-matrices stays meaningful for MASH).
conditions <- colnames(dat[[1L]]$z)
for (rid in names(dat)) {
  if (!identical(colnames(dat[[rid]]$z), conditions))
    stop("Region '", rid,
         "' has a different condition set than the first region '",
         names(dat)[[1L]],
         "'; mashRandNullSample requires a common set of conditions.")
}
zStack <- do.call(rbind, lapply(dat, function(x) x$z))
cat(sprintf("Stacked %d region(s) -> %d variants x %d conditions\n",
            length(dat), nrow(zStack), ncol(zStack)))

# Partition into strong / random / null. Strong is the input itself
# (max(|z|) >= threshold variants are downstream-filtered by mash);
# random / null are the subsets mashRandNullSample emits.
partition <- mashRandNullSample(
  list(z = zStack),
  nRandom          = argv$n_random,
  nNull            = argv$n_null,
  excludeCondition = exclude,
  seed             = argv$seed)
randomZ <- partition$random$z
nullZ   <- partition$null$z
if (is.null(randomZ) || nrow(randomZ) == 0L)
  stop("mashRandNullSample returned an empty random subset; ",
       "increase --n-random or check the input z-matrix.")
strongZ <- zStack
# Drop excluded conditions if mashRandNullSample applied them.
if (length(exclude) > 0L) {
  keepCols <- setdiff(colnames(strongZ), exclude)
  strongZ <- strongZ[, keepCols, drop = FALSE]
}

# Wrap each partition as a QtlSumStats collection: one row per
# (study, context, trait) with a single GRanges entry of the partition's
# variants. Use synthetic positions when variant IDs don't parse as
# "chr:pos:..." (mashPipeline only reads Z out of mcols).
.toQss <- function(zMat, role) {
  vids <- rownames(zMat)
  if (is.null(vids)) vids <- paste0("var", seq_len(nrow(zMat)))
  # Try chr:pos:... decode; otherwise synthesise chr1 positions.
  m <- regmatches(vids, regexec("^([^:_]+)[:_]([0-9]+)", vids))
  chrom <- vapply(m, function(x) if (length(x) >= 2L) x[[2L]] else "chr1",
                  character(1L))
  pos   <- suppressWarnings(vapply(m,
            function(x) if (length(x) >= 3L) as.integer(x[[3L]]) else NA_integer_,
            integer(1L)))
  pos[is.na(pos)] <- seq_along(pos)[is.na(pos)]
  # One per-context entry.
  entries <- lapply(seq_len(ncol(zMat)), function(j) {
    gr <- GRanges(seqnames = chrom,
                  ranges   = IRanges(start = pos, width = 1L))
    mcols(gr) <- DataFrame(
      SNP = vids, A1 = rep("A", length(vids)), A2 = rep("G", length(vids)),
      Z   = as.numeric(zMat[, j]),
      N   = rep(1000L, length(vids)))
    gr
  })
  ctxs <- colnames(zMat)
  qss <- QtlSumStats(
    study    = rep(argv$study, ncol(zMat)),
    context  = ctxs,
    trait    = rep("mash", ncol(zMat)),
    entry    = entries,
    genome   = "GRCh38",
    ldSketch = ld_handle,
    qcInfo   = list(role = role,
                    entryAudit = vector("list", ncol(zMat))))
  qss
}

sumStatsList <- list(strong = .toQss(strongZ, "strong"),
                     random = .toQss(randomZ, "random"))
if (!is.null(nullZ) && nrow(nullZ) > 0L)
  sumStatsList$null <- .toQss(nullZ, "null")

cat(sprintf("Built sumStatsList: strong=%d, random=%d, null=%s\n",
            nrow(strongZ), nrow(randomZ),
            if (is.null(nullZ)) "(none)" else as.character(nrow(nullZ))))

res <- mashPipeline(
  sumStatsList        = sumStatsList,
  alpha               = argv$alpha,
  residualCorrelation = vhat_obj,
  priorCovariances    = prior_U,
  setSeed             = argv$seed)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output, compress = "xz")
cat(sprintf("Wrote MASH result (U + w) to %s\n", argv$output))
