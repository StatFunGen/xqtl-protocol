#!/usr/bin/env Rscript
# rds_compare.R <produced.rds> <expected.rds> <tolerance>
#
# Deep-compare two RDS objects and emit a JSON verdict on stdout:
#   {"equal": true}
#   {"equal": false, "report": ["Component \"pip\": Mean relative difference: ...", ...]}
#
# tolerance >= 0  ->  all.equal(produced, expected, tolerance)  (numeric leaves are
#                     compared with all.equal's mean-relative-difference semantics;
#                     everything else must match exactly).
# tolerance <  0  ->  identical()  (byte-for-byte object equality; the "exact" mode).
#
# S4 objects (pecotmr QtlDataset / *FineMappingResult / TwasWeights / CtwasResult /
# ...) are canonicalized to nested named lists of their slots so all.equal recurses
# into the slot contents instead of falling back to a shallow S4 comparison. pecotmr
# is loaded so readRDS can resolve those S4 classes (mirrors rds_probe.R).
suppressWarnings(suppressMessages({
  library(pecotmr)
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) stop("usage: rds_compare.R <produced.rds> <expected.rds> <tolerance> [normPaths]")
tol <- as.numeric(args[[3L]])
# normPaths (optional 4th arg, "1"): replace absolute-path character values with
# their basename before comparing, so machine-specific paths embedded in outputs
# (genotype-handle files, LD-reference paths, cached fit filenames) don't defeat the
# comparison across the CI matrix — the substantive analysis content still compares.
normPaths <- length(args) >= 4L && !is.na(suppressWarnings(as.integer(args[[4L]]))) &&
  as.integer(args[[4L]]) == 1L
# sortRows (optional 5th arg, "1"): canonicalize every data.frame's ROW ORDER (sort by
# all columns) before comparing, for outputs whose rows are an unordered set emitted in a
# non-deterministic order (e.g. SUPPA/psichomics per-event-type tables). Opt-in, since row
# order is meaningful in most outputs.
sortRows <- length(args) >= 5L && !is.na(suppressWarnings(as.integer(args[[5L]]))) &&
  as.integer(args[[5L]]) == 1L

.abs_path_re <- "^/[^ ]+/[^ /]+$"
norm_paths <- function(v) {
  if (is.character(v) && length(v)) {
    ix <- grepl(.abs_path_re, v)
    if (any(ix)) v[ix] <- basename(v[ix])
  }
  v
}

# Recursively turn S4 objects into named lists of their slots so all.equal /
# identical see the contents. Ordinary lists recurse; data.frames and atomics are
# left as-is (all.equal already compares those element-wise with tolerance), except
# under normPaths where character leaves / columns are basename-normalized.
canon <- function(x) {
  # Strip run-to-run-varying TIME metadata to a fixed sentinel (length preserved so a
  # genuine structural diff still surfaces). Wall-clock timings and timestamps are
  # never analysis output in these pipelines — e.g. colocboost's `computing_time`
  # (difftime leaves) and Seurat `@commands` time.stamp (POSIXct). This is the
  # comparator-side normalization that lets workers keep emitting real timings.
  if (inherits(x, c("difftime", "POSIXct", "POSIXt", "Date"))) {
    return(rep("<time-stripped>", length(x)))
  }
  out <- if (isS4(x)) {
    sl <- methods::slotNames(x)
    stats::setNames(lapply(sl, function(s) canon(methods::slot(x, s))), sl)
  } else if (is.data.frame(x)) {
    if (normPaths) x[] <- lapply(x, function(col) if (is.character(col)) norm_paths(col) else col)
    if (sortRows && nrow(x) > 1L) {
      keys <- Filter(is.atomic, as.list(x))    # skip list-columns (order() can't rank them)
      if (length(keys)) {
        x <- x[do.call(order, unname(keys)), , drop = FALSE]
        rownames(x) <- NULL                    # drop the now run-dependent row-name order
      }
    }
    x
  } else if (is.list(x)) {
    lapply(x, canon)
  } else {
    if (normPaths) norm_paths(x) else x
  }
  # Under normPaths, also basename-normalize path-like names / data.frame row.names:
  # some workers derive row.names or element names from the tmp input-file path, which
  # value/column normalization alone misses.
  if (normPaths) {
    nm <- names(out)
    if (is.character(nm)) names(out) <- norm_paths(nm)
    rn <- attr(out, "row.names")
    if (is.character(rn)) attr(out, "row.names") <- norm_paths(rn)
  }
  out
}

produced <- canon(readRDS(args[[1L]]))
expected <- canon(readRDS(args[[2L]]))

if (tol < 0) {
  eq <- identical(produced, expected)
  report <- if (isTRUE(eq)) NULL else "objects are not identical()"
} else {
  cmp <- all.equal(produced, expected, tolerance = tol)
  eq <- isTRUE(cmp)
  report <- if (isTRUE(cmp)) NULL else as.character(cmp)
}

out <- list(equal = eq)
if (!is.null(report)) out$report <- I(report)   # I() keeps a length-1 report a JSON array
cat(toJSON(out, auto_unbox = TRUE, null = "null"))
