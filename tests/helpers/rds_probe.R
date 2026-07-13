#!/usr/bin/env Rscript
# rds_probe.R <path.rds>
#
# Emit a compact jsonlite structural summary of an RDS on stdout so the Python
# test side can assert on a wrapper's output without an R testing stack. Reports
# the S4 class, row count (for DFrame-subclass collections), and whichever
# pecotmr accessors resolve (method names / contexts / study). Any accessor that
# does not apply to the object is simply omitted.
suppressWarnings(suppressMessages({
  library(pecotmr)
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("usage: rds_probe.R <path.rds>")
obj <- readRDS(args[[1L]])

summ <- list(class = class(obj)[[1L]])

nr <- tryCatch(nrow(obj), error = function(e) NULL)
if (!is.null(nr)) summ$nrow <- as.integer(nr)

# Optional accessors: include each only when it resolves without error and
# returns something non-NULL. Keeps the probe generic across QtlDataset /
# FineMappingResult / TwasWeights / CtwasResult / ...
for (acc in c("getStudy", "getContexts", "getMethodNames")) {
  if (!exists(acc)) next
  v <- tryCatch(do.call(acc, list(obj)), error = function(e) NULL)
  if (!is.null(v) && length(v) > 0L)
    # I() keeps these as JSON arrays even for length 1, so the Python side can
    # assert uniformly (["susie"], not the auto-unboxed scalar "susie").
    summ[[sub("^get", "", acc)]] <- I(as.character(v))
}

# For a plain named list (e.g. assembled ctwas inputs), report the top-level
# names so the Python side can assert on the expected keys.
if (is.list(obj) && !isS4(obj) && !is.null(names(obj)))
  summ$names <- names(obj)

# CtwasResult: surface the top gene-level signal (max susie_pip + its z), so a
# smoke test can assert numeric parity independent of the SNP-background rows.
if (methods::is(obj, "CtwasResult") && exists("getFinemap")) {
  fm <- tryCatch(as.data.frame(getFinemap(obj)), error = function(e) NULL)
  if (!is.null(fm) && "susie_pip" %in% names(fm) && nrow(fm) > 0L) {
    genes <- if ("type" %in% names(fm)) fm[fm$type != "SNP", , drop = FALSE] else fm
    if (nrow(genes) > 0L) {
      top <- genes[which.max(genes$susie_pip), , drop = FALSE]
      summ$nGeneRows  <- nrow(genes)
      summ$geneMaxPip <- as.numeric(top$susie_pip[[1L]])
      if ("z" %in% names(genes)) summ$geneTopZ <- as.numeric(top$z[[1L]])
    }
  }
}

cat(toJSON(summ, auto_unbox = TRUE, null = "null"))
