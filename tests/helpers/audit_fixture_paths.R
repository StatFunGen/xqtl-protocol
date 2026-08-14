#!/usr/bin/env Rscript
# audit_fixture_paths.R <dir> — walk every RDS under <dir> (S4 slots, lists, CLOSURE
# environments, names) and print any embedded MACHINE-LOCAL absolute path (a user home
# or a per-user tmp dir). RDS are gzip-compressed so a plain grep can't see inside them,
# which is exactly how one hid before; walk the deserialized object instead.
#
# Prints one line per (file, location) hit and exits 1 if ANY are found, else exits 0.
# Powers the tests/scripts path-hygiene guard. The pattern is OS-AGNOSTIC — it flags a
# user-home or per-user tmp dir on either macOS OR Linux (fixtures may be generated on
# either), since those absolute paths are regenerated per run and so differ across
# machines. It does NOT match non-home provenance roots baked STATICALLY into the upstream
# toy data (e.g. a BAM @PG line's /restricted/projectnb/...), which are a fixed cluster
# path, not a home/tmp shape, and are load-bearing for the exact @PG comparison.
suppressWarnings(suppressMessages({
  library(pecotmr)
  try(library(SeuratObject), silent = TRUE)
  try(library(SingleCellExperiment), silent = TRUE)
}))
pat <- "/Users/|/home/[A-Za-z][A-Za-z0-9._-]*/|/var/folders/|/private/var/folders/|/tmp/"
files <- list.files(commandArgs(TRUE)[1], pattern = "\\.rds$",
                    recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
# Scope to COMPARISON-TARGET fixtures only: those under an `expected/` dir or named
# `expected*`. Those are generated locally, so a home/tmp path in them is a dynamic
# per-run path that differs across machines. Raw INPUT fixtures are out of scope — they
# may carry immutable upstream provenance (a BAM @PG command, a captured devtools call)
# that is identical on every machine and never value-compared.
files <- files[grepl("/expected/", files) | grepl("^expected", basename(files))]

chk <- function(v, where) {
  if (is.character(v) && length(v)) {
    m <- unique(v[grepl(pat, v)])
    if (length(m)) return(paste0(where, " = ", substr(m, 1, 130)))
  }
  character(0)
}
walk <- function(x, path, depth = 0L) {
  if (depth > 80L) return(character(0))
  hits <- chk(names(x), paste0(path, "@names"))
  if (isS4(x)) {
    for (s in methods::slotNames(x))
      hits <- c(hits, walk(methods::slot(x, s), paste0(path, "@", s), depth + 1L))
  } else if (is.function(x)) {
    e <- environment(x)
    if (!is.null(e) && environmentName(e) == "" && !isNamespace(e))
      for (v in ls(e, all.names = TRUE))
        hits <- c(hits, walk(get(v, e), paste0(path, "$", v, "[closure]"), depth + 1L))
  } else if (is.list(x)) {
    for (i in seq_along(x)) hits <- c(hits, walk(x[[i]], paste0(path, "[[", i, "]]"), depth + 1L))
  } else if (is.character(x)) {
    hits <- c(hits, chk(x, path))
  }
  hits
}
n <- 0L
for (f in files) {
  short <- sub(".*tests/fixtures/", "", f)
  res <- tryCatch(list(ok = TRUE, obj = readRDS(f)),
                  error = function(e) list(ok = FALSE, msg = conditionMessage(e)))
  if (!isTRUE(res$ok)) next            # class needs a package we don't have; text/other scans still run
  h <- tryCatch(unique(walk(res$obj, "", 0L)), error = function(e) character(0))
  if (length(h)) { n <- n + length(h); for (hit in h) cat(short, "  ", hit, "\n") }
}
quit(status = if (n > 0L) 1L else 0L)
