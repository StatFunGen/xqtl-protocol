#!/usr/bin/env Rscript
# wrapper_api_check.R [<root> ...]
#
# Static API-contract check for the R wrapper scripts. Parses every wrapper
# WITHOUT running it and reports two classes of break:
#
#   UNRESOLVED  a function is called by a name that resolves nowhere -- not in
#               the file, not in a file it source()s, not in any package it
#               loads, not in base R. This is what an upstream rename looks
#               like from here.
#   BAD-ARG     a named argument that is not in the callee's formals, for a
#               pecotmr function whose signature has no `...` to absorb it.
#
# Exists because a runtime test only covers a wrapper that a test actually
# runs: most of the pecotmr_integration wrappers have no runtime test, and a
# wrapper that wraps its accessor call in tryCatch() can swallow a rename and
# still exit 0. This check sees all of them.
#
# Known blind spots (deliberate, documented rather than silently missed):
#   * do.call(fn, args) -- the argument names are a runtime value, not syntax.
#   * source(<computed path>) -- the helper file is chosen at run time, so for
#     those callers any definition in the scanned tree is accepted.
#   * do.call(fn, args) is the remaining one: the argument names live in a
#     variable, not in the call's syntax. The `...`-dispatching S4 generics
#     (fineMappingPipeline, twasWeightsPipeline, colocboostPipeline,
#     qtlEnrichmentPipeline, ...) ARE checked -- see methodFormals() below.
suppressWarnings(suppressMessages(library(pecotmr)))

args <- commandArgs(trailingOnly = TRUE)
roots <- if (length(args)) args else "code/script"

ns <- asNamespace("pecotmr")
pecotmrExports <- getNamespaceExports(ns)

# The formals a method REALLY takes. When a method's signature differs from its
# generic's, setMethod() stores a wrapper whose own formals are the generic's
# -- `(data, ...)` -- which defines the real function as `.local` and forwards
# to it. Reading formals(m) therefore yields `(data, ...)`, whose `...` makes
# the BAD-ARG check below bail out, leaving every argument of every
# dispatching pipeline unchecked -- a whole class of upstream rename reaching
# the wrappers with this check still reporting clean. Dig `.local` out.
methodFormals <- function(m) {
  nms <- names(formals(m))
  if (!identical(nms[length(nms)], "...")) return(nms)
  b <- body(m)
  if (is.call(b) && identical(as.character(b[[1L]])[1L], "{")) {
    for (st in as.list(b)[-1L]) {
      if (!(is.call(st) && length(st) == 3L &&
            as.character(st[[1L]])[1L] %in% c("<-", "=") &&
            identical(as.character(st[[2L]])[1L], ".local"))) next
      # setMethod() embeds .local as a CLOSURE in the body, not as a call to
      # `function` -- the distinction matters, since is.call() is FALSE for it
      # and a check written only for the call form silently reports clean.
      v <- st[[3L]]
      if (is.function(v)) return(names(formals(v)))
      if (is.call(v) && identical(as.character(v[[1L]])[1L], "function"))
        return(names(formals(eval(v))))
    }
  }
  nms
}

isEmptyArg <- function(x) identical(x, quote(expr = ))

# ---- AST walking -----------------------------------------------------------

# Apply `fn` to every call in an expression tree.
walkCalls <- function(e, fn) {
  if (is.call(e)) {
    fn(e)
    l <- as.list(e)
    for (i in seq_along(l)) {
      if (isEmptyArg(l[[i]])) next
      walkCalls(l[[i]], fn)
    }
  } else if (is.pairlist(e) || is.list(e)) {
    l <- as.list(e)
    for (i in seq_along(l)) {
      if (isEmptyArg(l[[i]])) next
      walkCalls(l[[i]], fn)
    }
  }
  invisible(NULL)
}

# Every name bound by assignment anywhere in the tree (`x <- `, `x = `, `x <<- `,
# assign("x", ...)) plus for-loop variables and function formals: the set of
# names that are NOT free variables at the point of use.
boundNames <- function(exprs) {
  out <- character(0)
  collect <- function(e) {
    if (!is.call(e)) return(invisible())
    op <- as.character(e[[1]])[1]
    if (op %in% c("<-", "=", "<<-") && length(e) >= 2L &&
        (is.name(e[[2]]) || is.character(e[[2]])))
      out <<- c(out, as.character(e[[2]]))
    if (op == "assign" && length(e) >= 2L && is.character(e[[2]]))
      out <<- c(out, e[[2]])
    if (op == "for" && length(e) >= 2L && is.name(e[[2]]))
      out <<- c(out, as.character(e[[2]]))
    if (op == "function" && is.pairlist(e[[2]]))
      out <<- c(out, names(e[[2]]))
  }
  for (e in exprs) walkCalls(e, collect)
  unique(out)
}

# Packages the file attaches, plus every pkg used as pkg::fn.
packagesUsed <- function(exprs) {
  out <- character(0)
  collect <- function(e) {
    if (!is.call(e)) return(invisible())
    op <- as.character(e[[1]])[1]
    if (op %in% c("library", "require", "requireNamespace", "loadNamespace") &&
        length(e) >= 2L) {
      a <- e[[2]]
      if (is.name(a) || is.character(a)) out <<- c(out, as.character(a))
    }
    if (op %in% c("::", ":::") && length(e) >= 2L)
      out <<- c(out, as.character(e[[2]]))
  }
  for (e in exprs) walkCalls(e, collect)
  unique(out)
}

# TRUE when the file source()s a path it computes at run time (e.g.
# `source(argv$lib)`), which no static pass can resolve to a file.
sourcesDynamically <- function(exprs) {
  hit <- FALSE
  collect <- function(e) {
    if (!is.call(e) || as.character(e[[1]])[1] != "source") return(invisible())
    a <- as.list(e)[-1]
    if (length(a) == 0L) return(invisible())
    lits <- FALSE
    walkCalls(e, function(x) {
      for (v in as.list(x)[-1]) {
        if (isEmptyArg(v)) next
        if (is.character(v)) lits <<- TRUE
      }
    })
    if (!lits) hit <<- TRUE
  }
  for (e in exprs) walkCalls(e, collect)
  hit
}

# Sibling files this one source()s, resolved by basename against the same tree.
sourcedFiles <- function(exprs, self, allFiles) {
  out <- character(0)
  collect <- function(e) {
    if (!is.call(e) || as.character(e[[1]])[1] != "source") return(invisible())
    lits <- character(0)
    walkCalls(e, function(x) {
      for (a in as.list(x)[-1]) {
        if (isEmptyArg(a)) next
        if (is.character(a)) lits <<- c(lits, a)
      }
    })
    for (s in lits) {
      hit <- allFiles[basename(allFiles) == basename(s)]
      out <<- c(out, setdiff(hit, self))
    }
  }
  for (e in exprs) walkCalls(e, collect)
  unique(out)
}

# library(pkg) also ATTACHES everything in pkg's Depends:, and those exports
# are visible to the script too -- library(GenomicRanges) is what puts
# IRanges() and seqnames() on the search path. Walk that closure.
# A meta-package attaches its members from .onAttach, which is invisible to
# the DESCRIPTION walk below; list what they put on the search path.
META_PACKAGES <- list(
  tidyverse = c("dplyr", "tidyr", "readr", "purrr", "tibble", "stringr",
                "forcats", "ggplot2", "lubridate")
)

attachClosure <- function(pkgs) {
  seen <- character(0)
  todo <- unique(c(pkgs, unlist(META_PACKAGES[intersect(pkgs, names(META_PACKAGES))],
                                use.names = FALSE)))
  while (length(todo)) {
    pk <- todo[[1L]]
    todo <- todo[-1L]
    if (pk %in% seen) next
    seen <- c(seen, pk)
    dep <- tryCatch(utils::packageDescription(pk)$Depends, error = function(...) NULL)
    if (is.null(dep) || is.na(dep)) next
    dep <- trimws(strsplit(dep, ",")[[1L]])
    dep <- sub("\\s*\\(.*$", "", dep)
    dep <- dep[nzchar(dep) & dep != "R"]
    todo <- c(todo, dep)
  }
  seen
}

# ---- the check -------------------------------------------------------------

files <- unique(unlist(lapply(roots, function(r)
  list.files(r, pattern = "\\.R$", recursive = TRUE, full.names = TRUE))))
files <- files[!grepl("/graveyard/", files)]
files <- sort(files)

findings <- list()
note <- function(file, kind, detail)
  findings[[length(findings) + 1L]] <<-
    data.frame(file = file, kind = kind, detail = detail, stringsAsFactors = FALSE)

parsed <- list()
for (f in files) {
  p <- tryCatch(as.list(parse(f)), error = function(e) {
    note(f, "PARSE", gsub("[\r\n]+", " ", conditionMessage(e)))
    NULL
  })
  if (!is.null(p)) parsed[[f]] <- p
}

# A file that is source()d runs inside the sourcing script's session, so it
# sees whatever that script attached. pecotmr_aliases.R attaches nothing of its
# own and would otherwise look like it calls pecotmr out of nowhere.
sourcedBy <- list()
for (f in names(parsed)) {
  for (sf in sourcedFiles(parsed[[f]], f, files)) {
    sourcedBy[[sf]] <- unique(c(sourcedBy[[sf]], f))
  }
}

for (f in names(parsed)) {
  exprs <- parsed[[f]]
  pkgs <- packagesUsed(exprs)
  for (parent in sourcedBy[[f]]) {
    pp <- parsed[[parent]]
    if (!is.null(pp)) pkgs <- unique(c(pkgs, packagesUsed(pp)))
  }
  # Base + attached packages make up the visible search space. all.names
  # reaches the dot-prefixed base objects (.lm.fit, .Machine, ...).
  base_pkgs <- c("base", "stats", "utils", "methods", "graphics", "grDevices",
                 "tools", "parallel", "compiler", "datasets")
  visible <- unique(unlist(lapply(base_pkgs, function(p)
    tryCatch(ls(getNamespace(p), all.names = TRUE), error = function(...) character(0)))))
  for (pk in attachClosure(pkgs)) {
    nm <- tryCatch(getNamespaceExports(pk), error = function(...) character(0))
    visible <- c(visible, nm)
  }
  local_defs <- boundNames(exprs)
  for (sf in sourcedFiles(exprs, f, files)) {
    sp <- parsed[[sf]]
    if (!is.null(sp)) local_defs <- c(local_defs, boundNames(sp))
  }
  # A run-time source() path (source(argv$lib)) names a file only the caller
  # knows, so accept anything the scanned tree defines rather than reporting
  # the helper it pulls in. Loose by necessity, and only for these files.
  if (sourcesDynamically(exprs)) {
    for (sp in parsed) local_defs <- c(local_defs, boundNames(sp))
  }
  known <- unique(c(visible, local_defs))

  seenUnresolved <- character(0)
  check <- function(e) {
    fn <- e[[1]]
    if (!is.name(fn)) return(invisible())
    nm <- as.character(fn)
    # Operators and control flow are calls too; they are all in base.
    if (!(nm %in% known) && !(nm %in% seenUnresolved)) {
      seenUnresolved <<- c(seenUnresolved, nm)
      note(f, "UNRESOLVED", paste0(nm, "()"))
      return(invisible())
    }
    if (!(nm %in% pecotmrExports)) return(invisible())
    # A local binding of the same name shadows the pecotmr one: do not
    # arg-check against a signature the script is not calling.
    if (nm %in% local_defs) return(invisible())
    target <- get(nm, envir = ns)
    fml <- if (is(target, "genericFunction")) {
      ms <- findMethods(nm, where = ns)
      # The methods' own argument lists, not the generic's `(x, ...)` facade --
      # taking the union with the generic would drag its `...` in and bail out.
      if (length(ms)) {
        per <- lapply(ms, methodFormals)
        # A `...` in ONE method must not disable the check for the generic.
        # The variadic methods here are list fan-outs -- writeBed("list"),
        # writeGwasVcf("list") -- that forward each element to the per-element
        # method, so a name no method names is wrong whichever one dispatches.
        # Only a generic whose every method is variadic is genuinely unknowable.
        if (all(vapply(per, function(f) "..." %in% f, logical(1))))
          unique(unlist(per))
        else setdiff(unique(unlist(per)), "...")
      } else names(formals(args(target)))
    } else {
      names(formals(args(target)))
    }
    if ("..." %in% fml) return(invisible())
    argn <- names(as.list(e))
    if (length(argn) < 2L) return(invisible())
    argn <- argn[-1]
    argn <- argn[!is.na(argn) & nzchar(argn)]
    bad <- setdiff(argn, fml)
    if (length(bad))
      note(f, "BAD-ARG", paste0(nm, "(): ", paste(bad, collapse = ", ")))
  }
  for (e in exprs) walkCalls(e, check)
}

if (length(findings) == 0L) {
  cat("wrapper API check: OK -", length(parsed), "scripts, no findings\n")
  quit(save = "no", status = 0L)
}

df <- unique(do.call(rbind, findings))
df <- df[order(df$file, df$kind, df$detail), ]
cat("wrapper API check: ", nrow(df), " finding(s) in ",
    length(unique(df$file)), " of ", length(files), " scripts\n\n", sep = "")
for (f in unique(df$file)) {
  d <- df[df$file == f, ]
  cat(sub("^\\./", "", f), "\n")
  for (i in seq_len(nrow(d))) cat(sprintf("    %-11s %s\n", d$kind[i], d$detail[i]))
}
cat("\nUNRESOLVED = called by a name nothing provides (an upstream rename, or a typo).\n")
cat("BAD-ARG    = named argument absent from the callee's formals.\n")
quit(save = "no", status = 1L)
