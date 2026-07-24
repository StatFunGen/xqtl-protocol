#!/usr/bin/env Rscript
# ============================================================
# leafcutter_cluster_regtools.R
# Faithful R port of leafcutter's leafcutter_cluster_regtools.py
# (davidaknowles/leafcutter, clustering/), the default (non-const) path used by
# splicing_calling.ipynb [leafcutter_2].
#
# Pipeline (mirrors the Python function-for-function):
#   pool_junc_reads  -> _pooled          (aggregate junctions, >=3 reads, cluster_intervals)
#   refine_clusters  -> _refined         (refine_linked = connected components by splice site;
#                                         refine_cluster = recursive read-ratio trimming)
#   sort_junctions   -> per-lib sorted   (per-cluster count/total for each library)
#   merge_junctions  -> _perind.counts.gz
#   get_numers       -> _perind_numers.counts.gz
#
# Fidelity notes (required to match the Python byte-for-byte):
#   * by_chrom preserves FIRST-APPEARANCE order of (chrom,strand) across the junc
#     files -> this drives cluster numbering (clu_N) and output row order.
#   * intron bounds: A = chromStart + blockSize[1]; B = chromEnd - blockSize[2] + 1.
#   * pool keeps junctions with total >= 3 reads; chromLst filters ALT contigs.
#   * counts are re-aggregated per library in sort_junctions WITHOUT the >=3 filter.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("leafcutter_cluster_regtools (R port; default non-const path)")
parser <- add_argument(parser, "--juncfiles", type = "character", help = "text file listing junction files")
parser <- add_argument(parser, "--outprefix", type = "character", default = "leafcutter", help = "output prefix")
parser <- add_argument(parser, "--rundir", type = "character", default = ".", help = "output directory")
parser <- add_argument(parser, "--maxintronlen", type = "numeric", default = 100000, help = "max intron length (bp)")
parser <- add_argument(parser, "--minclureads", type = "numeric", default = 30, help = "min reads in a cluster")
parser <- add_argument(parser, "--mincluratio", type = "numeric", default = 0.001, help = "min junction read fraction")
parser <- add_argument(parser, "--nochromcheck", flag = TRUE, help = "skip chromosome-name check")
argv <- parse_args(parser)
if (is.na(argv$juncfiles)) stop("--juncfiles is required")

CHROM_LST <- c(paste0("chr", 1:22), "chrX", "chrY", as.character(1:22), "X", "Y")

# ---- geometry helpers (mirror overlaps() / cluster_intervals()) ---------------------------
overlaps <- function(a, b) !(a[2] < b[1] || b[2] < a[1])

# E: list of integer c(A,B). Returns list of clusters (each a list of c(A,B)), greedy by overlap.
cluster_intervals <- function(E) {
  if (length(E) == 0) return(list())
  A1 <- vapply(E, `[`, integer(1), 1L); A2 <- vapply(E, `[`, integer(1), 2L)
  E <- E[order(A1, A2)]
  current <- E[[1]]; clusters <- list(); cluster <- list()
  for (i in seq_along(E)) {
    if (overlaps(E[[i]], current)) {
      cluster[[length(cluster) + 1L]] <- E[[i]]
    } else {
      clusters[[length(clusters) + 1L]] <- cluster
      cluster <- list(E[[i]])
    }
    current <- c(E[[i]][1], max(current[2], E[[i]][2]))
  }
  if (length(cluster) > 0) clusters[[length(clusters) + 1L]] <- cluster
  clusters
}

# connected components of introns that share a splice site (mirror refine_linked()).
# clusters: list of list(inter=c(A,B), count=N). Returns list of components (each a list of such).
refine_linked <- function(clusters) {
  if (length(clusters) == 0) return(list())
  unassigned <- clusters[-1]
  current <- list(clusters[[1]])
  splicesites <- c(clusters[[1]]$inter[1], clusters[[1]]$inter[2])
  newClusters <- list()
  while (length(unassigned) > 0) {
    finished <- FALSE
    while (!finished) {
      finished <- TRUE
      torm <- integer(0)
      for (i in seq_along(unassigned)) {
        st <- unassigned[[i]]$inter[1]; en <- unassigned[[i]]$inter[2]
        if (st %in% splicesites || en %in% splicesites) {
          current[[length(current) + 1L]] <- unassigned[[i]]
          splicesites <- c(splicesites, st, en)
          finished <- FALSE
          torm <- c(torm, i)
        }
      }
      if (length(torm)) unassigned <- unassigned[-torm]
    }
    newClusters[[length(newClusters) + 1L]] <- current
    current <- list()
    if (length(unassigned) > 0) {
      current <- list(unassigned[[1]])
      splicesites <- c(unassigned[[1]]$inter[1], unassigned[[1]]$inter[2])
      unassigned <- unassigned[-1]
    }
  }
  newClusters
}

ikey <- function(inter) paste(inter[1], inter[2], sep = ":")

# recursive read-ratio trimming (mirror refine_cluster()).
refine_cluster <- function(clu, cutoff, readcutoff) {
  totN <- sum(vapply(clu, function(x) x$count, numeric(1)))
  intervals <- list(); dic <- list(); reCLU <- FALSE
  for (ic in clu) {
    if (ic$count / totN >= cutoff && ic$count >= readcutoff) {
      intervals[[length(intervals) + 1L]] <- ic$inter
      dic[[ikey(ic$inter)]] <- ic$count
    } else reCLU <- TRUE
  }
  if (length(intervals) == 0) return(list())
  Atmp <- cluster_intervals(intervals)
  A <- list()
  for (cl in Atmp) {
    linked <- refine_linked(lapply(cl, function(x) list(inter = x, count = 0)))
    for (comp in linked) if (length(comp) > 0) A[[length(A) + 1L]] <- lapply(comp, function(z) z$inter)
  }
  mk <- function(inters) lapply(inters, function(x) list(inter = x, count = dic[[ikey(x)]]))
  if (length(A) == 1) {
    rc <- mk(A[[1]])
    if (length(rc) > 1) {
      if (reCLU) return(refine_cluster(rc, cutoff, readcutoff)) else return(list(rc))
    }
    return(list())
  }
  NCs <- list()
  for (comp in A) if (length(comp) > 1) NCs <- c(NCs, refine_cluster(mk(comp), cutoff, readcutoff))
  NCs
}

# ---- read a junc file (regtools BED12); return parsed introns ------------------------------
# Returns data.frame(chrom, strand, A, B, count) after block-offset adjustment + filters.
# `apply_max`/`apply_min3` toggle the pool-only filters; sort re-reads without them.
read_junc <- function(path, nochromcheck) {
  suppressPackageStartupMessages(library(vroom))
  j <- vroom::vroom(path, delim = "\t", col_names = FALSE, col_types = cols(.default = "c"),
                    progress = FALSE)
  if (nrow(j) == 0) return(data.frame(chrom = character(0), strand = character(0),
                                      A = integer(0), B = integer(0), count = integer(0)))
  blockCount <- as.integer(j$X10)
  keep <- blockCount <= 2                                   # blockCount>2 lines are skipped
  bs <- strsplit(j$X11, ",", fixed = TRUE)
  Aoff <- as.integer(vapply(bs, `[`, character(1), 1L))
  Boff <- as.integer(vapply(bs, `[`, character(1), 2L))
  df <- data.frame(chrom = j$X1, strand = j$X6,
                   A = as.integer(j$X2) + Aoff, B = as.integer(j$X3) - Boff + 1L,
                   count = as.integer(j$X5), stringsAsFactors = FALSE)[keep, , drop = FALSE]
  if (!nochromcheck) df <- df[df$chrom %in% CHROM_LST, , drop = FALSE]
  df
}

# ---- pool_junc_reads ----------------------------------------------------------------------
pool_junc_reads <- function(libl, argv) {
  maxIntronLen <- as.integer(argv$maxintronlen)
  all <- do.call(rbind, lapply(libl, function(f) {
    d <- read_junc(f, argv$nochromcheck)
    d <- d[d$strand != "?", , drop = FALSE]                 # pool skips ambiguous strand
    d[(d$B - d$A) <= maxIntronLen, , drop = FALSE]
  }))
  grp <- paste(all$chrom, all$strand, sep = "\a")
  # aggregate counts by (chrom,strand,A,B); preserve first-appearance order of groups + introns
  ikeys <- paste(grp, all$A, all$B, sep = "\a")
  agg <- rowsum(all$count, ikeys, reorder = FALSE)
  first_idx <- !duplicated(ikeys)
  meta <- all[first_idx, c("chrom", "strand", "A", "B")]
  meta$count <- as.integer(agg[match(ikeys[first_idx], rownames(agg)), 1])
  meta$grp <- grp[first_idx]

  out <- character(0); Ncluster <- 0L
  for (g in unique(meta$grp)) {                             # first-appearance group order
    sub <- meta[meta$grp == g & meta$count >= 3, , drop = FALSE]
    if (nrow(sub) == 0) next
    csg <- strsplit(g, "\a", fixed = TRUE)[[1]]
    E <- Map(function(a, b) c(a, b), sub$A, sub$B)
    cnt <- setNames(sub$count, paste(sub$A, sub$B, sep = ":"))
    for (cl in cluster_intervals(E)) {
      if (length(cl) > 1) {
        parts <- vapply(cl, function(x) sprintf("%d:%d:%d", x[1], x[2], cnt[[ikey(x)]]), character(1))
        out <- c(out, paste0(csg[1], ":", csg[2], " ", paste(parts, collapse = " "), " "))
      }
      Ncluster <- Ncluster + 1L
    }
  }
  writeLines(out, file.path(argv$rundir, paste0(argv$outprefix, "_pooled")))
}

# ---- refine_clusters ----------------------------------------------------------------------
refine_clusters <- function(argv) {
  minratio <- argv$mincluratio; minreads <- as.integer(argv$minclureads)
  inFile <- file.path(argv$rundir, paste0(argv$outprefix, "_pooled"))
  out <- character(0)
  for (ln in readLines(inFile)) {
    toks <- strsplit(trimws(ln), " ", fixed = TRUE)[[1]]
    chrom <- toks[1]
    clu <- lapply(toks[-1], function(ex) {
      v <- as.integer(strsplit(ex, ":", fixed = TRUE)[[1]])
      list(inter = c(v[1], v[2]), count = v[3])
    })
    totN <- sum(vapply(clu, function(x) x$count, numeric(1)))
    if (totN < minreads) next
    for (cl in refine_linked(clu)) {
      rc <- refine_cluster(cl, minratio, minreads)
      if (length(rc) > 0) for (cc in rc) {
        parts <- vapply(cc, function(x) sprintf("%d:%d:%d", x$inter[1], x$inter[2], x$count), character(1))
        out <- c(out, paste0(chrom, " ", paste(parts, collapse = " "), " "))
      }
    }
  }
  writeLines(out, file.path(argv$rundir, paste0(argv$outprefix, "_refined")))
}

# ---- sort_junctions + merge_junctions + get_numers ----------------------------------------
# Reads _refined -> cluster exon lists (cluN in file order); per lib computes count/total.
finalize <- function(libl, argv) {
  refFile <- file.path(argv$rundir, paste0(argv$outprefix, "_refined"))
  cluExons <- list(); cluN <- 0L
  for (ln in readLines(refFile)) {
    toks <- strsplit(trimws(ln), " ", fixed = TRUE)[[1]]
    chrom <- toks[1]; cluN <- cluN + 1L
    cluExons[[cluN]] <- lapply(toks[-1], function(ex) {
      ab <- as.integer(strsplit(ex, ":", fixed = TRUE)[[1]][1:2]); c(A = ab[1], B = ab[2])
    })
    attr(cluExons[[cluN]], "chrom") <- chrom                # "chrN:strand"
  }

  libnames <- sub("\\.junc.*$", "", basename(libl))         # Python libN.split(".junc")[0]
  # per-lib count map keyed by chrom\astrand\aA\aB (no >=3 / maxlen filter in sort)
  perlib <- lapply(libl, function(f) {
    d <- read_junc(f, argv$nochromcheck)
    k <- paste(d$chrom, d$strand, d$A, d$B, sep = "\a")
    tapply(d$count, k, sum)
  })

  # build the count/total strings per cluster per lib
  header <- paste(c("chrom", libnames), collapse = " ")
  numer_header <- paste(libnames, collapse = " ")
  body <- character(0); numer <- character(0)
  for (ci in seq_along(cluExons)) {
    exons <- cluExons[[ci]]
    cs <- strsplit(attr(cluExons[[ci]], "chrom"), ":", fixed = TRUE)[[1]]
    chromID <- cs[1]; strand <- cs[2]
    ex_sorted <- exons[order(vapply(exons, `[`, integer(1), 1L), vapply(exons, `[`, integer(1), 2L))]
    # per-lib totals over this cluster's exons
    keys_by_lib <- lapply(seq_along(libl), function(li)
      vapply(ex_sorted, function(e) paste(chromID, strand, e[1], e[2], sep = "\a"), character(1)))
    tot <- vapply(seq_along(libl), function(li) {
      m <- perlib[[li]][keys_by_lib[[li]]]; sum(m[!is.na(m)])
    }, numeric(1))
    for (ei in seq_along(ex_sorted)) {
      e <- ex_sorted[[ei]]
      id <- sprintf("%s:%d:%d:clu_%d_%s", chromID, e[1], e[2], ci, strand)
      counts <- vapply(seq_along(libl), function(li) {
        v <- perlib[[li]][keys_by_lib[[li]][ei]]; if (is.na(v)) 0L else as.integer(v)
      }, integer(1))
      body  <- c(body,  paste(c(id, sprintf("%d/%d", counts, as.integer(tot))), collapse = " "))
      numer <- c(numer, paste(c(id, sprintf("%d", counts)), collapse = " "))
    }
  }
  gzw <- function(lines, path) { con <- gzfile(path, "wb"); writeLines(lines, con); close(con) }
  gzw(c(header, body),        file.path(argv$rundir, paste0(argv$outprefix, "_perind.counts.gz")))
  gzw(c(numer_header, numer), file.path(argv$rundir, paste0(argv$outprefix, "_perind_numers.counts.gz")))
}

# ---- main ---------------------------------------------------------------------------------
libl <- trimws(readLines(argv$juncfiles))
libl <- libl[nzchar(libl)]
for (f in libl) if (!file.exists(f)) stop(sprintf("%s does not exist", f))
pool_junc_reads(libl, argv)
refine_clusters(argv)
finalize(libl, argv)
message(sprintf("Wrote %s_perind.counts.gz", file.path(argv$rundir, argv$outprefix)))
