#!/usr/bin/env Rscript
# ============================================================
# regtools_junctions.R
# Pure-R/Bioconductor reimplementation of `regtools junctions extract`
# (griffithlab/regtools, src/junctions/junctions_extractor.cc). No custom C++:
# BAM reading + CIGAR access via Rsamtools; the junction algorithm is R.
#
# Faithful to regtools:
#   * walk each read's CIGAR; every `N` op is an intron. `M`/`=` extend the anchor;
#     `D`/`X` (consume reference) and `I`/`S` (don't) BREAK the anchor (no mismatch
#     allowed inside an anchor) and reset thick_start.
#   * per-read junction QC: keep only if min_intron <= (end-start) <= max_intron;
#     an anchor "counts" if its length >= min_anchor. A junction is emitted only if
#     SOME read gave a left anchor >= min_anchor AND SOME read a right anchor
#     (possibly different reads).
#   * aggregate by (chrom,start,end,strand): read_count = n reads; thick_start = min,
#     thick_end = max across reads. Junction NAMES (JUNC%08d) are assigned in
#     first-seen (BAM/cigar) order, over all intron-length-passing junctions
#     (anchor-failing ones still consume a name), then output is sorted by
#     (chrom, thick_start, thick_end, name). BED12, 0-based.
#   * strand: XS aux tag (-s XS), or SAM flag bits (-s RF/FR).
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("regtools junctions extract (pure-R port)")
parser <- add_argument(parser, "--bam", type = "character", help = "indexed BAM")
parser <- add_argument(parser, "--output", type = "character", default = "", help = "output .junc (BED12); default stdout")
parser <- add_argument(parser, "--min-anchor", type = "numeric", default = 8, help = "-a minimum anchor length")
parser <- add_argument(parser, "--min-intron", type = "numeric", default = 70, help = "-m minimum intron length")
parser <- add_argument(parser, "--max-intron", type = "numeric", default = 500000, help = "-M maximum intron length")
parser <- add_argument(parser, "--strandness", type = "character", help = "-s XS | RF | FR")
parser <- add_argument(parser, "--strand-tag", type = "character", default = "XS", help = "-t aux tag for strand (XS mode)")
argv <- parse_args(parser)
if (is.na(argv$bam)) stop("--bam is required")
if (is.na(argv$strandness)) stop("--strandness (XS|RF|FR) is required")

min_anchor <- as.integer(argv$min_anchor); min_intron <- as.integer(argv$min_intron)
max_intron <- as.integer(argv$max_intron)

suppressPackageStartupMessages({ library(Rsamtools); library(GenomicAlignments) })

# --- per-read strand (chunk-local vectors) -------------------------------------------------
read_strand <- function(cigar, flag, xs) {
  if (argv$strandness == "XS") {
    s <- if (is.null(xs)) rep(NA_character_, length(cigar)) else as.character(xs)
    ifelse(is.na(s), "?", s)
  } else {                                                  # RF (1) / FR (2) from flag bits
    inv <- if (argv$strandness == "RF") 1L else 0L          # !bool_strandness: RF-> !0=1 ; FR-> !1=0
    reversed      <- bitwAnd(bitwShiftR(flag, 4), 1L)
    mate_reversed <- bitwAnd(bitwShiftR(flag, 5), 1L)
    first_in_pair  <- bitwAnd(bitwShiftR(flag, 6), 1L)
    second_in_pair <- bitwAnd(bitwShiftR(flag, 7), 1L)
    first_strand  <- bitwXor(bitwXor(inv, first_in_pair), reversed)
    second_strand <- bitwXor(bitwXor(inv, second_in_pair), mate_reversed)
    ifelse(first_strand != second_strand, "?", ifelse(first_strand == 1L, "+", "-"))
  }
}

# --- walk one read's CIGAR into junctions (0-based) -----------------------------------------
walk_read <- function(chrom, pos0, ops, lens, strand) {
  j_start <- pos0; j_tstart <- pos0; j_end <- 0L; j_tend <- 0L; started <- FALSE
  S <- E <- TS <- TE <- integer(0); L <- R <- logical(0)
  emit <- function() {
    ilen <- j_end - j_start
    if (ilen < min_intron || ilen > max_intron) return(invisible())
    S[[length(S) + 1L]]  <<- j_start;  E[[length(E) + 1L]]  <<- j_end
    TS[[length(TS) + 1L]] <<- j_tstart; TE[[length(TE) + 1L]] <<- j_tend
    L[[length(L) + 1L]] <<- (j_start - j_tstart) >= min_anchor
    R[[length(R) + 1L]] <<- (j_tend - j_end)   >= min_anchor
  }
  for (k in seq_along(ops)) {
    op <- ops[k]; len <- lens[k]
    if (op == "N") {
      if (!started) { j_end <- j_start + len; j_tend <- j_end; started <- TRUE }
      else { emit(); j_tstart <- j_end; j_start <- j_tend; j_end <- j_start + len; j_tend <- j_end }
    } else if (op == "M" || op == "=") {
      if (!started) j_start <- j_start + len else j_tend <- j_tend + len
    } else if (op == "D" || op == "X") {
      if (!started) { j_start <- j_start + len; j_tstart <- j_start }
      else { emit(); j_start <- j_tend + len; j_tstart <- j_start; started <- FALSE }
    } else if (op == "I" || op == "S") {
      if (!started) { j_tstart <- j_start }
      else { emit(); j_start <- j_tend; j_tstart <- j_start; started <- FALSE }
    }                                                       # H, P: ignored
  }
  if (started) emit()
  m <- length(S)
  if (m == 0L) return(NULL)
  list(chrom = rep(chrom, m), start = unlist(S), end = unlist(E),
       tstart = unlist(TS), tend = unlist(TE), strand = rep(strand, m),
       left = unlist(L), right = unlist(R))
}

# Fast-path: reads whose CIGAR is `[S] aM bN cM [S]` (one intron, clean anchors, optional flanking
# softclips) are computed vectorially; the rest go through the per-read walk. Softclips give the same
# junction (leading S doesn't change POS; trailing S emits before reset). Both paths keep the exact
# stream order (spl position, then cigar order) so junction naming is unchanged from a full walk.
SIMPLE_RE  <- "^([0-9]+S)?[0-9]+M[0-9]+N[0-9]+M([0-9]+S)?$"
SIMPLE_CAP <- "^([0-9]+S)?([0-9]+)M([0-9]+)N([0-9]+)M([0-9]+S)?$"

# Stream the BAM in chunks so we never hold all reads at once; accumulate only spliced-read junctions.
bf <- BamFile(argv$bam, yieldSize = 1000000L)
param <- ScanBamParam(what = c("rname", "pos", "cigar", "flag"), tag = argv$strand_tag)
acc <- list()
open(bf)
repeat {
  bam <- scanBam(bf, param = param)[[1]]
  if (!length(bam$cigar)) break
  spl <- which(grepl("N", bam$cigar, fixed = TRUE))
  if (!length(spl)) next
  cigs <- bam$cigar[spl]; pos0 <- bam$pos[spl] - 1L; chroms <- as.character(bam$rname)[spl]
  xs_chunk <- bam$tag[[argv$strand_tag]]; if (!is.null(xs_chunk)) xs_chunk <- xs_chunk[spl]
  strands <- read_strand(cigs, bam$flag[spl], xs_chunk)
  simple <- grepl(SIMPLE_RE, cigs)

  ok <- os <- integer(0); vc <- vstr <- character(0)                # order keys + junction vectors
  vst <- ven <- vts <- vte <- integer(0); vl <- vr <- logical(0)
  if (any(simple)) {                                               # vectorized single-intron reads
    si <- which(simple); cs <- cigs[si]; p0 <- pos0[si]
    a  <- as.integer(sub(SIMPLE_CAP, "\\2", cs))
    b  <- as.integer(sub(SIMPLE_CAP, "\\3", cs))
    cc <- as.integer(sub(SIMPLE_CAP, "\\4", cs))
    keep <- b >= min_intron & b <= max_intron
    ok  <- c(ok, si[keep]);          os  <- c(os, integer(sum(keep)))
    vc  <- c(vc, chroms[si][keep]);  vstr <- c(vstr, strands[si][keep])
    vst <- c(vst, (p0 + a)[keep]);   ven <- c(ven, (p0 + a + b)[keep])
    vts <- c(vts, p0[keep]);         vte <- c(vte, (p0 + a + b + cc)[keep])
    vl  <- c(vl, (a >= min_anchor)[keep]); vr <- c(vr, (cc >= min_anchor)[keep])
  }
  if (any(!simple)) {                                              # per-read walk for the rest
    ci <- which(!simple)
    ops_list  <- suppressWarnings(explodeCigarOps(cigs[ci]))   # cigarillo-migration notice
    lens_list <- suppressWarnings(explodeCigarOpLengths(cigs[ci]))
    cpl <- lapply(seq_along(ci), function(j)
      walk_read(chroms[ci[j]], pos0[ci[j]], ops_list[[j]], lens_list[[j]], strands[ci[j]]))
    keepj <- !vapply(cpl, is.null, logical(1)); cpl <- cpl[keepj]; ci <- ci[keepj]
    if (length(cpl)) {
      ms <- vapply(cpl, function(z) length(z$start), integer(1))
      ok  <- c(ok, rep(ci, ms)); os <- c(os, unlist(lapply(ms, function(m) seq_len(m) - 1L)))
      pull <- function(f) unlist(lapply(cpl, `[[`, f))
      vc  <- c(vc, pull("chrom")); vstr <- c(vstr, pull("strand"))
      vst <- c(vst, pull("start")); ven <- c(ven, pull("end"))
      vts <- c(vts, pull("tstart")); vte <- c(vte, pull("tend"))
      vl  <- c(vl, pull("left")); vr <- c(vr, pull("right"))
    }
  }
  o <- order(ok, os)                                               # restore stream order
  acc[[length(acc) + 1L]] <- list(chrom = vc[o], start = vst[o], end = ven[o], tstart = vts[o],
                                  tend = vte[o], strand = vstr[o], left = vl[o], right = vr[o])
}
close(bf)

grab <- function(f) unlist(lapply(acc, `[[`, f))
chrom <- grab("chrom"); start <- grab("start"); end <- grab("end")
tstart <- grab("tstart"); tend <- grab("tend"); strand <- grab("strand")
left <- grab("left"); right <- grab("right")

# --- aggregate by junction key (first-seen order for names) ---------------------------------
key <- paste(chrom, start, end, strand, sep = ":")
first <- !duplicated(key)
uk <- key[first]                                            # unique keys, first-appearance order
df <- data.frame(chrom = chrom[first], start = start[first], end = end[first],
                 strand = strand[first], name = sprintf("JUNC%08d", seq_along(uk)),
                 stringsAsFactors = FALSE)
df$tstart <- as.integer(tapply(tstart, key, min)[uk])
df$tend   <- as.integer(tapply(tend,   key, max)[uk])
df$count  <- as.integer(tapply(key, key, length)[uk])
df$left   <- as.logical(tapply(left,  key, any)[uk])
df$right  <- as.logical(tapply(right, key, any)[uk])

df <- df[df$left & df$right, , drop = FALSE]
df <- df[order(df$chrom, df$tstart, df$tend, df$name), , drop = FALSE]

bed <- data.frame(df$chrom, df$tstart, df$tend, df$name, df$count, df$strand,
                  df$tstart, df$tend, "255,0,0", 2L,
                  paste0(df$start - df$tstart, ",", df$tend - df$end),
                  paste0("0,", df$end - df$tstart))
out <- if (nzchar(argv$output)) argv$output else stdout()
write.table(bed, out, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
