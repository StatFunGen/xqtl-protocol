#!/usr/bin/env Rscript
# ============================================================
# collapse_annotation.R
# Standalone CLI worker for reference_data_preparation.ipynb [hg_gtf_2].
#
# R port of the GTEx pipeline's collapse_annotation.py (Francois Aguet), the
# `--collapse_only` path: collapse every gene's transcripts into a single
# gene model (per-gene exon union), dropping retained_intron / readthrough
# transcripts. Byte-faithful to the Python output: the raw attribute string is
# preserved, add_transcript_attributes() fills the GENCODE-order transcript_*
# fields, exon order is reversed on the minus strand, and the exact
# `exon_id "<gene>_<k>; exon_number <k>";` literal is reproduced.
#
# The overlap-removal path (no --collapse-only) is NOT ported: this pipeline
# only ever runs with --collapse-only and an empty blacklist. Requesting it
# raises an error pointing back to the original Python.
#
# Off-the-shelf: per-gene exon union = GenomicRanges::reduce(min.gapwidth=0L),
# which merges overlapping/point-touching intervals but not abutting ones —
# exactly interval_union()'s `i[0] <= union[-1][1]` rule.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
})

parser <- arg_parser("Collapse isoforms into a single gene model (R port of collapse_annotation.py)")
parser <- add_argument(parser, "--input", type = "character", help = "transcript annotation GTF (.gtf/.gtf.gz)")
parser <- add_argument(parser, "--output", type = "character", help = "output collapsed GTF")
parser <- add_argument(parser, "--collapse-only", flag = TRUE,
                       help = "collapse transcripts of each gene (required; the only supported mode)")
parser <- add_argument(parser, "--stranded", flag = TRUE, help = "unsupported (overlap-removal path)")
parser <- add_argument(parser, "--transcript-blacklist", type = "character", default = "",
                       help = "unsupported (must be empty)")
argv <- parse_args(parser)
if (is.na(argv$input) || is.na(argv$output)) stop("--input and --output are required")
if (!argv$collapse_only)
  stop("Only --collapse-only is supported (this pipeline never removes overlaps); ",
       "use the original collapse_annotation.py for the overlap-removal path.")
if (nzchar(argv$transcript_blacklist))
  stop("--transcript-blacklist is not supported (this pipeline always uses an empty blacklist).")

suppressPackageStartupMessages(library(GenomicRanges))

# --- add_transcript_attributes: fill missing transcript_* fields, GENCODE order -----------
add_transcript_attributes <- function(s) {
  if (grepl("gene_status", s)) {
    ord      <- c("gene_id", "transcript_id", "gene_type", "gene_status", "gene_name",
                  "transcript_type", "transcript_status", "transcript_name")
    add_list <- c("transcript_id", "transcript_type", "transcript_status", "transcript_name")
  } else {
    ord      <- c("gene_id", "transcript_id", "gene_type", "gene_name",
                  "transcript_type", "transcript_name")
    add_list <- c("transcript_id", "transcript_type", "transcript_name")
  }
  if (grepl("level", s)) ord <- c(ord, "level")

  parts <- strsplit(sub(";$", "", s), "; ", fixed = TRUE)[[1]]
  keys  <- sub("^(\\S+)\\s.*$", "\\1", parts)
  is_req <- keys %in% ord
  req <- parts[is_req]; opt <- parts[!is_req]
  # value = the SECOND whitespace token (matches Python's split()[1]), ';' stripped
  d <- setNames(gsub(";", "", sub("^\\S+\\s+(\\S+).*$", "\\1", req)), keys[is_req])

  if (!("gene_name" %in% names(d)))     d["gene_name"]     <- d["gene_id"]
  if (!("transcript_id" %in% names(d))) d["transcript_id"] <- d["gene_id"]
  for (k in add_list)
    if (!(k %in% names(d))) d[k] <- d[sub("transcript", "gene", k)]

  paste0(paste(c(paste(ord, d[ord]), opt), collapse = "; "), ";")
}

# --- parse GTF -----------------------------------------------------------------------------
g <- vroom::vroom(argv$input, delim = "\t", col_names = FALSE, comment = "#",
                  col_types = cols(X1 = "c", X2 = "c", X3 = "c", X4 = "i", X5 = "i",
                                   X6 = "c", X7 = "c", X8 = "c", X9 = "c"),
                  .name_repair = ~ paste0("X", seq_along(.x)))
attr9 <- gsub("_biotype", "_type", g$X9)                       # collapse_annotation renames all *_biotype
gene_id <- sub('.*gene_id "([^"]+)".*', "\\1", attr9)
feat <- g$X3

# transcript filter: drop retained_intron type / retained_intron|readthrough_transcript tags
is_tx  <- feat == "transcript"
tx_type <- ifelse(grepl('transcript_type "', attr9), sub('.*transcript_type "([^"]+)".*', "\\1", attr9), NA_character_)
has_excl_tag <- grepl('tag "retained_intron"', attr9) | grepl('tag "readthrough_transcript"', attr9)
tx_ok <- is_tx & (is.na(tx_type) | tx_type != "retained_intron") & !has_excl_tag

# carry each exon's parent-transcript "ok" flag forward: index of the most recent transcript row
n <- length(feat)
last_tx <- cummax(ifelse(is_tx, seq_len(n), 0L))
exon_ok <- feat == "exon" & last_tx > 0 & tx_ok[ifelse(last_tx > 0, last_tx, 1L)]

# --- per-gene exon union (interval_union == reduce(min.gapwidth=0L)) ------------------------
ex <- which(exon_ok)
iv_gene <- gene_id[ex]
red <- reduce(split(IRanges(start = g$X4[ex], end = g$X5[ex]), iv_gene), min.gapwidth = 0L)

# --- gene metadata in input (gene-row) order -----------------------------------------------
gi <- which(feat == "gene")
gmeta <- data.frame(id = gene_id[gi], chr = g$X1[gi], source = g$X2[gi],
                    strand = g$X7[gi], phase = g$X8[gi], attr = attr9[gi],
                    stringsAsFactors = FALSE)
gmeta <- gmeta[gmeta$id %in% names(red), , drop = FALSE]       # only genes with kept exons

# --- write GTF -----------------------------------------------------------------------------
header <- character(0)
con <- file(argv$input, "r")                                   # copy any leading '#' comment lines
repeat { ln <- readLines(con, n = 1); if (!length(ln) || !startsWith(ln, "#")) break; header <- c(header, ln) }
close(con)

lines <- vector("list", nrow(gmeta))
for (r in seq_len(nrow(gmeta))) {
  id <- gmeta$id[r]; iv <- red[[id]]
  starts <- start(iv); ends <- end(iv)
  attr <- if (grepl("transcript_id", gmeta$attr[r])) gmeta$attr[r] else add_transcript_attributes(gmeta$attr[r])
  gstart <- min(starts); gend <- max(ends)
  pre <- paste(gmeta$chr[r], gmeta$source[r], sep = "\t")
  suf <- paste(".", gmeta$strand[r], gmeta$phase[r], attr, sep = "\t")
  gene_line <- paste(pre, "gene",       gstart, gend, suf, sep = "\t")
  tx_line   <- paste(pre, "transcript", gstart, gend, suf, sep = "\t")
  o <- if (gmeta$strand[r] == "-") rev(seq_along(starts)) else seq_along(starts)
  k <- seq_along(o)
  exon_attr <- paste0(attr, ' exon_id "', id, "_", k, "; exon_number ", k, '";')
  exon_lines <- paste(pre, "exon", starts[o], ends[o], ".", gmeta$strand[r], gmeta$phase[r],
                      exon_attr, sep = "\t")
  lines[[r]] <- c(gene_line, tx_line, exon_lines)
}

writeLines(c(header, "##collapsed version generated by GTEx pipeline", unlist(lines)), argv$output)
message(sprintf("Written: %s (%d genes collapsed)", argv$output, nrow(gmeta)))
