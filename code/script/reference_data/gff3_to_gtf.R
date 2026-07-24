#!/usr/bin/env Rscript
# ============================================================
# gff3_to_gtf.R
# Standalone CLI worker for reference_data_preparation.ipynb [gff3_to_gtf].
#
# Convert a GFF3 annotation to GTF via rtracklayer (replaces `gffread -T`).
# More faithful than gffread: it preserves gene_type (biotype) and gene_name
# — which the downstream hg_gtf_1 / collapse_annotation steps need — and emits
# clean gene_id / transcript_id (matching the Ensembl-GTF default path) instead
# of gffread's `gene:` / `transcript:`-prefixed IDs.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(rtracklayer)
})

parser <- arg_parser("Convert a GFF3 annotation to GTF (rtracklayer; replaces gffread -T)")
parser <- add_argument(parser, "--input", type = "character", help = "input GFF3")
parser <- add_argument(parser, "--output", type = "character", help = "output GTF")
argv <- parse_args(parser)
if (is.na(argv$input) || is.na(argv$output)) stop("--input and --output are required")

g  <- import(argv$input)
ty <- as.character(g$type)
strip   <- function(x) sub("^[^:]+:", "", x)                       # "gene:X" -> "X"
parent1 <- function(p) vapply(p, function(x) if (length(x)) x[1] else NA_character_, "")
attr_of <- function(name) {
  if (!is.null(mcols(g)[[name]])) as.character(mcols(g)[[name]]) else rep(NA_character_, length(g))
}
# First non-NA across attribute-name variants (Ensembl vs GENCODE spellings).
coalesce_attr <- function(...) Reduce(function(x, y) ifelse(is.na(x), y, x),
                                      lapply(c(...), attr_of))

# Transcripts = features that are the Parent of an exon/CDS.
exon_parents <- unique(unlist(g$Parent[ty %in% c("exon", "CDS")]))
is_tx <- !is.na(g$ID) & g$ID %in% exon_parents

tx_id_attr   <- coalesce_attr("transcript_id")
gene_id_attr <- coalesce_attr("gene_id", "geneID")
gene_nm_attr <- coalesce_attr("gene_name")

# Per-transcript maps keyed by the transcript's GFF `ID`.
txgff   <- g$ID[is_tx]
tx_txid <- ifelse(!is.na(tx_id_attr[is_tx]),   tx_id_attr[is_tx],   strip(txgff))
tx_gid  <- ifelse(!is.na(gene_id_attr[is_tx]), gene_id_attr[is_tx], strip(parent1(g$Parent[is_tx])))
m_txid  <- setNames(tx_txid, txgff)
m_gid   <- setNames(tx_gid,  txgff)

keep <- is_tx | ty %in% c("exon", "CDS")
out  <- g[keep]
par  <- parent1(out$Parent)
gene_id       <- ifelse(is_tx[keep], tx_gid[match(out$ID, txgff)],  unname(m_gid[par]))
transcript_id <- ifelse(is_tx[keep], tx_txid[match(out$ID, txgff)], unname(m_txid[par]))
# gffread normalizes every transcript-level feature (mRNA, ncRNA, rRNA, ...) to "transcript".
new_type <- ifelse(is_tx[keep], "transcript", as.character(out$type))

mc <- S4Vectors::DataFrame(source = out$source, type = new_type, score = out$score,
                           phase = out$phase, gene_id = gene_id, transcript_id = transcript_id)
gname <- gene_nm_attr[keep]
if (any(!is.na(gname))) mc$gene_name <- gname
gtype <- coalesce_attr("gene_type", "biotype")[keep]               # GENCODE 'gene_type' / Ensembl 'biotype'
if (any(!is.na(gtype))) mc$gene_type <- gtype

o <- out
mcols(o) <- mc
export(o, argv$output, "gtf")
message(sprintf("Written: %s (%d features)", argv$output, length(o)))
