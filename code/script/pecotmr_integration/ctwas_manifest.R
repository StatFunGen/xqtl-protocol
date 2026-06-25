#!/usr/bin/env Rscript
# ctwas_manifest.R
#
# Build the per-LD-block manifest consumed by ctwas_assemble.R. cTWAS runs
# over the whole-chromosome LD-block grid (every LD-reference block is a SNP
# background "region"; only a few carry gene weights), so this worker:
#
#   1. reads each per-gene TwasWeights RDS to learn which genes have weights
#      (the `trait` field) and which file holds each,
#   2. looks up each gene's chromosome + TSS in the xQTL meta table,
#   3. enumerates every LD block on the relevant chromosome(s) from the
#      LD-meta TSV, and
#   4. assigns each gene's TwasWeights to its HOME block (the block whose
#      [start, end) contains the gene's TSS) — assembleCtwasInputs uses a
#      global GWAS-variant union so weight variants that straddle the block
#      boundary still survive.
#
# The emitted manifest has one row per LD block, with the per-block
# GwasSumStats RDS path the caller is expected to build (region column drives
# that fan-out) and a (possibly empty) comma-separated TwasWeights list.
#
# NOTE: no data-layout path is hardcoded. The TwasWeights paths, meta tables,
# and the GwasSumStats output directory all come from arguments; the caller
# decides whether the weights are the upstream twas-step output or a
# substituted set.
#
# Inputs:
#   --ld-meta            LD-meta TSV (#chr/start/end/path); rows are LD blocks
#   --xqtl-meta          xQTL meta TSV (region_id = gene, #chr, TSS, ...)
#   --twas-weights       Comma-separated per-gene TwasWeights RDS paths
#   --gwas-sumstats-dir  Directory the per-block GwasSumStats RDS live in
#                        (path = <dir>/<region_id>.gwas_sumstats.rds)
#   --chrom              Optional chromosome filter (e.g. "22" or "chr22");
#                        default: every chromosome that carries a weighted gene
#   --output            Output manifest TSV
#
# Output columns: region_id, region, gwas_sumstats_rds, twas_weights_rds

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("Build the per-LD-block cTWAS manifest")
p <- add_argument(p, "--ld-meta", type = "character",
                  help = "LD-meta TSV (#chr/start/end/path)")
p <- add_argument(p, "--xqtl-meta", type = "character",
                  help = "xQTL meta TSV (region_id = gene, #chr, TSS)")
p <- add_argument(p, "--twas-weights", type = "character",
                  help = "Comma-separated per-gene TwasWeights RDS paths")
p <- add_argument(p, "--gwas-sumstats-dir", type = "character",
                  help = "Directory holding the per-block GwasSumStats RDS")
p <- add_argument(p, "--chrom", type = "character", default = "",
                  help = "Optional chromosome filter (default: chroms with genes)")
p <- add_argument(p, "--output", type = "character", help = "Output manifest TSV")
argv <- parse_args(p)

splitCsv <- function(s) {
  if (is.null(s) || is.na(s) || !nzchar(s)) return(character(0))
  trimws(strsplit(s, ",", fixed = TRUE)[[1L]])
}
normChr <- function(x) sub("^chr", "", as.character(x), ignore.case = TRUE)

weightPaths <- splitCsv(argv$twas_weights)
if (length(weightPaths) == 0L)
  stop("--twas-weights lists no TwasWeights RDS paths.")

# ---- gene -> weights file (read each RDS; trait field is the gene id) -------
geneToWeight <- list()
for (wp in weightPaths) {
  if (!file.exists(wp)) stop("TwasWeights RDS not found: ", wp)
  tw <- readRDS(wp)
  if (!methods::is(tw, "TwasWeights"))
    stop("Not a TwasWeights RDS: ", wp)
  for (g in unique(as.character(tw$trait))) {
    geneToWeight[[g]] <- union(geneToWeight[[g]], wp)
  }
}
genes <- names(geneToWeight)

# ---- gene -> (chrom, TSS) from the xQTL meta -------------------------------
xqtl <- read.table(argv$xqtl_meta, header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE, check.names = FALSE,
                   comment.char = "")
xchr <- intersect(c("#chr", "#chrom", "chr", "chrom"), names(xqtl))[1L]
if (is.na(xchr) || !all(c("region_id", "TSS") %in% names(xqtl)))
  stop("--xqtl-meta needs region_id, TSS and a chromosome column; got: ",
       paste(names(xqtl), collapse = ", "))
xqtl <- xqtl[!duplicated(xqtl$region_id), , drop = FALSE]
geneChr <- setNames(normChr(xqtl[[xchr]]), xqtl$region_id)
geneTss <- setNames(suppressWarnings(as.integer(xqtl$TSS)), xqtl$region_id)

missing <- setdiff(genes, names(geneChr))
if (length(missing) > 0L)
  stop("Genes have TwasWeights but are absent from --xqtl-meta: ",
       paste(missing, collapse = ", "))

# Chromosomes in scope: the --chrom filter, else every chrom carrying a gene.
chromsWithGenes <- unique(geneChr[genes])
chroms <- if (nzchar(argv$chrom)) normChr(argv$chrom) else chromsWithGenes
if (nzchar(argv$chrom))
  genes <- genes[geneChr[genes] %in% chroms]

# ---- enumerate LD blocks for the in-scope chromosomes ----------------------
ld <- read.table(argv$ld_meta, header = TRUE, sep = "\t",
                 stringsAsFactors = FALSE, check.names = FALSE,
                 comment.char = "")
ldChrCol <- intersect(c("#chr", "#chrom", "chr", "chrom"), names(ld))[1L]
if (is.na(ldChrCol))
  stop("--ld-meta needs a chromosome column; got: ",
       paste(names(ld), collapse = ", "))
ld$.chr   <- normChr(ld[[ldChrCol]])
ld$.start <- suppressWarnings(as.integer(ld$start))
ld$.end   <- suppressWarnings(as.integer(ld$end))
ld <- ld[ld$.chr %in% chroms & !is.na(ld$.start) & !is.na(ld$.end), , drop = FALSE]
ld <- ld[!duplicated(ld[, c(".chr", ".start", ".end")]), , drop = FALSE]
ld <- ld[order(ld$.chr, ld$.start), , drop = FALSE]
if (nrow(ld) < 2L)
  stop("Fewer than two LD blocks in scope (got ", nrow(ld),
       "); cTWAS's EM needs multi-block context.")

region    <- sprintf("chr%s:%d-%d", ld$.chr, ld$.start, ld$.end)
region_id <- gsub("[:-]", "_", region)
gwas_rds  <- file.path(argv$gwas_sumstats_dir,
                       paste0(region_id, ".gwas_sumstats.rds"))

# ---- assign each gene to its home block (TSS in [start, end)) --------------
twas_weights_rds <- rep("", nrow(ld))
for (g in genes) {
  hit <- which(ld$.chr == geneChr[[g]] &
               geneTss[[g]] >= ld$.start & geneTss[[g]] < ld$.end)
  if (length(hit) == 0L) {
    warning("Gene ", g, " (TSS ", geneTss[[g]], ") falls in no LD block; skipped.")
    next
  }
  h <- hit[[1L]]
  cur <- splitCsv(twas_weights_rds[[h]])
  twas_weights_rds[[h]] <- paste(union(cur, geneToWeight[[g]]), collapse = ",")
}

placed <- sum(nzchar(twas_weights_rds))
if (placed == 0L)
  stop("No gene was placed into any LD block; check --xqtl-meta TSS vs --ld-meta.")

out <- data.frame(region_id = region_id, region = region,
                  gwas_sumstats_rds = gwas_rds,
                  twas_weights_rds = twas_weights_rds,
                  stringsAsFactors = FALSE)
dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
write.table(out, file = argv$output, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("Wrote cTWAS manifest: %d LD block(s), %d gene-bearing, to %s\n",
            nrow(out), placed, argv$output))
