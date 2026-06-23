#!/usr/bin/env Rscript
# legacy_ctwas_weights_to_s4.R
#
# One-shot extractor: legacy `*.ctwas_weights.*.rds` (the OUTPUT of the
# legacy [ctwas_1] step) into the new S4 `pecotmr::TwasWeights` shape.
#
# Why this exists: the legacy MWE ships
# `protocol_example.ctwas_weights.protocol_example_twas_chr22.chr22.rds`
# with strong weights (max|w| ≈ 1.35) that demonstrably drive ctwas to
# the documented gene-Z = 5.46 result, BUT those weights are not
# reproducible from the upstream `.reshaped_toy.rds` via the documented
# pipeline (the legacy code path produces ~1e-5 values, not ~1e-0).
# The legacy `ctwas_weights.rds` therefore appears to come from a
# stronger upstream fit that isn't shipped. To smoke-test our new
# `ctwasPipeline` end-to-end with default thresholds, we extract those
# already-on-correlation-scale weights into an S4 TwasWeights so the
# pipeline sees the same numerical input the legacy ctwas_3 step did.
#
# The extracted weights are STANDARDIZED (already on the correlation
# scale — the legacy multiplied by sqrt(variance) when building the
# file), so the resulting TwasWeightsEntry carries `standardized = TRUE`
# and our `.ctwasBuildWeights` will skip the sqrt(variance) scaling.
#
# Usage:
#   Rscript legacy_ctwas_weights_to_s4.R \
#     --legacy <legacy_ctwas_weights.rds> \
#     --study <study_label> \
#     --method <method_label>  (default "susie")
#     --ld-meta <ld_meta.tsv> --ld-block <chr:start-end>   (or --ld-prefix)
#     --output <out.rds>

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Convert legacy ctwas_weights.rds to S4 TwasWeights")
parser <- add_argument(parser, "--legacy",
                       help = "Path to legacy *.ctwas_weights.*.rds",
                       type = "character")
parser <- add_argument(parser, "--study",
                       help = "Study label to stamp on every entry",
                       type = "character")
parser <- add_argument(parser, "--method",
                       help = "Method label to stamp on every entry (default 'susie')",
                       type = "character", default = "susie")
parser <- add_argument(parser, "--ld-meta",
                       help = "LD-meta TSV path (used with --ld-block)",
                       type = "character", default = "")
parser <- add_argument(parser, "--ld-block",
                       help = "LD block as chr:start-end (with --ld-meta)",
                       type = "character", default = "")
parser <- add_argument(parser, "--ld-prefix",
                       help = "Explicit LD prefix (bypasses LD-meta lookup)",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

resolve_ld_prefix <- function(meta_path, block_str) {
  m <- regmatches(block_str, regexec("^([^:]+):([0-9]+)-([0-9]+)$", block_str))[[1L]]
  if (length(m) != 4L)
    stop("--ld-block must be chr:start-end (got: ", block_str, ")")
  block <- list(chr = m[[2L]], start = as.integer(m[[3L]]),
                end = as.integer(m[[4L]]))
  meta_dir <- dirname(normalizePath(meta_path))
  meta <- read.table(meta_path, header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE, check.names = FALSE,
                     comment.char = "")
  chr_col <- intersect(c("#chr", "#chrom", "chr", "chrom"), names(meta))[1L]
  chr_norm   <- sub("^chr", "", as.character(meta[[chr_col]]), ignore.case = TRUE)
  block_norm <- sub("^chr", "", block$chr, ignore.case = TRUE)
  s <- suppressWarnings(as.integer(meta$start))
  e <- suppressWarnings(as.integer(meta$end))
  whole <- !is.na(s) & !is.na(e) & s == 0L & e == 0L
  hit <- which(chr_norm == block_norm &
               (whole | (s <= block$start & e >= block$end)))
  if (length(hit) != 1L)
    stop("LD-meta lookup for ", block_str, " returned ", length(hit), " rows.")
  pfx <- meta$path[hit]
  if (!startsWith(pfx, "/")) pfx <- file.path(meta_dir, pfx)
  pfx
}

open_handle <- function(prefix) {
  if (file.exists(paste0(prefix, ".pgen"))) GenotypeHandle(plink2Prefix = prefix)
  else if (file.exists(paste0(prefix, ".bed"))) GenotypeHandle(plink1Prefix = prefix)
  else if (file.exists(paste0(prefix, ".gds"))) GenotypeHandle(path = paste0(prefix, ".gds"))
  else if (file.exists(paste0(prefix, ".vcf.gz"))) GenotypeHandle(path = paste0(prefix, ".vcf.gz"))
  else stop("No genotype payload at LD prefix: ", prefix)
}

ld_handle <- if (nzchar(argv$ld_prefix)) {
  open_handle(argv$ld_prefix)
} else if (nzchar(argv$ld_meta) && nzchar(argv$ld_block)) {
  open_handle(resolve_ld_prefix(argv$ld_meta, argv$ld_block))
} else {
  stop("Provide either --ld-prefix or (--ld-meta AND --ld-block).")
}

# --- Walk the legacy structure and build TwasWeightsEntry per gene ----
legacy <- readRDS(argv$legacy)
if (!is.list(legacy) || length(legacy) == 0L)
  stop("Legacy ctwas_weights RDS is empty or not a list: ", argv$legacy)

studies  <- character(0)
contexts <- character(0)
traits   <- character(0)
methods  <- character(0)
entries  <- list()

# Legacy keys: "<molecular_id>|<type>_<context>"; values are a list with
# wgt (variants × 1 matrix), molecular_id, type, context, etc.
for (gene_key in names(legacy)) {
  entry_obj <- legacy[[gene_key]]
  if (!is.list(entry_obj) || is.null(entry_obj$wgt)) next
  wgt_mat <- entry_obj$wgt
  if (nrow(wgt_mat) == 0L) next

  vids <- rownames(wgt_mat)
  wvec <- as.numeric(wgt_mat[, 1L])

  trait     <- entry_obj$molecular_id %||% sub("\\|.*$", "", gene_key)
  ctx_label <- entry_obj$context %||% sub("^.*\\|", "", gene_key)

  weights_one_col <- matrix(wvec, ncol = 1L,
                             dimnames = list(vids, argv$method))

  tw_entry <- TwasWeightsEntry(
    variantIds    = vids,
    weights       = weights_one_col,
    # Already on the correlation scale (legacy multiplied by
    # sqrt(variance)); skip the sqrt(variance) step downstream.
    standardized  = TRUE)

  studies  <- c(studies,  argv$study)
  contexts <- c(contexts, ctx_label)
  traits   <- c(traits,   trait)
  methods  <- c(methods,  argv$method)
  entries  <- c(entries,  list(tw_entry))
}

if (length(entries) == 0L)
  stop("Converter produced 0 entries from ", argv$legacy)

tw <- TwasWeights(
  study    = studies,
  context  = contexts,
  trait    = traits,
  method   = methods,
  entry    = entries,
  ldSketch = ld_handle)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(tw, argv$output)
cat(sprintf("Wrote S4 TwasWeights (%d entries; %d unique traits) to %s\n",
            length(entries), length(unique(traits)), argv$output))
