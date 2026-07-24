#!/usr/bin/env Rscript
# ============================================================
# bulk_expression_normalization.R
# Pure-R port of bulk_expression_normalization.py (GTEx eqtl_prepare_expression)
# for bulk_expression_normalization.ipynb. Removes the Broad `qtl`/`pyqtl`
# dependency (which is not installable here) by using edgeR/limma + rtracklayer
# + Rsamtools.
#
# Steps (--step):
#   normalize — TMM+CPM(voom) / TMM+CPM(edgeR) / quantile normalization of a bulk
#               gene expression matrix, followed by an optional inverse-normal
#               transform, joined to a TSS BED and bgzip/tabix-indexed.
#
# Faithful to the python (qtl.norm / qtl.io):
#   * read_gct            -> GCT with 2 header lines; first column = gene id.
#   * voom_transform      -> log2((counts + 0.5) / (lib_size*TMM + 1) * 1e6),
#                            TMM from edgeR::calcNormFactors on ALL genes.
#   * edger_cpm           -> counts / (lib_size*TMM) * 1e6.
#   * quantile_normalize  -> preprocessCore::normalize.quantiles.
#   * inverse_normal_transform -> per-gene qnorm(rank(x)/(n+1)), ties averaged.
#   * gene expression mask uses the UNnormalized tpm/counts matrices.
#   * gtf_to_bed(feature="gene") -> strand-aware 0-based 1bp TSS interval.
#   * gene ids stripped of Ensembl version (keeping optional _PAR_Y).
#   * sort_bed_for_tabix -> chrom rank (1-22, X=23, Y=24, M/MT=25, else 1000),
#                           then #chr, start, end, gene_id (stable).
# Output name: {prefix}.{method}{.no_qnorm if no qnorm}.expression.bed.gz
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(edgeR)
  library(rtracklayer)
  library(dplyr)
})

parser <- arg_parser("bulk expression normalization (GTEx eqtl_prepare_expression in R)")
parser <- add_argument(parser, "--step", type = "character", default = "normalize",
                       help = "normalize")
parser <- add_argument(parser, "--cwd", type = "character", default = "output",
                       help = "output directory")
parser <- add_argument(parser, "--tpm-gct", type = "character", default = "",
                       help = "[normalize] TPM GCT(.gz)")
parser <- add_argument(parser, "--counts-gct", type = "character", default = "",
                       help = "[normalize] raw gene-count GCT(.gz)")
parser <- add_argument(parser, "--annotation-gtf", type = "character", default = "",
                       help = "[normalize] gene annotation GTF")
parser <- add_argument(parser, "--sample-participant-lookup", type = "character", default = "",
                       help = "[normalize] TSV mapping sample_id -> participant_id")
parser <- add_argument(parser, "--tpm-threshold", type = "numeric", default = 0.1,
                       help = "[normalize] min TPM to count a sample as expressed")
parser <- add_argument(parser, "--count-threshold", type = "numeric", default = 6,
                       help = "[normalize] min count to count a sample as expressed")
parser <- add_argument(parser, "--sample-frac-threshold", type = "numeric", default = 0.2,
                       help = "[normalize] min fraction of samples that must be expressed")
parser <- add_argument(parser, "--normalization-method", type = "character", default = "tmm_cpm_voom",
                       help = "[normalize] tmm_cpm_voom | tmm_cpm_edger | qn")
parser <- add_argument(parser, "--quantile-normalize", flag = TRUE,
                       help = "[normalize] apply per-gene inverse-normal transform after rescale")
parser <- add_argument(parser, "--numThreads", type = "numeric", default = 20,
                       help = "[normalize] unused (kept for CLI compatibility)")
argv <- parse_args(parser)

# ---------------------------------------------------------------------------
# GCT reader: 2 header lines, then header row (gene id + samples). qtl.io.read_gct
# keeps an optional Description column then drops it downstream; we drop it here.
read_gct <- function(path) {
  d <- as.data.frame(vroom::vroom(path, skip = 2, delim = "\t",
                                  show_col_types = FALSE, progress = FALSE))
  ids <- as.character(d[[1]]); d[[1]] <- NULL
  if ("Description" %in% colnames(d)) d[["Description"]] <- NULL
  m <- as.matrix(d); storage.mode(m) <- "double"; rownames(m) <- ids
  m
}

# Strip Ensembl version suffix, preserving optional _PAR_Y (qtl normalize_ensembl_gene_ids).
strip_gene_version <- function(ids) sub("\\.[0-9]+(_PAR_Y)?$", "\\1", ids)

# Per-gene inverse-normal transform across samples (qtl.norm.inverse_normal_transform).
inverse_normal_transform <- function(mat) {
  n <- ncol(mat)
  out <- t(apply(mat, 1, function(x) qnorm(rank(x, ties.method = "average") / (n + 1))))
  dimnames(out) <- dimnames(mat)
  out
}

# TSS BED (feature="gene", strand-aware, 0-based 1bp) via rtracklayer (qtl.io.gtf_to_bed).
gtf_to_tss_bed <- function(annotation_gtf) {
  gr <- rtracklayer::import(annotation_gtf, feature.type = "gene")
  plus <- as.character(strand(gr)) == "+"
  tibble(chr = as.character(seqnames(gr)),
         start = ifelse(plus, start(gr) - 1L, end(gr) - 1L),
         end   = ifelse(plus, start(gr), end(gr)),
         gene_id = gr$gene_id)
}

# tabix chromosome ordering (qtl sort_bed_for_tabix).
chrom_rank <- function(chr) {
  key <- toupper(sub("^chr", "", chr))
  rank <- setNames(1:22, as.character(1:22))
  out <- rank[key]
  out[key == "X"] <- 23; out[key == "Y"] <- 24
  out[key %in% c("M", "MT")] <- 25
  out[is.na(out)] <- 1000
  as.integer(out)
}

run_normalize <- function(argv) {
  dir.create(argv$cwd, showWarnings = FALSE, recursive = TRUE)

  # Output prefix: strip .gct[.gz] then a trailing .tpm / .gene_tpm from the TPM name.
  bname <- basename(argv$tpm_gct)
  bname <- sub("\\.gct(\\.gz)?$", "", bname)
  bname <- sub("\\.(gene_tpm|tpm)$", "", bname)

  qnorm <- isTRUE(argv$quantile_normalize)
  method <- argv$normalization_method

  tpm    <- read_gct(argv$tpm_gct)
  counts <- read_gct(argv$counts_gct)

  lookup <- as.data.frame(vroom::vroom(argv$sample_participant_lookup, delim = "\t",
                                       show_col_types = FALSE, progress = FALSE))
  rownames(lookup) <- as.character(lookup[[1]])

  # Restrict to samples shared between the expression matrix and the lookup.
  shared <- intersect(colnames(tpm), rownames(lookup))
  tpm    <- tpm[, shared, drop = FALSE]
  counts <- counts[, shared, drop = FALSE]

  # Expression mask on the UNnormalized matrices.
  ns <- ncol(tpm)
  mask <- (rowSums(tpm >= argv$tpm_threshold) >= argv$sample_frac_threshold * ns) &
          (rowSums(counts >= argv$count_threshold) >= argv$sample_frac_threshold * ns)

  # Normalize (TMM computed on all genes; filter to `mask` afterwards).
  method_l <- tolower(method)
  if (method_l == "tmm_cpm_edger") {
    lib_size <- colSums(counts) * calcNormFactors(counts)
    norm <- sweep(counts, 2, lib_size, "/") * 1e6
  } else if (method_l == "tmm_cpm_voom") {
    eff <- colSums(counts) * calcNormFactors(counts) + 1
    norm <- log2(sweep(counts + 0.5, 2, eff, "/") * 1e6)
  } else if (method_l == "qn") {
    norm <- preprocessCore::normalize.quantiles(counts)
    dimnames(norm) <- dimnames(counts)
  } else {
    stop(sprintf("Unknown normalization method: %s", method))
  }
  norm <- norm[mask, , drop = FALSE]
  if (qnorm) norm <- inverse_normal_transform(norm)

  # Strip Ensembl versions; guard against collisions.
  rownames(norm) <- strip_gene_version(rownames(norm))
  if (anyDuplicated(rownames(norm))) {
    dups <- unique(rownames(norm)[duplicated(rownames(norm))])
    stop(sprintf("Duplicate gene IDs after stripping Ensembl versions: %s",
                 paste(head(dups, 10), collapse = ", ")))
  }

  # Map sample columns -> participant IDs.
  colnames(norm) <- lookup[colnames(norm), "participant_id"]

  # Join to the TSS BED and assemble the output.
  gene_bed <- gtf_to_tss_bed(argv$annotation_gtf)
  norm_df <- tibble(gene_id = rownames(norm)) %>%
    bind_cols(as_tibble(norm))
  bed <- inner_join(gene_bed, norm_df, by = "gene_id")
  if (nrow(bed) == 0)
    stop("No expression genes overlapped the annotation GTF after gene ID normalization.")

  bed <- bed %>%
    rename(`#chr` = chr) %>%
    relocate(`#chr`, start, end, gene_id) %>%
    mutate(.rank = chrom_rank(`#chr`)) %>%
    arrange(.rank, `#chr`, start, end, gene_id) %>%
    select(-.rank)

  no_qnorm <- if (qnorm) "" else ".no_qnorm"
  out_gz <- file.path(argv$cwd, sprintf("%s.%s%s.expression.bed.gz", bname, method, no_qnorm))
  plain <- sub("\\.gz$", "", out_gz)
  vroom::vroom_write(bed, plain, delim = "\t", quote = "none", escape = "none")
  Rsamtools::bgzip(plain, dest = out_gz, overwrite = TRUE); file.remove(plain)
  Rsamtools::indexTabix(out_gz, format = "bed")
  message(sprintf("Output: %s", out_gz))
}

if (argv$step == "normalize") {
  run_normalize(argv)
} else {
  stop(sprintf("Unknown step: %s", argv$step))
}
