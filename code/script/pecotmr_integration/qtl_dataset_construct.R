#!/usr/bin/env Rscript
# qtl_dataset_construct.R
#
# Build one pecotmr::QtlDataset from a single study's data and serialize
# to RDS. The resulting RDS is the upstream dependency for every per-gene
# fineMappingPipeline / twasWeightsPipeline / colocboostPipeline task;
# gene-level parallelization happens against this single object.
#
# Inputs:
#   --study                Study identifier (length-1 character).
#   --genotype-prefix      PLINK1 bed/bim/fam prefix (no extension).
#   --phenotype-manifest   Path to a TSV manifest. Required columns:
#                            ID       trait identifier
#                            path     bgzipped BED phenotype file
#                            cond     context name
#                          Optional column:
#                            cov_path per-context covariate TSV
#                          One row per (region, context). Phenotype /
#                          covariate paths may be absolute or relative to
#                          the manifest's own directory. Multiple rows can
#                          share the same context; per-context the
#                          phenotype/cov path must agree.
#   --genotype-covariates  Optional TSV of genotype-derived covariates
#                          (e.g. ancestry PCs) applied uniformly across all
#                          contexts. Same shape as the per-context covariate
#                          files.
#   --transpose-covariates When set, transposes every covariate TSV
#                          (phenotype + genotype) before treating it as
#                          samples-as-rows. Use this for QTLtools-format
#                          inputs where rows are covariate names and
#                          columns are samples.
#   --maf-cutoff           Pass-through MAF cutoff for QtlDataset().
#   --xvar-cutoff          Pass-through variance cutoff for QtlDataset().
#   --output               Output RDS path.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(SummarizedExperiment)
  library(GenomicRanges)
  library(S4Vectors)
})

parser <- arg_parser("Build a pecotmr QtlDataset for one study and save to RDS")
parser <- add_argument(parser, "--study",
                       help = "Study identifier", type = "character")
parser <- add_argument(parser, "--genotype-prefix",
                       help = "PLINK1 bed/bim/fam prefix", type = "character")
parser <- add_argument(parser, "--phenotype-manifest",
                       help = "TSV manifest path (ID, path, cond, cov_path)",
                       type = "character")
parser <- add_argument(parser, "--genotype-covariates",
                       help = "TSV of uniformly-applied genotype PCs",
                       type = "character", default = "")
parser <- add_argument(parser, "--transpose-covariates",
                       help = "Transpose covariate TSVs (QTLtools format)",
                       flag = TRUE)
parser <- add_argument(parser, "--maf-cutoff",
                       help = "QtlDataset() MAF cutoff",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--xvar-cutoff",
                       help = "QtlDataset() variance cutoff",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# ----- File readers ---------------------------------------------------------
read_pheno_bed <- function(path) {
  read.table(gzfile(path), header = TRUE, sep = "\t",
             stringsAsFactors = FALSE, check.names = FALSE,
             comment.char = "")
}

# Read a covariate TSV. Default expects samples-as-rows + PC columns.
# With transpose = TRUE, expects QTLtools-format input (covariates as rows,
# samples as columns) and transposes to samples-as-rows.
read_pcs_tsv <- function(path, transpose = FALSE) {
  raw <- read.table(path, header = TRUE, sep = "\t", row.names = 1,
                    check.names = FALSE, comment.char = "")
  m <- as.matrix(raw)
  if (transpose) m <- t(m)
  m
}

# ----- Manifest ingestion ---------------------------------------------------
manifest <- read.table(argv$phenotype_manifest, header = TRUE, sep = "\t",
                       stringsAsFactors = FALSE, check.names = FALSE,
                       comment.char = "")
required <- c("ID", "path", "cond")
missingCols <- setdiff(required, names(manifest))
if (length(missingCols) > 0L) {
  stop("--phenotype-manifest missing required column(s): ",
       paste(missingCols, collapse = ", "))
}
has_cov_col <- "cov_path" %in% names(manifest)

# Resolve manifest-relative paths against the manifest's own directory.
manifest_dir <- dirname(normalizePath(argv$phenotype_manifest))
resolve_path <- function(p) {
  if (is.na(p) || !nzchar(p)) return("")
  if (startsWith(p, "/")) return(p)
  file.path(manifest_dir, p)
}

# Collapse to one (phenotype, cov_path) pair per context. Multiple rows per
# context (different traits) must agree on the file paths.
contexts <- unique(manifest$cond)
phenotype_files <- list()
pheno_cov_files <- list()
for (cx in contexts) {
  sub <- manifest[manifest$cond == cx, , drop = FALSE]
  paths_here <- unique(sub$path)
  if (length(paths_here) > 1L) {
    stop(sprintf(
      "Context '%s' references multiple phenotype paths: %s.",
      cx, paste(paths_here, collapse = ", ")))
  }
  phenotype_files[[cx]] <- paths_here[[1L]]
  if (has_cov_col) {
    covs_here <- unique(sub$cov_path[!is.na(sub$cov_path) &
                                      nzchar(sub$cov_path)])
    if (length(covs_here) > 1L) {
      stop(sprintf(
        "Context '%s' references multiple cov_path values: %s.",
        cx, paste(covs_here, collapse = ", ")))
    }
    pheno_cov_files[[cx]] <- if (length(covs_here) == 1L) covs_here[[1L]]
                              else ""
  } else {
    pheno_cov_files[[cx]] <- ""
  }
}

# ----- Per-context SummarizedExperiment builder -----------------------------
build_se <- function(bed_path, pcov_path, transpose_cov) {
  bed <- read_pheno_bed(bed_path)
  chr_col   <- intersect(c("#chr", "chrom", "chr"),    names(bed))[1L]
  start_col <- intersect(c("start", "Start"),          names(bed))[1L]
  end_col   <- intersect(c("end", "End"),              names(bed))[1L]
  gene_col  <- intersect(c("gene_id", "ID",
                           "phenotype_id"),            names(bed))[1L]
  if (any(is.na(c(chr_col, start_col, end_col, gene_col))))
    stop("Missing one of chrom/start/end/gene_id columns in: ", bed_path)

  meta <- bed[, c(chr_col, start_col, end_col, gene_col)]
  names(meta) <- c("chrom", "start", "end", "gene_id")
  sample_cols <- setdiff(names(bed),
    c("#chr", "chrom", "chr", "start", "Start", "end", "End",
      "gene_id", "ID", "phenotype_id", "strand", "Strand"))

  expr <- as.matrix(bed[, sample_cols, drop = FALSE])
  storage.mode(expr) <- "double"
  rownames(expr) <- meta$gene_id

  rr <- GRanges(seqnames = meta$chrom,
                ranges = IRanges(start = meta$start + 1L, end = meta$end))
  names(rr) <- meta$gene_id

  cd <- if (nzchar(pcov_path)) {
    pcov <- read_pcs_tsv(pcov_path, transpose = transpose_cov)
    common <- intersect(rownames(pcov), colnames(expr))
    if (length(common) == 0L)
      stop("No shared samples between phenotype and phenotype-covariate: ",
           bed_path)
    expr <- expr[, common, drop = FALSE]
    DataFrame(pcov[common, , drop = FALSE], row.names = common)
  } else {
    DataFrame(row.names = colnames(expr))
  }
  SummarizedExperiment(assays   = list(expression = expr),
                       rowRanges = rr,
                       colData   = cd)
}

phenotypes <- setNames(
  lapply(contexts, function(cx)
    build_se(resolve_path(phenotype_files[[cx]]),
             resolve_path(pheno_cov_files[[cx]]),
             argv$transpose_covariates)),
  contexts)

# ----- Genotype handle (PLINK1) + uniform genotype covariates ---------------
geno_handle <- GenotypeHandle(plink1Prefix = argv$genotype_prefix)

geno_cov_path <- argv$genotype_covariates
has_geno_cov <- nzchar(geno_cov_path) && geno_cov_path != "." &&
                file.exists(geno_cov_path)
genoCov <- if (has_geno_cov) {
  read_pcs_tsv(geno_cov_path, transpose = argv$transpose_covariates)
} else {
  matrix(numeric(0), nrow = 0, ncol = 0)
}

# ----- Construct + save ------------------------------------------------------
qd <- QtlDataset(
  study              = argv$study,
  genotypes          = geno_handle,
  phenotypes         = phenotypes,
  genotypeCovariates = genoCov,
  mafCutoff          = argv$maf_cutoff,
  xvarCutoff         = argv$xvar_cutoff)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(qd, argv$output)
cat(sprintf("Wrote QtlDataset for study '%s' (%d contexts: %s) to %s\n",
            argv$study, length(phenotypes),
            paste(names(phenotypes), collapse = ", "),
            argv$output))
