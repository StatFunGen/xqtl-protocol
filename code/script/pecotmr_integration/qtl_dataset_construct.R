#!/usr/bin/env Rscript
# qtl_dataset_construct.R
#
# Build one pecotmr::QtlDataset from a single study's data and serialize
# to RDS. The resulting RDS becomes the upstream dependency for every
# per-gene fineMappingPipeline / twasWeightsPipeline / colocboostPipeline
# task; gene-level parallelization happens against this single object.
#
# Inputs (per-context phenotype + phenotype-covariate files use
# CONTEXT=PATH pairs joined with commas, e.g.
# --phenotype DLPFC=foo.bed.gz,AC=bar.bed.gz):
#   --study                Study identifier
#   --genotype-prefix      PLINK2 pgen/pvar/psam prefix
#   --phenotype            CONTEXT=PATH list of BED.GZ phenotype files
#   --phenotype-covariates CONTEXT=PATH list of per-context molecular-trait
#                          PC TSVs (samples as rows, PCs as columns —
#                          same shape as the genotype-PC TSV).
#                          Optional; default none.
#   --genotype-covariates  TSV of genotype-derived covariates (e.g. ancestry
#                          PCs) applied uniformly across all contexts
#                          (samples as rows, PCs as columns). Optional.
#   --maf-cutoff           Pass-through to QtlDataset(mafCutoff = ...)
#   --xvar-cutoff          Pass-through to QtlDataset(xvarCutoff = ...)
#   --output               Output RDS path

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
                       help = "PLINK2 pgen/pvar/psam prefix",
                       type = "character")
parser <- add_argument(parser, "--phenotype",
                       help = "Comma-joined CONTEXT=PATH phenotype BED files",
                       type = "character")
parser <- add_argument(parser, "--phenotype-covariates",
                       help = "Comma-joined CONTEXT=PATH PC TSV files",
                       type = "character", default = "")
parser <- add_argument(parser, "--genotype-covariates",
                       help = "TSV of uniformly-applied genotype PCs",
                       type = "character", default = "")
parser <- add_argument(parser, "--maf-cutoff",
                       help = "Pass-through MAF cutoff for QtlDataset()",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--xvar-cutoff",
                       help = "Pass-through variance cutoff for QtlDataset()",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# ----- CONTEXT=PATH list parser ---------------------------------------------
parse_kv <- function(s) {
  if (!nzchar(s)) return(list())
  out <- list()
  for (p in strsplit(s, ",", fixed = TRUE)[[1L]]) {
    kv <- strsplit(p, "=", fixed = TRUE)[[1L]]
    if (length(kv) != 2L) stop("Expected CONTEXT=PATH, got: ", p)
    out[[trimws(kv[[1L]])]] <- trimws(kv[[2L]])
  }
  out
}

phenotype_files <- parse_kv(argv$phenotype)
pheno_cov_files <- parse_kv(argv$phenotype_covariates)
contexts <- names(phenotype_files)
if (length(contexts) == 0L) stop("--phenotype must list at least one context")

# ----- File readers ---------------------------------------------------------
read_pheno_bed <- function(path) {
  read.table(gzfile(path), header = TRUE, sep = "\t",
             stringsAsFactors = FALSE, check.names = FALSE,
             comment.char = "")
}

# Both --phenotype-covariates and --genotype-covariates files share this
# shape: TSV with a sample-id column followed by one column per PC. The
# returned matrix has samples as rows and PCs as columns.
read_pcs_tsv <- function(path) {
  as.matrix(read.table(path, header = TRUE, sep = "\t", row.names = 1,
                       check.names = FALSE, comment.char = ""))
}

build_se <- function(bed_path, pcov_path) {
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
    pcov <- read_pcs_tsv(pcov_path)
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
    build_se(phenotype_files[[cx]],
             if (!is.null(pheno_cov_files[[cx]])) pheno_cov_files[[cx]] else "")),
  contexts)

# ----- Genotype handle (PLINK2) + uniform genotype covariates ---------------
geno_handle <- GenotypeHandle(plink2Prefix = argv$genotype_prefix)

geno_cov_path <- argv$genotype_covariates
has_geno_cov <- nzchar(geno_cov_path) && geno_cov_path != "." &&
                file.exists(geno_cov_path)
genoCov <- if (has_geno_cov) {
  read_pcs_tsv(geno_cov_path)
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
cat(sprintf("Wrote QtlDataset for study '%s' (%d contexts) to %s\n",
            argv$study, length(phenotypes), argv$output))
