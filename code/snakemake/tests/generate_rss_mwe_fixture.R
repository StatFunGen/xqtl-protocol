#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(pecotmr)
})

args <- commandArgs(trailingOnly = TRUE)
repo <- if (length(args) >= 1) {
  normalizePath(args[[1]], mustWork = TRUE)
} else {
  normalizePath(getwd(), mustWork = TRUE)
}

fixture_dir <- file.path(repo, "code", "snakemake", "tests", "data", "rss_analysis_mwe")
ld_dir <- file.path(fixture_dir, "ld_reference")
rss_dir <- file.path(fixture_dir, "rss_analysis")
dir.create(ld_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(rss_dir, recursive = TRUE, showWarnings = FALSE)

ext_candidates <- c(
  system.file("extdata", package = "pecotmr"),
  Sys.getenv("PECOTMR_EXTDATA", unset = ""),
  file.path(
    dirname(repo), "xqtl-renovated", "mwe_data", ".pixi", "envs",
    "r-base", "lib", "R", "library", "pecotmr", "extdata"
  )
)
ext_candidates <- ext_candidates[nzchar(ext_candidates)]
ext_candidates <- ext_candidates[
  file.exists(file.path(ext_candidates, "toy_ref.bim")) &
    file.exists(file.path(ext_candidates, "toy_summary.txt.gz"))
]
if (length(ext_candidates) == 0) {
  stop("Could not find pecotmr extdata with toy_ref.bim and toy_summary.txt.gz")
}
ext <- normalizePath(ext_candidates[[1]], mustWork = TRUE)

region <- "chr22:49355984-50799822"
region_start <- 49355984L
region_end <- 50799822L

X <- pecotmr:::load_genotype_region(file.path(ext, "toy_ref"), region)
X <- pecotmr:::filter_X(X, missing_rate_thresh = 1, maf_thresh = 0)
ld_cor <- cor(X, use = "pairwise.complete.obs")
ld_cor[is.na(ld_cor)] <- 0
diag(ld_cor) <- 1

bim <- fread(file.path(ext, "toy_ref.bim"), header = FALSE)
setnames(bim, c("chr", "SNP", "gd", "pos", "A1", "A2"))
bim <- bim[match(colnames(ld_cor), SNP)]
if (any(is.na(bim$SNP))) {
  stop("Failed to match all LD matrix columns back to toy_ref.bim")
}
bim[, variant_id := paste(chr, pos, A2, A1, sep = ":")]

cor_path <- file.path(
  ld_dir, "protocol_example.chr22_49355984_50799822.cor.xz")
con <- xzfile(cor_path, "wt")
write.table(ld_cor, con, row.names = FALSE, col.names = FALSE, quote = FALSE)
close(con)

bim_out <- bim[, .(chr, variant_id, gd, pos, A1, A2)]
fwrite(
  bim_out,
  file.path(ld_dir, "protocol_example.chr22_49355984_50799822.bim"),
  sep = "\t", col.names = FALSE
)

ld_meta <- data.table(
  chrom = "chr22",
  start = region_start,
  end = region_end,
  path = paste(
    "protocol_example.chr22_49355984_50799822.cor.xz",
    "protocol_example.chr22_49355984_50799822.bim",
    sep = ","
  )
)
setnames(ld_meta, "chrom", "#chr")
fwrite(
  ld_meta,
  file.path(ld_dir, "protocol_example.ld_meta_file.tsv"),
  sep = "\t"
)

sumstats <- fread(file.path(ext, "toy_summary.txt.gz"))
sumstats <- merge(
  sumstats,
  bim[, .(SNP, chr, pos, variant_id)],
  by = "SNP",
  sort = FALSE
)
if (nrow(sumstats) != nrow(bim)) {
  stop("Expected ", nrow(bim), " sumstats rows, got ", nrow(sumstats))
}
sumstats <- sumstats[match(bim$SNP, SNP)]
sumstats[, chrom := chr]
sumstats[, z := beta / se]
sumstats <- sumstats[, .(
  chrom, pos, A1, A2, beta, se, z, p, freq, N, SNP
)]

plain_sumstats <- file.path(
  rss_dir, "protocol_example.gwas_sumstats.chr22.tsv")
fwrite(sumstats, plain_sumstats, sep = "\t")

bgzip_status <- system2("bgzip", c("-f", plain_sumstats))
if (!identical(bgzip_status, 0L)) {
  stop("bgzip failed with status ", bgzip_status)
}
tabix_status <- system2(
  "tabix",
  c(
    "-f", "-S", "1", "-s", "1", "-b", "2", "-e", "2",
    paste0(plain_sumstats, ".gz")
  )
)
if (!identical(tabix_status, 0L)) {
  stop("tabix failed with status ", tabix_status)
}

writeLines(
  c(
    "chrom:chrom",
    "pos:pos",
    "A1:A1",
    "A2:A2",
    "beta:beta",
    "se:se",
    "z:z",
    "n_sample:N"
  ),
  file.path(rss_dir, "protocol_example.column_mapping.txt")
)

gwas_meta <- data.table(
  study_id = "pecotmr_toy",
  chrom = 22L,
  file_path = "protocol_example.gwas_sumstats.chr22.tsv.gz",
  column_mapping_file = "protocol_example.column_mapping.txt",
  n_sample = 10000L,
  n_case = 0L,
  n_control = 0L
)
fwrite(
  gwas_meta,
  file.path(rss_dir, "protocol_example.gwas_meta_data.tsv"),
  sep = "\t"
)

cat(
  "generated fixture rows=", nrow(sumstats),
  " matrix_dim=", paste(dim(ld_cor), collapse = "x"),
  "\n",
  sep = ""
)
