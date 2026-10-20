#!/usr/bin/env Rscript
# gwas_sumstats_to_tsv.R
#
# Export per-LD-block pecotmr GwasSumStats RDS files (e.g. the QC'd output of the
# RSS GWAS pipeline, <study>.<chr>_<start>_<end>.gwas_sumstats.rds) into one
# bgzipped, tabix-indexed TSV per chromosome, in the layout the twas step reads
# through --gwas_meta_data (#chrom pos A1 A2 variant_id z n_sample), and write the
# matching gwas_meta TSV (study_id chrom file_path sample_size).
# A1 is the effect (alt) allele; variant_id is chrom:pos:A2:A1. Blocks with no
# variants after QC are skipped. A variant present in two blocks is kept once: the
# copy from a block listed in --prefer-blocks (e.g. the larger "added" block of
# an overlapping pair), otherwise
# the measured copy over a RAISS-imputed one (imputed variants have no allele
# frequency), otherwise the copy from the block that starts first. Needs bgzip and tabix on PATH.
#
# Usage:
#   Rscript gwas_sumstats_to_tsv.R --input study.gwas_sumstats.list --study AD_main \
#     --output-dir out/ --meta out/AD_main.gwas_meta.tsv

suppressPackageStartupMessages({ library(argparser); library(pecotmr); library(data.table) })
p <- arg_parser("Per-block GwasSumStats RDS -> per-chromosome GWAS TSV")
p <- add_argument(p, "--input", help = "Text file listing the GwasSumStats RDS files of ONE study, one per line")
p <- add_argument(p, "--study", help = "Study ID written to the gwas_meta file")
p <- add_argument(p, "--output-dir", help = "Directory for <study>.chr<N>.tsv.gz files")
p <- add_argument(p, "--meta", help = "Output gwas_meta TSV")
p <- add_argument(p, "--prefer-blocks", help = "Optional text file of block names (chr<N>_<start>_<end>), one per line, whose copy of a shared variant is kept first", default = "")
a <- parse_args(p)

fs <- trimws(readLines(a$input)); fs <- fs[nzchar(fs)]
prefer <- if (nzchar(a$prefer_blocks)) trimws(readLines(a$prefer_blocks)) else character(0)
dir.create(a$output_dir, showWarnings = FALSE, recursive = TRUE)
x <- rbindlist(lapply(fs, function(f) {
  g <- as.data.frame(readRDS(f))
  if (!nrow(g)) return(NULL)
  data.table(chrom = sub("^chr", "", g$chrom), pos = g$pos, A1 = g$A1, A2 = g$A2, z = g$z, n_sample = g$N, imputed = is.na(g$af), block_start = min(g$pos),
             preferred = sub(".*[.](chr[0-9XY]+_[0-9]+_[0-9]+)[.]gwas_sumstats[.]rds$", "\\1", basename(f)) %in% prefer)
}))
if (!nrow(x)) stop("No variants in any input file.")
x <- x[order(as.integer(chrom), pos, !preferred, imputed, block_start)]
ndup <- sum(duplicated(x[, .(chrom, pos, A1, A2)]))
x <- x[!duplicated(x[, .(chrom, pos, A1, A2)])][, c("preferred", "imputed", "block_start") := NULL]
x[, variant_id := paste(chrom, pos, A2, A1, sep = ":")]
setcolorder(x, c("chrom", "pos", "A1", "A2", "variant_id", "z", "n_sample"))
message(sprintf("%s: %d blocks, %d variants (%d variants in two blocks kept once, preferring --prefer-blocks, then the measured over the RAISS-imputed copy, then the first block)", a$study, length(fs), nrow(x), ndup))
meta <- rbindlist(lapply(split(x, by = "chrom"), function(y) {
  f <- file.path(normalizePath(a$output_dir), sprintf("%s.chr%s.tsv", a$study, y$chrom[1]))
  setnames(y, "chrom", "#chrom"); fwrite(y, f, sep = "\t")
  if (system2("bgzip", c("-f", f)) != 0L) stop("bgzip failed: ", f)
  if (system2("tabix", c("-f", "-s1", "-b2", "-e2", paste0(f, ".gz"))) != 0L) stop("tabix failed: ", f)
  data.table(study_id = a$study, chrom = y[["#chrom"]][1], file_path = paste0(f, ".gz"), sample_size = round(median(y$n_sample)))
}))
fwrite(meta[order(as.integer(chrom))], a$meta, sep = "\t")
