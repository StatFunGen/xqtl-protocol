#!/usr/bin/env Rscript
# gwas_sumstats_to_tsv.R
#
# Export per-LD-block pecotmr GwasSumStats RDS files (e.g. the QC'd output of the
# RSS GWAS pipeline, <study>.<chr>_<start>_<end>.gwas_sumstats.rds) into one
# bgzipped, tabix-indexed TSV per chromosome, in the layout the twas step reads
# through --gwas_meta_data (#chrom pos A1 A2 variant_id z n_sample), and write the
# matching gwas_meta TSV (study_id chrom file_path sample_size).
#
# A1 is the effect (alt) allele; variant_id is chrom:pos:A2:A1. Blocks with no
# variants after QC are skipped. A variant present in two blocks is kept once: the
# copy from a block listed in --prefer-blocks (e.g. the larger "added" block of an
# overlapping pair), otherwise the measured copy over a RAISS-imputed one (imputed
# variants have no allele frequency), otherwise the copy from the block that
# starts first.
#
# Usage:
#   Rscript gwas_sumstats_to_tsv.R --input study.gwas_sumstats.list --study AD_main \
#     --output-dir out/ --meta out/AD_main.gwas_meta.tsv

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(dplyr)
  library(vroom)
  library(Rsamtools)
})
p <- arg_parser("Per-block GwasSumStats RDS -> per-chromosome GWAS TSV")
p <- add_argument(p, "--input", help = "Text file listing the GwasSumStats RDS files of ONE study, one per line")
p <- add_argument(p, "--study", help = "Study ID written to the gwas_meta file")
p <- add_argument(p, "--output-dir", help = "Directory for <study>.chr<N>.tsv.gz files")
p <- add_argument(p, "--meta", help = "Output gwas_meta TSV")
p <- add_argument(p, "--prefer-blocks", help = "Optional text file of block names (chr<N>_<start>_<end>), one per line, whose copy of a shared variant is kept first", default = "")
a <- parse_args(p)

fs <- trimws(readLines(a$input))
fs <- fs[nzchar(fs)]
prefer <- if (nzchar(a$prefer_blocks)) trimws(readLines(a$prefer_blocks)) else character(0)
dir.create(a$output_dir, showWarnings = FALSE, recursive = TRUE)

x <- bind_rows(lapply(fs, function(f) {
  g <- as.data.frame(readRDS(f))
  if (!nrow(g)) return(NULL)
  tibble(chrom = sub("^chr", "", g$chrom), pos = g$pos, A1 = g$A1, A2 = g$A2,
         z = g$z, n_sample = g$N,
         # RAISS-imputed variants carry no allele frequency
         imputed = is.na(g$af),
         block_start = min(g$pos),
         preferred = sub(".*[.](chr[0-9XY]+_[0-9]+_[0-9]+)[.]gwas_sumstats[.]rds$",
                         "\\1", basename(f)) %in% prefer)
}))
if (!nrow(x)) stop("No variants in any input file.")

# One row per (chrom, pos, A1, A2): order the duplicates by the documented
# preference and keep the first of each group.
x <- x |>
  arrange(as.integer(chrom), pos, !preferred, imputed, block_start)
ndup <- sum(duplicated(x[, c("chrom", "pos", "A1", "A2")]))
x <- x |>
  filter(!duplicated(pick(chrom, pos, A1, A2))) |>
  mutate(variant_id = paste(chrom, pos, A2, A1, sep = ":")) |>
  select(chrom, pos, A1, A2, variant_id, z, n_sample)

message(sprintf(paste0("%s: %d blocks, %d variants (%d variants in two blocks ",
                       "kept once, preferring --prefer-blocks, then the measured ",
                       "over the RAISS-imputed copy, then the first block)"),
                a$study, length(fs), nrow(x), ndup))

# One bgzipped + tabix-indexed TSV per chromosome; the header's leading "#"
# makes tabix treat it as a comment, so no skip= is needed on read.
meta <- bind_rows(lapply(split(x, x$chrom), function(y) {
  plain <- file.path(normalizePath(a$output_dir), sprintf("%s.chr%s.tsv", a$study, y$chrom[1]))
  vroom_write(rename(y, `#chrom` = chrom), plain, delim = "\t", quote = "none",
              progress = FALSE)
  bgzip(plain, dest = paste0(plain, ".gz"), overwrite = TRUE)
  file.remove(plain)
  indexTabix(paste0(plain, ".gz"), seq = 1L, start = 2L, end = 2L, comment = "#")
  tibble(study_id = a$study, chrom = y$chrom[1], file_path = paste0(plain, ".gz"),
         sample_size = round(median(y$n_sample)))
}))
vroom_write(arrange(meta, as.integer(chrom)), a$meta, delim = "\t", quote = "none",
            progress = FALSE)
