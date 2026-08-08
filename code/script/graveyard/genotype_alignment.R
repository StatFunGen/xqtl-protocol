#!/usr/bin/env Rscript
# ============================================================
# genotype_alignment.R
# Mirrors: code/SoS/misc/genotype_alignment.ipynb  ([genotype_alignment] step)
#
# Aligns the .bim variant tables of one or more cohorts (for a single chromosome)
# to a reference cohort: variants whose alleles are the swap of the reference's are
# flipped to the reference orientation, then the de-duplicated union of variants
# (chr, pos, alt, ref) is written as a bgzip+tabix'd 4-column table. The FIRST
# positional .bim file is the reference.
#
# Usage:
#   Rscript genotype_alignment.R --output <chr.aligned.bim.gz> <ref.bim> [<other.bim> ...]
# ============================================================
suppressPackageStartupMessages({
  library(optparse)
  library(vroom)
  library(dplyr)
})

parsed <- parse_args(
  OptionParser(option_list = list(
    make_option("--output", type = "character", default = NULL,
                help = "Output aligned .bim.gz path (bgzip + tabix)"))),
  positional_arguments = TRUE)
opt <- parsed$options
bim_files <- parsed$args
if (is.null(opt$output)) stop("--output is required")
if (length(bim_files) < 1)
  stop("at least one .bim file (positional) is required; the first is the reference")

read_bim <- function(f) {
  vroom(f, delim = "\t", col_names = c("chr", "id", "cm", "pos", "alt", "ref"),
        col_types = cols(chr = col_character(), pos = col_double(),
                         alt = col_character(), ref = col_character(),
                         .default = col_character())) %>%
    select(chr, pos, alt, ref)
}

# Order-independent allele key so a variant and its ref/alt swap collide.
allele_key <- function(d) paste(d$chr, d$pos, pmin(d$alt, d$ref), pmax(d$alt, d$ref))

# Flip variants of `other` whose alleles are the reference's swapped, then return the
# de-duplicated union of both, all in the reference allele orientation.
align_to_reference <- function(ref, other) {
  ref_keys <- ref %>%
    transmute(key = allele_key(.), ref_alt = alt, ref_ref = ref) %>%
    distinct(key, .keep_all = TRUE)
  other_aligned <- other %>%
    mutate(key = allele_key(.)) %>%
    left_join(ref_keys, by = "key") %>%
    mutate(flip    = !is.na(ref_alt) & alt == ref_ref & ref == ref_alt,
           new_alt = if_else(flip, ref, alt),
           new_ref = if_else(flip, alt, ref)) %>%
    transmute(chr, pos, alt = new_alt, ref = new_ref)
  bind_rows(ref, other_aligned) %>% distinct(chr, pos, alt, ref)
}

reference <- read_bim(bim_files[1])
for (f in bim_files[-1]) {
  message("Aligning ", f)
  reference <- align_to_reference(reference, read_bim(f))
}
reference <- reference %>% arrange(pos)

dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
plain <- sub("\\.gz$", "", opt$output)
vroom_write(reference, plain, delim = "\t", col_names = FALSE)
Rsamtools::bgzip(plain, dest = opt$output, overwrite = TRUE)
unlink(plain)
Rsamtools::indexTabix(opt$output, seq = 1L, start = 2L, end = 2L)
message("Wrote ", nrow(reference), " variants to ", opt$output)
