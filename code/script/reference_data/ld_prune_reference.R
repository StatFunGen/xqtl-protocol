#!/usr/bin/env Rscript
# ld_prune_reference.R — LD_pruning_2 worker for ld_prune_reference.ipynb.
#
# Concatenates the per-block PLINK ``.prune.in`` variant lists (uncorrelated
# variants kept by ``plink2 --indep-pairwise``) and reshapes each variant ID into
# chrom / pos / alt / ref columns plus a canonical ``chrom:pos:alt:ref`` id.
#
# PLINK writes IDs as ``chrom:pos_ALT_REF``; replacing ``_`` with ``:`` and
# splitting on ``:`` yields the four fields (position 3 = alt, position 4 = ref,
# matching the original notebook cell).

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
  library(tidyr)
})

p <- arg_parser("Concatenate per-block LD-pruned variant lists and reshape variant IDs")
p <- add_argument(p, "--input", nargs = Inf, help = "per-block .prune.in files")
p <- add_argument(p, "--output", help = "output TSV (chrom, pos, alt, ref, variant_id)")
argv <- parse_args(p)
argv$input <- argv$input[!is.na(argv$input)]

# each .prune.in is a headerless single-column list of variant IDs
variants <- vroom(argv$input, col_names = "id", col_types = "c", delim = "\t")

out <- variants %>%
  mutate(id = gsub("_", ":", .data$id)) %>%
  separate(.data$id, into = c("chrom", "pos", "alt", "ref"), sep = ":", remove = TRUE) %>%
  mutate(variant_id = paste(.data$chrom, .data$pos, .data$alt, .data$ref, sep = ":"))

vroom_write(out, argv$output, delim = "\t", quote = "none")
