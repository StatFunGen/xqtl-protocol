#!/usr/bin/env Rscript
# ============================================================
# phenotype_formatting.R
# Standalone CLI worker for phenotype_formatting.ipynb steps.
#
# Steps (selected via --step):
#   phenotype_by_chrom_1      — extract one chromosome from a tabix BED.gz
#   phenotype_by_chrom_2      — build the per-chrom file + region-list manifests
#   phenotype_by_chrom_gct_1  — extract one chromosome as a coordinate-stripped GCT
#   phenotype_by_region_1     — extract one named region from a tabix BED.gz
#   phenotype_by_region_2     — build the per-region file manifest
#   gct_extract_samples       — filter samples from a GCT by a keep-list
#   phenotype_annotate_by_tad — build a phenotype-per-TAD region list
#   bam_subsetting            — subset a BAM to the requested regions
#
# Conventions: read with vroom, manipulate with dplyr, bgzip/tabix/bam via
# Rsamtools (no data.table). CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(Rsamtools)
  library(GenomicRanges)
  library(dplyr)   # last, so dplyr verbs win over S4Vectors::rename() etc.
})

TABIX_MAX <- 536870911L  # 2^29 - 1: max coordinate a .tbi index can address

parser <- arg_parser("phenotype_formatting worker (see --step)")
parser <- add_argument(parser, "--step", help = "Step to run", type = "character")
parser <- add_argument(parser, "--phenoFile",
                       help = "Input phenotype file (BED.gz/GCT.gz/BAM depending on step)",
                       type = "character")
parser <- add_argument(parser, "--cwd", help = "Output directory", type = "character",
                       default = "output")
parser <- add_argument(parser, "--name",
                       help = "Output name prefix (default: derived from phenoFile)",
                       type = "character", default = "")
parser <- add_argument(parser, "--chrom", help = "[*_chrom_*] Chromosome to extract",
                       type = "character", default = "")
parser <- add_argument(parser, "--region", nargs = Inf,
                       help = "[by_region_1] chr start end name; [bam_subsetting] region strings",
                       type = "character")
parser <- add_argument(parser, "--inputs", nargs = Inf,
                       help = "[*_2] per-shard output paths to collate", type = "character")
parser <- add_argument(parser, "--output", help = "Explicit output path", type = "character",
                       default = "")
parser <- add_argument(parser, "--output-files", help = "[*_2] chrom/region file manifest",
                       type = "character", default = "")
parser <- add_argument(parser, "--output-region-list", help = "[by_chrom_2] region-list output",
                       type = "character", default = "")
parser <- add_argument(parser, "--keep-samples",
                       help = "[gct_extract_samples] File with one sample ID per line to keep",
                       type = "character", default = "")
parser <- add_argument(parser, "--TAD_list",
                       help = "[phenotype_annotate_by_tad] TAD list (#chr start end index)",
                       type = "character", default = "")
parser <- add_argument(parser, "--phenotype-per-tad",
                       help = "[phenotype_annotate_by_tad] Min phenotypes per TAD to keep",
                       type = "numeric", default = 2)
parser <- add_argument(parser, "--numThreads", help = "Number of threads",
                       type = "numeric", default = 1)
argv <- parse_args(parser)

if (is.na(argv$step)) stop("--step is required")
if (!nzchar(argv$name) && !is.na(argv$phenoFile)) {
  argv$name <- sub("\\.gct(\\.gz)?$", "", basename(argv$phenoFile))
}
dir.create(argv$cwd, showWarnings = FALSE, recursive = TRUE)

# --- helpers ---------------------------------------------------------------
need <- function(argv, flags) {
  blank <- function(f) { v <- argv[[f]]
    is.null(v) || all(is.na(v)) || (is.character(v) && length(v) == 1 && !nzchar(v)) }
  miss <- flags[vapply(flags, blank, logical(1))]
  if (length(miss))
    stop(sprintf("Step %s requires: %s", argv$step,
                 paste0("--", gsub("_", "-", miss), collapse = ", ")))
}

read_header_line <- function(f) vroom::vroom_lines(f, n_max = 1)

# Fetch all data rows for a GRanges from a tabix-indexed file (character lines).
tabix_rows <- function(pheno, gr) {
  tbx <- Rsamtools::TabixFile(pheno); open(tbx); on.exit(close(tbx))
  Rsamtools::scanTabix(tbx, param = gr)[[1]]
}

# Write header + data rows as a bgzipped BED, optionally tabix-indexed.
write_bgzip_bed <- function(hdr, rows, output, index = TRUE) {
  dir.create(dirname(output), showWarnings = FALSE, recursive = TRUE)
  plain <- sub("\\.gz$", "", output)
  con <- file(plain, "w"); writeLines(c(hdr, rows), con); close(con)
  Rsamtools::bgzip(plain, dest = output, overwrite = TRUE)
  file.remove(plain)
  if (index) Rsamtools::indexTabix(output, format = "bed")
}

# basename "<name>.<field>.bed.gz" -> its 3rd-from-last dot field (the shard id)
shard_field <- function(paths)
  vapply(paths, function(p) { f <- strsplit(basename(p), ".", fixed = TRUE)[[1]]
                              f[length(f) - 2] }, character(1), USE.NAMES = FALSE)

# --- steps: by chromosome --------------------------------------------------
phenotype_by_chrom_1 <- function(argv) {
  need(argv, c("phenoFile", "chrom", "output"))
  rows <- tabix_rows(argv$phenoFile, GRanges(argv$chrom, IRanges(1, TABIX_MAX)))
  write_bgzip_bed(read_header_line(argv$phenoFile), rows, argv$output, index = TRUE)
}

phenotype_by_chrom_gct_1 <- function(argv) {
  need(argv, c("phenoFile", "chrom", "output"))
  dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
  lines <- c(read_header_line(argv$phenoFile),
             tabix_rows(argv$phenoFile, GRanges(argv$chrom, IRanges(1, TABIX_MAX))))
  # legacy awk `$1=$2=$3=""`: blank the 3 coordinate fields, join with spaces.
  blanked <- vapply(strsplit(lines, "\t", fixed = TRUE), function(f) {
    if (length(f) >= 3) f[1:3] <- ""
    paste(f, collapse = " ")
  }, character(1))
  writeLines(blanked, argv$output)
  message(sprintf("Written: %s", argv$output))
}

phenotype_by_chrom_2 <- function(argv) {
  need(argv, c("phenoFile", "inputs", "output_files", "output_region_list"))
  ids <- sub("^chr", "", shard_field(argv$inputs))
  # Use safe internal names for the dplyr ops; the "#"-prefixed headers the
  # downstream expects are set only at write time (dplyr NSE dislikes "#" names).
  chrom_df <- tibble(id = ids, dir = argv$inputs, chr = paste0("chr", ids))
  pheno <- vroom::vroom(argv$phenoFile, delim = "\t", col_select = 1:4,
                        col_types = cols(.default = "c"), show_col_types = FALSE)
  names(pheno) <- c("chr", "start", "end", "ID")
  pheno <- pheno %>%
    inner_join(chrom_df %>% select(chr, dir), by = "chr") %>%
    rename(path = dir)
  chrom_df <- chrom_df %>% filter(chr %in% unique(pheno$chr))

  files_out <- chrom_df %>% select(id, dir)
  names(files_out) <- c("#id", "#dir")
  region_out <- pheno
  names(region_out)[names(region_out) == "chr"] <- "#chr"
  vroom::vroom_write(files_out, argv$output_files, delim = "\t")
  vroom::vroom_write(region_out, argv$output_region_list, delim = "\t")
  message(sprintf("Written: %s, %s", argv$output_files, argv$output_region_list))
}

# --- steps: by region ------------------------------------------------------
phenotype_by_region_1 <- function(argv) {
  need(argv, c("phenoFile", "region", "output"))
  r <- argv$region  # chr, start, end, name
  rows <- tabix_rows(argv$phenoFile,
                     GRanges(r[1], IRanges(as.integer(r[2]), as.integer(r[3]))))
  write_bgzip_bed(read_header_line(argv$phenoFile), rows, argv$output, index = FALSE)
}

phenotype_by_region_2 <- function(argv) {
  need(argv, c("inputs", "output"))
  vroom::vroom_write(tibble(`#id` = shard_field(argv$inputs), dir = argv$inputs),
                     argv$output, delim = "\t")
  message(sprintf("Written: %s", argv$output))
}

# --- steps: GCT sample filtering -------------------------------------------
gct_extract_samples <- function(argv) {
  need(argv, c("phenoFile", "keep_samples"))
  keep_ids <- trimws(vroom::vroom_lines(argv$keep_samples))
  keep_ids <- keep_ids[nzchar(keep_ids)]
  message(sprintf("Keeping %d samples", length(keep_ids)))

  header_lines <- vroom::vroom_lines(argv$phenoFile, n_max = 2)   # "#1.2", dims
  dat <- vroom::vroom(argv$phenoFile, delim = "\t", skip = 2,
                      col_types = cols(.default = "c"), show_col_types = FALSE)
  fixed_cols  <- colnames(dat)[1:2]                               # Name, Description
  sample_cols <- base::setdiff(colnames(dat), fixed_cols)
  available   <- base::intersect(keep_ids, sample_cols)
  missing     <- base::setdiff(keep_ids, sample_cols)
  if (length(missing) > 0)
    warning(sprintf("%d requested samples not found in GCT: %s",
                    length(missing), paste(head(missing, 5), collapse = ", ")))
  message(sprintf("Retaining %d of %d samples", length(available), length(sample_cols)))

  dat_filt <- dat %>% select(all_of(c(fixed_cols, available)))
  out_file <- if (nzchar(argv$output)) argv$output else
    file.path(argv$cwd, paste0(argv$name, ".filtered.gct.gz"))
  dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

  con <- if (grepl("\\.gz$", out_file)) gzfile(out_file, "w") else file(out_file, "w")
  on.exit(close(con), add = TRUE)
  writeLines(header_lines[1], con)
  writeLines(sprintf("%d\t%d", nrow(dat_filt), length(available)), con)
  writeLines(paste(colnames(dat_filt), collapse = "\t"), con)
  writeLines(do.call(paste, c(as.list(dat_filt), sep = "\t")), con)
  message(sprintf("Written: %s", out_file))
}

# --- steps: TAD annotation -------------------------------------------------
phenotype_annotate_by_tad <- function(argv) {
  need(argv, c("phenoFile", "TAD_list"))
  tad <- vroom::vroom(argv$TAD_list, delim = "\t",
                      col_types = cols(.default = "c"), show_col_types = FALSE) %>%
    mutate(start = as.integer(.data$start), end = as.integer(.data$end))
  id_col <- if ("index" %in% names(tad)) "index" else names(tad)[4]

  gr <- GRanges(tad$`#chr`, IRanges(tad$start, tad$end))
  tbx <- Rsamtools::TabixFile(argv$phenoFile); open(tbx); on.exit(close(tbx))
  # scanTabix errors on a seqname absent from the index; the old `tabix` loop
  # returned no rows, so treat absent-contig regions as count 0.
  in_idx <- as.character(seqnames(gr)) %in% Rsamtools::seqnamesTabix(tbx)
  counts <- integer(length(gr))
  if (any(in_idx)) counts[in_idx] <- lengths(Rsamtools::scanTabix(tbx, param = gr[in_idx]))

  out <- tad %>%
    mutate(n = counts, path = argv$phenoFile) %>%
    filter(.data$n >= argv$phenotype_per_tad) %>%
    transmute(`#chr`, start, end, ID = .data[[id_col]], path)

  out_file <- if (nzchar(argv$output)) argv$output else file.path(
    argv$cwd, paste0(basename(argv$phenoFile), ".", basename(argv$TAD_list), ".",
                     argv$phenotype_per_tad, "_pheno_per_region.region_list"))
  dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
  vroom::vroom_write(out, out_file, delim = "\t")
  message(sprintf("Written: %s (%d of %d TADs kept)", out_file, nrow(out), nrow(tad)))
}

# --- steps: BAM subsetting -------------------------------------------------
bam_subsetting <- function(argv) {
  need(argv, c("phenoFile", "region", "output"))
  targets <- Rsamtools::scanBamHeader(argv$phenoFile)[[1]]$targets  # seqname -> length
  parse_region <- function(s) {
    if (grepl(":", s, fixed = TRUE)) {
      chr <- sub(":.*", "", s); rng <- sub(".*:", "", s)
      GRanges(chr, IRanges(as.integer(sub("-.*", "", rng)), as.integer(sub(".*-", "", rng))))
    } else {
      GRanges(s, IRanges(1L, as.integer(targets[[s]])))
    }
  }
  which <- do.call(c, lapply(argv$region, parse_region))
  dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
  Rsamtools::filterBam(argv$phenoFile, argv$output,
                       param = Rsamtools::ScanBamParam(which = which))
  message(sprintf("Written: %s", argv$output))
}

# ---------------------------------------------------------------------------
switch(argv$step,
  phenotype_by_chrom_1      = phenotype_by_chrom_1(argv),
  phenotype_by_chrom_2      = phenotype_by_chrom_2(argv),
  phenotype_by_chrom_gct_1  = phenotype_by_chrom_gct_1(argv),
  phenotype_by_region_1     = phenotype_by_region_1(argv),
  phenotype_by_region_2     = phenotype_by_region_2(argv),
  gct_extract_samples       = gct_extract_samples(argv),
  phenotype_annotate_by_tad = phenotype_annotate_by_tad(argv),
  bam_subsetting            = bam_subsetting(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
