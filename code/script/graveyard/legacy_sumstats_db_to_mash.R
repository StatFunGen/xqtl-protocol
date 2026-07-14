#!/usr/bin/env Rscript
# legacy_sumstats_db_to_mash.R
#
# Convert the legacy `protocol_example.sumstats_db.rds` MASH fixture
# (nested list keyed by condition -> region -> {variant_names, sumstats})
# into the inputs the new `pecotmr_integration/mash_preprocessing.ipynb`
# expects:
#
#   * per-condition tensorqtl-style TSVs (one per chromosome, gzipped),
#     filename embeds the chrom so the notebook's `_match_paths_for_chrom`
#     helper finds them.
#   * one `<output-dir>/{condition}.sum_list.txt` per condition listing
#     its per-chrom TSV paths (one line per chrom).
#   * a `<output-dir>/regions.tsv` file with columns `chr start end gene_id`
#     (one row per region present in the fixture).
#   * a `<output-dir>/ld_sketch.rds` GenotypeHandle built from a supplied
#     PLINK2 prefix.
#
# Inputs:
#   --legacy-rds <RDS>     Path to protocol_example.sumstats_db.rds.
#   --plink2-prefix <PFX>  PLINK2 pgen/pvar/psam prefix (no extension)
#                          covering the variants in the fixture; the
#                          new pipeline needs a GenotypeHandle to feed
#                          mashPipeline()'s QC gate.
#   --output-dir <DIR>     Output directory (created if absent).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Convert the legacy sumstats_db.rds fixture to new-pipeline MASH inputs")
parser <- add_argument(parser, "--legacy-rds",
                       help = "Path to the legacy protocol_example.sumstats_db.rds",
                       type = "character")
parser <- add_argument(parser, "--plink2-prefix",
                       help = "PLINK2 pgen/pvar/psam prefix for the LD sketch",
                       type = "character")
parser <- add_argument(parser, "--output-dir",
                       help = "Output directory",
                       type = "character")
argv <- parse_args(parser)

if (!file.exists(argv$legacy_rds))
  stop("--legacy-rds file not found: ", argv$legacy_rds)
for (ext in c(".pgen", ".pvar", ".psam")) {
  if (!file.exists(paste0(argv$plink2_prefix, ext)))
    stop("PLINK2 file missing: ", paste0(argv$plink2_prefix, ext))
}
dir.create(argv$output_dir, showWarnings = FALSE, recursive = TRUE)

db <- readRDS(argv$legacy_rds)
if (!is.list(db) || length(db) == 0L)
  stop("Legacy RDS is not a non-empty list (got ", class(db)[[1L]],
       ", length ", length(db), ").")

conditions <- names(db)
if (any(nchar(conditions) == 0L))
  stop("Every condition in the legacy RDS must be named.")

# ----- Pass 1: enumerate regions present across all conditions ----------
parseRegion <- function(s) {
  m <- regmatches(s, regexec("^([^:]+):([0-9]+)-([0-9]+)$", s))[[1L]]
  if (length(m) != 4L)
    stop("Cannot parse legacy region label '", s,
         "'; expected 'chr:start-end'.")
  list(chr = m[[2L]], start = as.integer(m[[3L]]),
       end = as.integer(m[[4L]]))
}

regionSet <- list()
for (cond in conditions) {
  for (rid in names(db[[cond]])) {
    if (is.null(regionSet[[rid]])) regionSet[[rid]] <- parseRegion(rid)
  }
}
if (length(regionSet) == 0L)
  stop("No regions found across any condition in the legacy RDS.")

# Pre-compute per-condition variant_id chromosomes so we can group
# per-chrom TSV writes. Variant IDs are of the form 'chr22:15528227:A:G'.
chromOf <- function(vids) {
  sub("^([^:]+):.*", "\\1", as.character(vids))
}

# ----- Pass 2: write per-condition per-chrom TSVs -----------------------
sumListPaths <- character(length(conditions))
names(sumListPaths) <- conditions
for (cond in conditions) {
  # Concatenate every region of this condition into a single data frame
  # (variant_id, z), then split by chromosome and write one TSV per chrom.
  perRegion <- db[[cond]]
  pieces <- list()
  for (rid in names(perRegion)) {
    rd <- perRegion[[rid]]
    if (is.null(rd$variant_names) || is.null(rd$sumstats) ||
        is.null(rd$sumstats$z)) {
      warning("Skipping region '", rid, "' in condition '", cond,
              "': missing variant_names or sumstats$z.")
      next
    }
    vids <- as.character(rd$variant_names)
    z    <- as.numeric(rd$sumstats$z)
    if (length(vids) != length(z))
      stop("Length mismatch in condition '", cond,
           "', region '", rid, "': ", length(vids), " variants vs ",
           length(z), " z-scores.")
    pieces[[length(pieces) + 1L]] <- data.frame(
      variant_id = vids, z = z, stringsAsFactors = FALSE)
  }
  if (length(pieces) == 0L) {
    warning("Condition '", cond, "' produced no rows; skipping.")
    sumListPaths[[cond]] <- ""
    next
  }
  df <- do.call(rbind, pieces)
  df <- df[!duplicated(df$variant_id), , drop = FALSE]

  chroms <- chromOf(df$variant_id)
  uchroms <- sort(unique(chroms))
  perCondPaths <- character(length(uchroms))
  names(perCondPaths) <- uchroms
  for (chr in uchroms) {
    sub <- df[chroms == chr, , drop = FALSE]
    # Embed the bare chromosome number in the filename (matching the
    # notebook's `.{chr_number}.` matching convention).
    chrNum <- sub("^chr", "", chr)
    outPath <- file.path(argv$output_dir,
                         sprintf("%s.%s.tsv.gz", cond, chrNum))
    gz <- gzfile(outPath, "w")
    write.table(sub, file = gz, sep = "\t", quote = FALSE,
                row.names = FALSE, na = "")
    close(gz)
    perCondPaths[[chr]] <- outPath
  }
  sumListPath <- file.path(argv$output_dir,
                           sprintf("%s.sum_list.txt", cond))
  writeLines(perCondPaths, sumListPath)
  sumListPaths[[cond]] <- sumListPath
  cat(sprintf("Wrote %d TSV(s) for condition '%s' (%d total variants)\n",
              length(uchroms), cond, nrow(df)))
}

# ----- Region TSV -------------------------------------------------------
regionsTsv <- file.path(argv$output_dir, "regions.tsv")
regionRows <- do.call(rbind, lapply(names(regionSet), function(rid) {
  r <- regionSet[[rid]]
  # Synthesise a stable gene_id from the region label so the notebook can
  # use it as a per-region filename token.
  geneId <- gsub("[:\\-]", "_", rid)
  data.frame(chr = r$chr, start = r$start, end = r$end,
             gene_id = geneId, stringsAsFactors = FALSE)
}))
write.table(regionRows, file = regionsTsv, sep = "\t",
            quote = FALSE, row.names = FALSE)
cat(sprintf("Wrote regions TSV (%d rows) to %s\n",
            nrow(regionRows), regionsTsv))

# ----- LD-sketch GenotypeHandle ----------------------------------------
ldHandle <- GenotypeHandle(plink2Prefix = argv$plink2_prefix)
ldOut <- file.path(argv$output_dir, "ld_sketch.rds")
saveRDS(ldHandle, ldOut)
cat(sprintf("Wrote GenotypeHandle (path '%s') to %s\n",
            argv$plink2_prefix, ldOut))

# Print a tiny manifest summary so the user knows what to pass downstream.
cat("\n--- Conversion summary ---\n")
cat("output dir:        ", argv$output_dir, "\n", sep = "")
cat("conditions:        ", paste(conditions, collapse = ", "), "\n", sep = "")
cat("--conditions arg:  ", paste(conditions, collapse = ","), "\n", sep = "")
cat("--sum-files args:  ", paste(unname(sumListPaths), collapse = " "), "\n", sep = "")
cat("--region-file arg: ", regionsTsv, "\n", sep = "")
cat("--ld-sketch arg:   ", ldOut, "\n", sep = "")
