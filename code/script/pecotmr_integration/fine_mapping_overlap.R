#!/usr/bin/env Rscript
# fine_mapping_overlap.R
#
# Overlap the top loci of a QTL FineMappingResult against one or more GWAS
# FineMappingResults, matching variants with pecotmr's allele-aware
# overlapTopLoci() (strand/ref-alt-swap aware; GWAS effects sign-flipped to the
# QTL orientation). Writes one concatenated TSV keyed on the QTL variant with
# every other column prefixed qtl_ / gwas_.
#
# Inputs:
#   --qtl <RDS>                 One QtlFineMappingResult RDS.
#   --gwas <RDS> [<RDS> ...]    One or more GwasFineMappingResult RDS.
#   --signal-cutoff             PIP cutoff forwarded to overlapTopLoci for both
#                               sides. Default 0.025.
#   --output <TSV>              Output TSV path (gzipped if it ends in .gz).

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

parser <- arg_parser("Overlap QTL x GWAS top loci (allele-aware)")
parser <- add_argument(parser, "--qtl",
                       help = "QtlFineMappingResult RDS path(s) (combined if >1)",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--gwas",
                       help = "One or more GwasFineMappingResult RDS paths",
                       type = "character", nargs = Inf)
parser <- add_argument(parser, "--signal-cutoff",
                       help = "PIP cutoff for both sides", type = "numeric",
                       default = 0.025)
parser <- add_argument(parser, "--output",
                       help = "Output TSV path (the overlap table)", type = "character")
# Optional enloc meta-update outputs: emit the QTL region's meta row with a
# `block_top_loci` column listing the GWAS blocks (region_ids) that share >=1
# top-locus variant -- the column the coloc manifest (enloc_manifest.R --mode
# coloc) pairs on.
parser <- add_argument(parser, "--qtl-meta",
                       help = "Optional xQTL meta TSV to stamp block_top_loci onto",
                       type = "character", default = NA)
parser <- add_argument(parser, "--region-id",
                       help = "region_id row of --qtl-meta to update", type = "character", default = NA)
parser <- add_argument(parser, "--meta-output",
                       help = "Output path for the updated meta row", type = "character", default = NA)
argv <- parse_args(parser)

# A QTL region may span several per-study FMRs (its meta original_data); combine
# them so overlapTopLoci sees the region's full top-loci set.
qtlPaths <- as.character(argv$qtl)
qtls <- lapply(qtlPaths, readRDS)
if (!all(vapply(qtls, function(x) methods::is(x, "QtlFineMappingResult"), logical(1))))
  stop("every --qtl input must be a QtlFineMappingResult RDS.")
qtl <- if (length(qtls) == 1L) qtls[[1L]] else Reduce(combineFineMappingResults, qtls)
gwasPaths <- as.character(argv$gwas)
if (length(gwasPaths) == 0L)
  stop("--gwas requires at least one GwasFineMappingResult RDS.")

pieces <- list()
blockTopLoci <- character(0)   # GWAS blocks (region_ids) with >=1 shared variant
for (gp in gwasPaths) {
  gwas <- readRDS(gp)
  if (!methods::is(gwas, "GwasFineMappingResult")) {
    warning("Skipping non-GwasFineMappingResult input: ", gp)
    next
  }
  ov <- tryCatch(
    as.data.frame(overlapTopLoci(qtl, gwas, signalCutoff = argv$signal_cutoff)),
    error = function(e) {
      message(basename(gp), ": ", conditionMessage(e))
      NULL
    })
  if (is.null(ov) || nrow(ov) == 0L) next
  ov$gwas_source <- basename(gp)
  pieces[[length(pieces) + 1L]] <- ov
  blockTopLoci <- c(blockTopLoci, unique(as.character(gwas$region_id)))
}
blockTopLoci <- unique(blockTopLoci)

if (length(pieces) == 0L) {
  message("No overlapping variants; writing an empty TSV with the key header.")
  out <- data.frame(variant_id = character(0), chrom = character(0),
                    pos = integer(0), A1 = character(0), A2 = character(0),
                    gwas_source = character(0), stringsAsFactors = FALSE)
} else {
  all_cols <- unique(unlist(lapply(pieces, names)))
  for (k in seq_along(pieces)) {
    miss <- setdiff(all_cols, names(pieces[[k]]))
    for (m in miss) pieces[[k]][[m]] <- NA
    pieces[[k]] <- pieces[[k]][, all_cols, drop = FALSE]
  }
  out <- do.call(rbind, pieces)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
conn <- if (grepl("\\.gz$", argv$output)) gzfile(argv$output, "w") else file(argv$output, "w")
write.table(out, file = conn, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
close(conn)
cat(sprintf("Wrote %d overlapping rows (%d GWAS input(s)) to %s\n",
            nrow(out), length(gwasPaths), argv$output))

# Stamp block_top_loci onto the QTL region's meta row (for the coloc manifest).
if (!is.na(argv$meta_output) && !is.na(argv$qtl_meta)) {
  meta <- read.delim(argv$qtl_meta, check.names = FALSE, stringsAsFactors = FALSE)
  ridCol <- if ("region_id" %in% names(meta)) "region_id" else names(meta)[[4L]]
  if (!is.na(argv$region_id))
    meta <- meta[as.character(meta[[ridCol]]) == argv$region_id, , drop = FALSE]
  meta$block_top_loci <- if (length(blockTopLoci) > 0L)
    paste(blockTopLoci, collapse = ",") else NA
  dir.create(dirname(argv$meta_output), showWarnings = FALSE, recursive = TRUE)
  write.table(meta, file = argv$meta_output, sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
  cat(sprintf("Stamped block_top_loci=%s onto %s\n",
              if (length(blockTopLoci) > 0L) paste(blockTopLoci, collapse = ",") else "NA",
              argv$meta_output))
}
