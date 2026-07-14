#!/usr/bin/env Rscript
# twas_manifest_validate.R
#
# Validate the per-gene TWAS manifest used by twas.ipynb and emit a
# canonicalised copy (header has no leading '#'; required columns
# present; gene_id values unique; optional fine_mapping_result_rds
# column normalised). Downstream SoS steps fan out over the canonical
# copy via csv.DictReader without needing any Python parsing logic.
#
# Inputs:
#   --manifest <TSV>  Per-gene manifest with at least the columns
#                      `gene_id`, `twas_weights_rds`, `gwas_sumstats_rds`
#                      (and optionally `fine_mapping_result_rds`).
#                      A leading '#' on the header line is tolerated
#                      and stripped.
#   --output <TSV>    Canonical manifest TSV path.

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Validate + canonicalise the per-gene TWAS manifest")
parser <- add_argument(parser, "--manifest",
                       help = "Per-gene TWAS manifest TSV",
                       type = "character")
parser <- add_argument(parser, "--output",
                       help = "Canonical manifest TSV path",
                       type = "character")
argv <- parse_args(parser)

if (!file.exists(argv$manifest))
  stop("--manifest file not found: ", argv$manifest)

lines <- readLines(argv$manifest)
if (length(lines) == 0L)
  stop("Empty manifest: ", argv$manifest)

# Tolerate (and strip) a leading '#' on the header line.
hdr <- strsplit(sub("^#", "", lines[[1L]]), "\t", fixed = TRUE)[[1L]]
required <- c("gene_id", "twas_weights_rds", "gwas_sumstats_rds")
missing <- setdiff(required, hdr)
if (length(missing) > 0L)
  stop("Manifest missing required column(s): ",
       paste(missing, collapse = ", "),
       " (header was: ", paste(hdr, collapse = ", "), ").")
hasFmr <- "fine_mapping_result_rds" %in% hdr

rows <- list()
for (line in lines[-1L]) {
  if (!nzchar(line)) next
  parts <- strsplit(line, "\t", fixed = TRUE)[[1L]]
  # Pad short rows with empty strings so column alignment stays sane.
  if (length(parts) < length(hdr))
    parts <- c(parts, rep("", length(hdr) - length(parts)))
  row <- setNames(as.list(parts[seq_along(hdr)]), hdr)
  if (!nzchar(row$gene_id)) next
  rows[[length(rows) + 1L]] <- row
}
if (length(rows) == 0L)
  stop("Manifest has no data rows: ", argv$manifest)

geneIds <- vapply(rows, `[[`, character(1), "gene_id")
if (anyDuplicated(geneIds))
  stop("Manifest has duplicate gene_id values: ",
       paste(unique(geneIds[duplicated(geneIds)]), collapse = ", "))

outCols <- c("gene_id", "twas_weights_rds", "gwas_sumstats_rds",
             if (hasFmr) "fine_mapping_result_rds" else NULL)
outDf <- do.call(rbind, lapply(rows, function(r) {
  as.data.frame(setNames(lapply(outCols, function(k) {
    v <- r[[k]]; if (is.null(v)) "" else as.character(v)
  }), outCols), stringsAsFactors = FALSE)
}))

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
write.table(outDf, file = argv$output, sep = "\t", quote = FALSE,
            row.names = FALSE, na = "")
cat(sprintf("Canonicalised manifest (%d rows, %s) to %s\n",
            nrow(outDf),
            if (hasFmr) "with fine_mapping_result_rds"
            else "no fine_mapping_result_rds column", argv$output))
