#!/usr/bin/env Rscript
# ============================================================
# eoo_enrichment.R
# Standalone CLI worker for eoo_enrichment.ipynb [enrichment].
#
# Leave-one-chromosome-out (block-jackknife) odds-ratio and enrichment of a set of
# significant variants within each baseline annotation column.
#
# Conventions: read/write with vroom, manipulate with dplyr (no data.table).
# CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
})

parser <- arg_parser("EOO enrichment: LOCO block-jackknife OR / enrichment")
parser <- add_argument(parser, "--significant-variants-path", type = "character",
                       help = "Significant variants (chr,pos) as tsv[.gz] or rds")
parser <- add_argument(parser, "--baseline-anno-path", type = "character",
                       help = "Baseline annotation matrix (tsv[.gz] or rds)")
parser <- add_argument(parser, "--annotations-start", type = "numeric", default = 7,
                       help = "1-based column index where annotation columns begin")
parser <- add_argument(parser, "--output", type = "character",
                       help = "Output enrichment RDS")
argv <- parse_args(parser)
if (is.na(argv$significant_variants_path) || is.na(argv$baseline_anno_path) || is.na(argv$output))
  stop("--significant-variants-path, --baseline-anno-path and --output are required")

read_input_file <- function(file_path) {
  if (grepl("\\.rds$", file_path)) return(readRDS(file_path))
  base_ext <- tools::file_ext(sub("\\.gz$", "", file_path))
  if (base_ext %in% c("txt", "tsv"))
    return(vroom::vroom(file_path, show_col_types = FALSE))
  stop(paste("Unsupported file format:", file_path))
}

calculate_OR_enrichment <- function(set1, set2, target_set = NULL) {
  if (is.null(target_set)) target_set <- unique(union(set1, set2))
  A <- intersect(set1, target_set)
  B <- intersect(set2, target_set)
  AB <- intersect(A, B)
  AnoB <- setdiff(A, AB)
  noAB <- setdiff(B, AB)
  noAnoB <- setdiff(target_set, c(A, B))
  if (length(noAB) == 0 || length(AnoB) == 0) {
    OR <- Enrichment <- 1
  } else {
    OR <- (length(AB) / length(AnoB)) * (length(noAnoB) / length(noAB))
    Enrichment <- (length(AB) / length(B)) / (length(A) / length(target_set))
  }
  list(OR = OR, Enrichment = Enrichment)
}

your_anno <- read_input_file(argv$significant_variants_path)
baseline  <- read_input_file(argv$baseline_anno_path)

if ("chr" %in% colnames(baseline) && !("CHR" %in% colnames(baseline)))
  names(baseline)[names(baseline) == "chr"] <- "CHR"
if ("pos" %in% colnames(baseline) && !("BP" %in% colnames(baseline)))
  names(baseline)[names(baseline) == "pos"] <- "BP"
if (!is.numeric(baseline$CHR)) baseline$CHR <- as.numeric(gsub("chr", "", baseline$CHR))

# Build "chr<chr>:<pos>" keys (vectorized; the .ipynb looped per row).
your_chr_has_prefix <- !(is.numeric(your_anno$chr) || all(grepl("^[0-9]+$", your_anno$chr)))
your_anno <- if (your_chr_has_prefix)
  paste0(your_anno$chr, ":", your_anno$pos) else paste0("chr", your_anno$chr, ":", your_anno$pos)

baseline <- baseline %>%
  mutate(chr_bp = paste0("chr", CHR, ":", BP)) %>%
  relocate(chr_bp, .before = 1)

annotations_start <- as.integer(argv$annotations_start)
annotations <- colnames(baseline)[annotations_start:ncol(baseline)]
message(sprintf("Number of annotations: %d", length(annotations)))

OR_blockJacknife <- Enrichment_blockJacknife <-
  matrix(NA, nrow = 22, ncol = length(annotations))
colnames(OR_blockJacknife) <- colnames(Enrichment_blockJacknife) <- annotations

for (i.chr in 1:22) {
  pp <- which(baseline$CHR == i.chr)
  baseline.jk <- if (length(pp)) baseline[-pp, ] else baseline
  target_set <- baseline.jk$chr_bp
  for (i in seq_along(annotations)) {
    baseline.tmp <- baseline$chr_bp[which(baseline[[annotations[i]]] == 1)]
    res <- calculate_OR_enrichment(baseline.tmp, your_anno, target_set = target_set)
    OR_blockJacknife[i.chr, i] <- res$OR
    Enrichment_blockJacknife[i.chr, i] <- res$Enrichment
  }
}

OR              <- colMeans(log2(OR_blockJacknife), na.rm = TRUE)
Enrichment      <- colMeans(Enrichment_blockJacknife, na.rm = TRUE)
Enrichment_log2 <- colMeans(log2(Enrichment_blockJacknife), na.rm = TRUE)

n <- length(annotations)
OR_sd <- Enrichment_sd <- OR_sd_log2 <- Enrichment_sd_log2 <- numeric(n)
for (j in seq_len(n)) {                                  # jackknife SE: var * (K-1)^2/K
  OR_sd[j]              <- sqrt(var(OR_blockJacknife[, j], na.rm = TRUE) * 21^2 / 22)
  Enrichment_sd[j]      <- sqrt(var(Enrichment_blockJacknife[, j], na.rm = TRUE) * 21^2 / 22)
  OR_sd_log2[j]         <- sqrt(var(log2(OR_blockJacknife[, j]), na.rm = TRUE) * 21^2 / 22)
  Enrichment_sd_log2[j] <- sqrt(var(log2(Enrichment_blockJacknife[, j]), na.rm = TRUE) * 21^2 / 22)
}

Enrichment_z_scores      <- Enrichment / Enrichment_sd
Enrichment_p_values      <- pchisq(Enrichment_z_scores^2, 1, lower.tail = FALSE)
Enrichment_log2_z_scores <- Enrichment_log2 / Enrichment_sd_log2
Enrichment_log2_p_values <- pchisq(Enrichment_log2_z_scores^2, 1, lower.tail = FALSE)

summary_df <- data.frame(
  Annotation = annotations, OR = 2^OR, OR_SE = OR_sd, OR_log2 = OR, OR_SE_log2 = OR_sd_log2,
  Enrichment = Enrichment, Enrichment_SE = Enrichment_sd,
  Enrichment_log2 = Enrichment_log2, Enrichment_SE_log2 = Enrichment_sd_log2,
  Enrichment_Z_score = Enrichment_z_scores, Enrichment_P_value = Enrichment_p_values,
  Enrichment_log2_z_scores = Enrichment_log2_z_scores,
  Enrichment_log2_p_values = Enrichment_log2_p_values)

results <- list(
  summary = summary_df, OR_blockJacknife = OR_blockJacknife,
  Enrichment_blockJacknife = Enrichment_blockJacknife, OR = OR, Enrichment = Enrichment,
  OR_sd = OR_sd, Enrichment_sd = Enrichment_sd, Enrichment_Z_scores = Enrichment_z_scores,
  Enrichment_P_values = Enrichment_p_values, annotations = annotations)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(results, argv$output, compress = "xz")
vroom::vroom_write(summary_df, sub("\\.rds$", "_summary.tsv.gz", argv$output), delim = "\t")
message(sprintf("Written: %s", argv$output))
