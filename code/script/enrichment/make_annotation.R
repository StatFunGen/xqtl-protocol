#!/usr/bin/env Rscript
# ============================================================
# make_annotation.R
# CLI worker for sldsc_enrichment.ipynb [make_annotation_files_ldscore] — the R
# analysis blocks (Step A: write .annot files; Step D: write .l2.M files). The
# polyfun LD-score computation (ldsc.py / compute_ldscores.py) stays external.
#
# Faithful port of the notebook R (data.table -> vroom/dplyr; arrow kept for
# parquet). .annot output written directly as .annot.gz.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("sLDSC annotation prep (see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "annot | mfiles")
parser <- add_argument(parser, "--targets", type = "character", default = "", help = "[annot] comma-separated target annotation files")
parser <- add_argument(parser, "--reference-anno", type = "character", default = "", help = "[annot] reference .annot(.gz)")
parser <- add_argument(parser, "--bfile-prefix", type = "character", default = "", help = "[annot] plink bfile prefix (for --print-snps normalize)")
parser <- add_argument(parser, "--score-column", type = "numeric", default = 3, help = "[annot] score column index")
parser <- add_argument(parser, "--is-range", flag = TRUE, help = "[annot] target files are chr/start/end ranges")
parser <- add_argument(parser, "--use-print-snps", flag = TRUE, help = "[annot/mfiles] snplist mode (normalize to .bim panel)")
parser <- add_argument(parser, "--emit-single", flag = TRUE, help = "write single-target outputs")
parser <- add_argument(parser, "--emit-joint", flag = TRUE, help = "write joint output")
parser <- add_argument(parser, "--annotation-name", type = "character", default = "", help = "output name prefix")
parser <- add_argument(parser, "--cwd", type = "character", default = "", help = "output directory")
parser <- add_argument(parser, "--chrom", type = "character", default = "", help = "chromosome tag")
parser <- add_argument(parser, "--n-targets", type = "numeric", default = 1, help = "[mfiles] number of single targets")
parser <- add_argument(parser, "--ldscore-ext", type = "character", default = "l2.ldscore.parquet", help = "[mfiles] ldscore extension")
parser <- add_argument(parser, "--frq-file", type = "character", default = "", help = "[mfiles] .frq file (for .l2.M_5_50)")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# ---------------------------------------------------------------------------
annot_step <- function(argv) {
  suppressPackageStartupMessages(library(vroom))
  clean_chr <- function(x) as.numeric(gsub("^chr", "", x))

  process_range_data <- function(data, chr_value) {
    data$chr <- clean_chr(data$chr)
    data <- data[data$chr == chr_value, ]
    if (nrow(data) == 0) return(NULL)
    expanded <- lapply(seq_len(nrow(data)), function(j) {
      row <- data[j, ]; pos_seq <- seq(row$start, row$end - 1)
      result <- data.frame(chr = rep(row$chr, length(pos_seq)), pos = pos_seq)
      if (ncol(data) > 3) for (col in 4:ncol(data)) result[[names(data)[col]]] <- rep(row[[col]], length(pos_seq))
      result
    })
    unique(do.call(rbind, expanded))
  }

  process_annotation <- function(target_anno, ref_anno, score_column_value) {
    target_anno <- as.data.frame(target_anno); ref_anno <- as.data.frame(ref_anno)
    target_anno$chr <- clean_chr(target_anno$chr); ref_anno$CHR <- clean_chr(ref_anno$CHR)
    anno_scores <- rep(0, nrow(ref_anno))
    match_pos <- match(target_anno$pos, ref_anno$BP)
    valid_pos <- as.numeric(na.omit(match_pos))
    if (score_column_value <= ncol(target_anno)) {
      anno_scores[valid_pos] <- target_anno[[score_column_value]][!is.na(match_pos)]
    } else {
      anno_scores[valid_pos] <- 1
      print("Warning: score column does not exist; setting scores to 1")
    }
    anno_scores
  }

  read_target_anno <- function(file_path, ref_anno) {
    if (endsWith(file_path, "rds")) return(process_annotation(readRDS(file_path), ref_anno, argv$score_column))
    target_anno <- vroom::vroom(file_path, show_col_types = FALSE, progress = FALSE)
    if (argv$is_range) {
      names(target_anno)[1:3] <- c("chr", "start", "end")
      target_anno <- process_range_data(target_anno, unique(clean_chr(ref_anno$CHR)))
      if (is.null(target_anno)) return(rep(0, nrow(ref_anno)))
    } else {
      names(target_anno)[1:2] <- c("chr", "pos")
    }
    process_annotation(target_anno, ref_anno, argv$score_column)
  }

  normalize_for_ldsc <- function(df) {
    if (!argv$use_print_snps) return(df)
    df <- df[, !names(df) %in% c("A1", "A2", "MAF", "CM"), drop = FALSE]
    annot_cols <- setdiff(names(df), c("CHR", "BP", "SNP"))
    bim <- as.data.frame(vroom::vroom(paste0(argv$bfile_prefix, ".bim"), col_names = c("CHR", "SNP", "CM", "BP", "A1", "A2"),
                                      show_col_types = FALSE, progress = FALSE))
    bim$CHR <- as.character(bim$CHR); df$CHR <- as.character(df$CHR)
    idx <- match(bim$SNP, df$SNP)
    out <- data.frame(CHR = bim$CHR, BP = bim$BP, SNP = bim$SNP, CM = bim$CM, stringsAsFactors = FALSE)
    for (col in annot_cols) { v <- rep(0, nrow(bim)); nn <- !is.na(idx); v[nn] <- df[[col]][idx[nn]]; out[[col]] <- v }
    out
  }

  ref_anno <- as.data.frame(vroom::vroom(argv$reference_anno, show_col_types = FALSE, progress = FALSE))
  if ("ANNOT" %in% colnames(ref_anno)) ref_anno <- ref_anno[, colnames(ref_anno) != "ANNOT", drop = FALSE]
  targets <- strsplit(argv$targets, ",", fixed = TRUE)[[1]]
  N <- length(targets)
  score_list <- lapply(targets, read_target_anno, ref_anno = ref_anno)

  write_annot <- function(df, name) {
    out_gz <- file.path(argv$cwd, name, paste0(name, ".", argv$chrom, ".annot.gz"))
    dir.create(dirname(out_gz), showWarnings = FALSE, recursive = TRUE)
    vroom::vroom_write(df, out_gz, delim = "\t", quote = "none", escape = "none")
  }
  if (argv$emit_single) for (i in seq_len(N)) {
    out <- ref_anno; out$ANNOT <- score_list[[i]]
    write_annot(normalize_for_ldsc(out), paste0(argv$annotation_name, "_single_", i))
  }
  if (argv$emit_joint) {
    joint <- ref_anno
    for (i in seq_len(N)) joint[[paste0("ANNOT_", i)]] <- score_list[[i]]
    write_annot(normalize_for_ldsc(joint), paste0(argv$annotation_name, "_joint"))
  }
  message(sprintf("annot written for chr %s", argv$chrom))
}

# ---------------------------------------------------------------------------
mfiles_step <- function(argv) {
  suppressPackageStartupMessages({ library(vroom); library(dplyr) })
  has_frq <- nzchar(argv$frq_file) && file.exists(argv$frq_file)
  frq_dt <- if (has_frq) vroom::vroom(argv$frq_file, show_col_types = FALSE, progress = FALSE)[, c("SNP", "MAF")] else NULL

  write_M_files <- function(annot_path, ldscore_path, m_path) {
    if (argv$use_print_snps && file.exists(m_path) && file.exists(paste0(m_path, "_5_50"))) return(invisible())
    ldscore_dt <- if (endsWith(ldscore_path, ".parquet")) {
      suppressPackageStartupMessages(library(arrow)); arrow::read_parquet(ldscore_path)
    } else as.data.frame(vroom::vroom(ldscore_path, show_col_types = FALSE, progress = FALSE))
    annot_dt <- as.data.frame(vroom::vroom(annot_path, show_col_types = FALSE, progress = FALSE))
    annot_filtered <- annot_dt[annot_dt$SNP %in% ldscore_dt$SNP, ]
    merged <- if (has_frq) dplyr::left_join(annot_filtered, frq_dt, by = "SNP") else annot_filtered
    std_cols <- c("CHR", "SNP", "BP", "CM", "A1", "A2", if (has_frq) "MAF")
    annot_cols <- setdiff(names(merged), std_cols)
    if (length(annot_cols) == 0L) { merged$ANNOT <- 1L; annot_cols <- "ANNOT" }
    M <- vapply(annot_cols, function(c) sum(merged[[c]], na.rm = TRUE), numeric(1))
    writeLines(paste(as.numeric(M), collapse = " "), m_path)
    if (has_frq) {
      common <- merged[!is.na(merged$MAF) & merged$MAF > 0.05, ]
      M5 <- vapply(annot_cols, function(c) sum(common[[c]], na.rm = TRUE), numeric(1))
      writeLines(paste(as.numeric(M5), collapse = " "), paste0(m_path, "_5_50"))
    }
  }

  targets <- character(0)
  if (argv$emit_single) for (i in seq_len(as.integer(argv$n_targets))) targets <- c(targets, paste0(argv$annotation_name, "_single_", i))
  if (argv$emit_joint) targets <- c(targets, paste0(argv$annotation_name, "_joint"))
  for (name in targets) {
    write_M_files(file.path(argv$cwd, name, paste0(name, ".", argv$chrom, ".annot.gz")),
                  file.path(argv$cwd, name, paste0(name, ".", argv$chrom, ".", argv$ldscore_ext)),
                  file.path(argv$cwd, name, paste0(name, ".", argv$chrom, ".l2.M")))
  }
  message(sprintf("M files written for chr %s", argv$chrom))
}

# ---------------------------------------------------------------------------
switch(argv$step,
  annot  = annot_step(argv),
  mfiles = mfiles_step(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
