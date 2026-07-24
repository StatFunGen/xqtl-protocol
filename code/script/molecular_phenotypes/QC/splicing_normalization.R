#!/usr/bin/env Rscript
# ============================================================
# splicing_normalization.R
# Standalone CLI worker for splicing_normalization.ipynb steps.
#
# Steps (selected via --step):
#   prepare_phenotype — leafcutter ratios -> per-chrom .phen_<chr> / .ave /
#                       .qqnorm_<chr> (sorted) + _phenotype_file_list.txt  ([leafcutter_norm_1])
#   qqnorm            — NA/variance filter, row-scale, per-column rank qqnorm ([leafcutter_norm_3],
#                       [psichomics_norm_2])
#   psichomics_bed    — parseSplicingEvent -> BED-ordered PSI table         ([psichomics_norm_1])
#   jointcall_samples — subset a sample lookup to samples present in a junc list ([Jointcall_samples])
#
# Faithful port of the leafcutter prepare_phenotype_table.py logic and the psichomics
# blocks. Numeric transforms reproduce scipy: preprocessing.scale = (x-mean)/pop_sd;
# qqnorm(x) = qnorm((rank_avg(x) - a)/(n + 1 - 2a)), a = 0.5 (n>10) else 3/8.
# (Float string formatting follows R, not Python's shortest-repr; values are identical.)
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("splicing_normalization worker (see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "prepare_phenotype | qqnorm | psichomics_bed | jointcall_samples")
parser <- add_argument(parser, "--input", type = "character", default = "", help = "input file")
parser <- add_argument(parser, "--output", type = "character", default = "", help = "output file")
parser <- add_argument(parser, "--chr-blacklist", type = "character", default = "", help = "[prepare_phenotype] chromosome blacklist file")
parser <- add_argument(parser, "--pseudo-ratio", flag = TRUE, help = "[prepare_phenotype] use 0.01*denom pseudocount instead of 0.5")
parser <- add_argument(parser, "--na-rate", type = "numeric", default = 0.4, help = "[qqnorm] max NA fraction per intron")
parser <- add_argument(parser, "--min-variance", type = "numeric", default = 0.001, help = "[qqnorm] min per-intron sd")
parser <- add_argument(parser, "--mean-impute", flag = TRUE, help = "[qqnorm] mean-impute NAs on kept rows")
parser <- add_argument(parser, "--sample-table", type = "character", default = "", help = "[jointcall_samples] sample lookup")
parser <- add_argument(parser, "--junc-list", type = "character", default = "", help = "[jointcall_samples] junction file list")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# Python str(float)-ish: shortest round-tripping decimal (matches most Python reprs; values equal regardless).
pyf <- function(x) {
  out <- character(length(x))
  na <- is.na(x)
  out[na] <- "NA"
  for (i in which(!na)) {
    for (d in 1:17) { s <- sprintf(paste0("%.", d, "g"), x[i]); if (as.numeric(s) == x[i]) break }
    out[i] <- s
  }
  out
}

# ---------------------------------------------------------------------------
prepare_phenotype <- function(argv) {
  con <- gzfile(argv$input, "rt"); lines <- readLines(con); close(con)
  header <- strsplit(trimws(lines[1]), " ", fixed = TRUE)[[1]][-1]
  blk <- if (nzchar(argv$chr_blacklist) && file_test("-f", argv$chr_blacklist)) readLines(argv$chr_blacklist) else character(0)
  sample_names <- sub(".Aligned.sortedByCoord.out.md", "", header, fixed = TRUE)

  chr_of <- character(0); start_of <- integer(0)
  phen <- character(0); ave <- character(0); qq <- character(0)  # per-intron text rows
  for (ln in lines[-1]) {
    toks <- strsplit(trimws(ln), " ", fixed = TRUE)[[1]]
    chrom <- toks[1]; counts <- toks[-1]
    parts <- strsplit(chrom, ":", fixed = TRUE)[[1]]
    chr_ <- parts[1]; s <- parts[2]; e <- parts[3]
    if (chr_ %in% blk) next
    nd <- do.call(rbind, strsplit(counts, "/", fixed = TRUE))
    num <- as.numeric(nd[, 1]); denom <- as.numeric(nd[, 2])
    pseudo <- if (argv$pseudo_ratio) 0.01 * denom else 0.5
    ratio <- (num + pseudo) / (denom + pseudo)
    ratio[denom < 1] <- NA                                   # denom<1 -> NA
    valstr <- pyf(ratio)
    av <- ratio[!is.na(ratio)]
    phen  <- c(phen,  paste(c(chr_, s, e, chrom, valstr), collapse = "\t"))
    ave   <- c(ave,   paste(chrom, pyf(min(av)), pyf(max(av)), pyf(mean(av))))
    qq    <- c(qq,    paste(c(chr_, s, e, chrom, valstr), collapse = "\t"))
    chr_of <- c(chr_of, chr_); start_of <- c(start_of, as.integer(s))
  }
  chroms <- unique(chr_of)
  phen_hdr <- paste(c("#chr", "start", "end", "ID", header), collapse = "\t")
  qq_hdr   <- paste(c("#Chr", "start", "end", "ID", sample_names), collapse = "\t")
  for (ch in chroms) {
    idx <- which(chr_of == ch)
    writeLines(c(phen_hdr, phen[idx]), paste0(argv$input, ".phen_", ch))
    ord <- idx[order(start_of[idx])]                         # .qqnorm sorted by (chrom, start)
    writeLines(c(qq_hdr, qq[ord]), paste0(argv$input, ".qqnorm_", ch))
  }
  writeLines(ave, paste0(argv$input, ".ave"))
  flist <- c("#chr\t#dir", vapply(chroms, function(ch) sprintf("%s\t%s.qqnorm_%s", ch, argv$input, ch), ""))
  writeLines(flist, paste0(argv$input, "_phenotype_file_list.txt"))
}

# ---------------------------------------------------------------------------
qqnorm_step <- function(argv) {
  suppressPackageStartupMessages(library(vroom))
  df <- vroom::vroom(argv$input, delim = "\t", show_col_types = FALSE)
  info <- df[, 1:4]; V <- as.matrix(df[, -(1:4)])
  nsamp <- ncol(V); na_limit <- nsamp * argv$na_rate
  na_count <- rowSums(is.na(V))
  pop_sd <- function(r) { r <- r[!is.na(r)]; if (length(r) == 0) return(NA_real_); sqrt(mean((r - mean(r))^2)) }
  nanstd <- apply(V, 1, pop_sd)                              # on original rows (before impute)
  drop <- (na_count > na_limit) | (nanstd < argv$min_variance)
  if (argv$mean_impute) V <- t(apply(V, 1, function(r) { r[is.na(r)] <- mean(r, na.rm = TRUE); r }))
  keep <- !drop
  info <- info[keep, , drop = FALSE]; V <- V[keep, , drop = FALSE]

  V <- t(apply(V, 1, function(r) { m <- mean(r); s <- sqrt(mean((r - m)^2)); (r - m) / s }))  # scale rows
  n <- nrow(V); a <- if (n <= 10) 3 / 8 else 0.5
  for (j in seq_len(ncol(V))) {
    x <- V[, j]
    V[, j] <- qnorm((rank(x, ties.method = "average", na.last = "keep") - a) / (n + 1 - 2 * a))
  }
  out <- cbind(as.data.frame(info), as.data.frame(V))
  names(out) <- names(df)
  con <- file(argv$output, "w")
  writeLines(paste(names(out), collapse = "\t"), con)
  for (i in seq_len(nrow(out)))
    writeLines(paste(c(unlist(out[i, 1:4]), pyf(as.numeric(out[i, -(1:4)]))), collapse = "\t"), con)
  close(con)
}

# ---------------------------------------------------------------------------
psichomics_bed <- function(argv) {
  suppressPackageStartupMessages({ library(psichomics); library(vroom) })
  d <- as.data.frame(vroom::vroom(argv$input, delim = "\t", show_col_types = FALSE))
  rn <- d[[1]]; psi <- d[, -1, drop = FALSE]; rownames(psi) <- rn
  ev <- parseSplicingEvent(rownames(psi))
  bed <- data.frame(`#Chr` = ev$chrom, start = ev$start, end = ev$end, ID = rownames(ev),
                    psi, check.names = FALSE)
  bed[["#Chr"]] <- sub("^", "chr", bed[["#Chr"]])
  bed <- bed[order(bed[["#Chr"]], bed$start, bed$end), ]
  vroom::vroom_write(bed, argv$output, delim = "\t", quote = "none", escape = "none")
}

# ---------------------------------------------------------------------------
jointcall_samples <- function(argv) {
  suppressPackageStartupMessages({ library(vroom); library(stringr); library(dplyr) })
  lookup <- vroom::vroom(argv$sample_table, delim = "\t", show_col_types = FALSE)
  lookup$sample_id <- gsub(".final", "", lookup$sample_id, fixed = TRUE)
  junc <- readLines(argv$junc_list)
  sample_ids <- str_split(basename(junc), "[.]", simplify = TRUE)[, 1]
  lookup <- lookup[lookup$sample_id %in% sample_ids, ]
  vroom::vroom_write(lookup, argv$output, delim = "\t", quote = "none", escape = "none")
}

# ---------------------------------------------------------------------------
switch(argv$step,
  prepare_phenotype = prepare_phenotype(argv),
  qqnorm            = qqnorm_step(argv),
  psichomics_bed    = psichomics_bed(argv),
  jointcall_samples = jointcall_samples(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
