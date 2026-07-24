#!/usr/bin/env Rscript
# ============================================================
# RNA_calling.R
# R-side workers for RNA_calling.ipynb. Replaces the pandas merges that used to
# live in RNA_calling.py (rnaseqc_merge / star_align_3 / rsem_call_2) with
# vroom/dplyr, alongside the existing Picard QC aggregation.
#
# Steps (--step):
#   aggregate_picard_qc — per-sample Picard alignment/RNA/duplicate metrics -> TSV
#   rnaseqc_merge       — per-sample RNA-SeQC GCTs -> gene x sample matrices + metrics
#   star_align_3        — STAR outputs -> bam_file_list manifest (filename derivation)
#   rsem_call_2         — per-sample RSEM results -> 7 merged matrices + .cnt metrics
#
# The three merge steps are faithful ports of RNA_calling.py: numeric columns are
# carried through verbatim (numerically identical to pandas; the float repr is
# cosmetic), label columns and the manifest are byte-identical.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(dplyr)
})

p <- arg_parser("RNA_calling R workers (see --step)")
p <- add_argument(p, "--step", type = "character",
                  help = "aggregate_picard_qc | rnaseqc_merge | star_align_3 | rsem_call_2")
# aggregate_picard_qc
p <- add_argument(p, "--input-dir", type = "character", default = "",
                  help = "[aggregate_picard_qc] directory of per-sample Picard metrics")
p <- add_argument(p, "--output", type = "character", default = "",
                  help = "output path (aggregate_picard_qc / star_align_3)")
p <- add_argument(p, "--is-paired-end", type = "numeric", default = 1,
                  help = "[aggregate_picard_qc] 1 paired-end (2 rows), 0 single-end (1 row)")
p <- add_argument(p, "--wasp", flag = TRUE,
                  help = "[aggregate_picard_qc] WASP mode (duplicate-metrics filename pattern)")
# rnaseqc_merge / rsem_call_2
p <- add_argument(p, "--cwd", type = "character", default = "",
                  help = "[rnaseqc_merge/rsem_call_2] output directory")
p <- add_argument(p, "--name", type = "character", default = "",
                  help = "[rnaseqc_merge/rsem_call_2] output prefix")
p <- add_argument(p, "--input", type = "character", nargs = Inf,
                  help = "interleaved input files for the selected step")
# star_align_3
p <- add_argument(p, "--sample-id", type = "character", nargs = Inf,
                  help = "[star_align_3] per-sample IDs")
p <- add_argument(p, "--strand", type = "character", nargs = Inf,
                  help = "[star_align_3] per-sample strand")
p <- add_argument(p, "--var-vcf-file", type = "character", default = "",
                  help = "[star_align_3] non-empty toggles the WASP transcriptome-BAM suffix")
# detect_strand / fastp_manifest / ribosomal_intervals
p <- add_argument(p, "--sample-list", type = "character", default = "",
                  help = "[fastp_manifest] sample sheet TSV to rewrite with trimmed fq paths")
p <- add_argument(p, "--gtf", type = "character", default = "",
                  help = "[ribosomal_intervals] gene GTF to scan for rRNA transcripts")
p <- add_argument(p, "--bam", type = "character", default = "",
                  help = "[ribosomal_intervals] BAM whose header seeds the interval list")
p <- add_argument(p, "--numThreads", type = "numeric", default = 8, help = "unused")
argv <- parse_args(p)

# nargs=Inf args are NA when not supplied for the selected step
for (k in c("input", "sample_id", "strand")) argv[[k]] <- argv[[k]][!is.na(argv[[k]])]

is_paired   <- argv$is_paired_end
source_dir  <- argv$input_dir
wasp_suffix <- if (isTRUE(argv$wasp)) "_wasp" else "_nowasp"
qc_suffix   <- if (isTRUE(argv$wasp)) "_qc"   else "_noqc"

# ── Picard readers (unchanged) ───────────────────────────────────────────────

readPicard.alignment_summary_metrics <- function(source_path) {
  stopifnot(length(source_path) == 1, file.exists(source_path))
  is_dir <- file.info(source_path)$isdir

  if (is_dir) {
    files   <- system(paste("find -L", source_path,
                            "-name '*.alignment_summary_metrics'"), intern = TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".alignment_summary_metrics", "", basename(files), fixed = TRUE)
  } else {
    files   <- source_path
    samples <- gsub(".alignment_summary_metrics", "", basename(files), fixed = TRUE)
  }

  metrics <- list()
  for (i in seq_along(files)) {
    m <- read.table(files[i], header = TRUE, sep = "\t", comment.char = "#",
                    stringsAsFactors = FALSE, nrows = is_paired + 1)
    metrics[[i]] <- data.frame(
      Sample               = samples[i],
      File                 = files[i],
      PF_READS             = sum(m$PF_READS[1:2]),
      PF_READS_ALIGNED     = sum(m$PF_READS_ALIGNED[1:2]),
      PCT_PF_READS_ALIGNED = sum(m$PF_READS_ALIGNED[1:2]) / sum(m$PF_READS[1:2]),
      stringsAsFactors     = FALSE
    )
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  metrics
}


readPicard.rna_metrics <- function(source_path) {
  stopifnot(length(source_path) == 1, file.exists(source_path))
  is_dir <- file.info(source_path)$isdir

  if (is_dir) {
    files   <- system(paste("find -L", source_path, "-name '*.rna_metrics'"), intern = TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(".rna_metrics", "", basename(files), fixed = TRUE)
  } else {
    files   <- source_path
    samples <- gsub(".rna_metrics", "", basename(files), fixed = TRUE)
  }

  metrics <- list()
  for (i in seq_along(files)) {
    m <- read.table(files[i], header = TRUE, sep = "\t", comment.char = "#",
                    stringsAsFactors = FALSE, nrows = 1)
    metrics[[i]] <- data.frame(
      Sample                    = samples[i],
      File                      = files[i],
      PCT_RIBOSOMAL_BASES       = m$PCT_RIBOSOMAL_BASES,
      PCT_CODING_BASES          = m$PCT_CODING_BASES,
      PCT_UTR_BASES             = m$PCT_UTR_BASES,
      PCT_INTRONIC_BASES        = m$PCT_INTRONIC_BASES,
      PCT_INTERGENIC_BASES      = m$PCT_INTERGENIC_BASES,
      PCT_MRNA_BASES            = m$PCT_MRNA_BASES,
      PCT_USABLE_BASES          = m$PCT_USABLE_BASES,
      MEDIAN_CV_COVERAGE        = m$MEDIAN_CV_COVERAGE,
      MEDIAN_5PRIME_BIAS        = m$MEDIAN_5PRIME_BIAS,
      MEDIAN_3PRIME_BIAS        = m$MEDIAN_3PRIME_BIAS,
      MEDIAN_5PRIME_TO_3PRIME_BIAS = m$MEDIAN_5PRIME_TO_3PRIME_BIAS,
      stringsAsFactors          = FALSE
    )
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  metrics
}


readPicard.duplicate_metrics <- function(source_path, wasp_sfx, qc_sfx) {
  stopifnot(length(source_path) == 1, file.exists(source_path))

  pattern   <- paste0("*.Aligned.sortedByCoord.out", wasp_sfx, qc_sfx, ".md.metrics")
  substitute_str <- paste0(".Aligned.sortedByCoord.out", wasp_sfx, qc_sfx, ".md.metrics")
  is_dir    <- file.info(source_path)$isdir

  if (is_dir) {
    files   <- system(paste("find -L", source_path, "-name", pattern), intern = TRUE)
    stopifnot(length(files) > 0)
    samples <- gsub(substitute_str, "", basename(files), fixed = TRUE)
  } else {
    files   <- source_path
    samples <- gsub(substitute_str, "", basename(files), fixed = TRUE)
  }

  metrics <- list()
  for (i in seq_along(files)) {
    m <- read.table(files[i], header = TRUE, sep = "\t", comment.char = "#",
                    stringsAsFactors = FALSE, nrows = 1)
    metrics[[i]] <- data.frame(
      Sample                   = samples[i],
      File                     = files[i],
      PERCENT_DUPLICATION      = m$PERCENT_DUPLICATION,
      ESTIMATED_LIBRARY_SIZE   = m$ESTIMATED_LIBRARY_SIZE,
      stringsAsFactors         = FALSE
    )
  }
  metrics <- do.call(rbind, metrics)
  row.names(metrics) <- metrics$Sample
  metrics
}


readPicard <- function(source_path, wasp_sfx, qc_sfx) {
  metrics_aln <- readPicard.alignment_summary_metrics(source_path)
  metrics_rna <- readPicard.rna_metrics(source_path)
  metrics_dup <- readPicard.duplicate_metrics(source_path, wasp_sfx, qc_sfx)

  stopifnot(
    all(row.names(metrics_aln) %in% row.names(metrics_rna)),
    all(row.names(metrics_rna) %in% row.names(metrics_dup)),
    all(row.names(metrics_dup) %in% row.names(metrics_aln))
  )

  metrics_aln$File <- NULL
  metrics_rna$File <- NULL
  metrics_dup$File <- NULL
  metrics_rna$Sample <- NULL
  metrics_dup$Sample <- NULL

  metrics <- cbind(metrics_aln, metrics_rna[row.names(metrics_aln), ])
  metrics <- cbind(metrics,     metrics_dup[row.names(metrics_aln), ])
  metrics
}

run_aggregate_picard_qc <- function() {
  alignment_files <- system(
    paste("find -L", shQuote(source_dir), "-name '*.alignment_summary_metrics'"),
    intern = TRUE
  )
  if (length(alignment_files) == 0) {
    write.table(data.frame(), file = argv$output,
                col.names = TRUE, row.names = FALSE, quote = FALSE, sep = "\t")
  } else {
    picard_metrics <- readPicard(source_dir, wasp_suffix, qc_suffix)
    write.table(picard_metrics, file = argv$output,
                col.names = TRUE, row.names = FALSE, quote = FALSE, sep = "\t")
  }
}

# ── Shared helpers for the pandas-merge ports ────────────────────────────────

# read a delimited table with every column as character (preserve exact tokens)
read_chr <- function(path, delim = "\t") {
  as.data.frame(vroom::vroom(path, delim = delim, col_types = vroom::cols(.default = "c"),
                             show_col_types = FALSE, progress = FALSE),
                check.names = FALSE)
}

# ── Step: rnaseqc_merge ──────────────────────────────────────────────────────

merge_gct <- function(gct_paths) {
  cols <- lapply(gct_paths, function(gp) {
    parts <- strsplit(basename(gp), ".", fixed = TRUE)[[1]]
    sample_name <- paste(head(parts, -4), collapse = ".")
    # value column carried as character (numerically identical to pandas; avoids
    # the whole-number float-repr drift, e.g. 0 vs 0.0)
    d <- as.data.frame(vroom::vroom(gp, delim = "\t", skip = 2,
                                    col_types = vroom::cols(.default = "c"),
                                    show_col_types = FALSE, progress = FALSE), check.names = FALSE)
    setNames(tibble::tibble(as.character(d[["Name"]]), as.character(d[[3]])), c("gene_ID", sample_name))
  })
  m <- Reduce(function(a, b) full_join(a, b, by = "gene_ID"), cols)
  m[order(m$gene_ID, method = "radix"), , drop = FALSE]   # pandas outer-merge-on-index sorts the index
}

run_rnaseqc_merge <- function() {
  prefix <- file.path(argv$cwd, argv$name)
  inp <- argv$input; n <- length(inp)
  tpm     <- inp[seq(1, n, 4)]
  gc_c    <- inp[seq(2, n, 4)]
  ec      <- inp[seq(3, n, 4)]
  metrics <- inp[seq(4, n, 4)]

  for (spec in list(list(tpm, ".rnaseqc.gene_tpm.gct.gz"),
                    list(gc_c, ".rnaseqc.gene_readsCount.gct.gz"),
                    list(ec,  ".rnaseqc.exon_readsCount.gct.gz"))) {
    merged <- merge_gct(spec[[1]])
    vroom::vroom_write(merged, paste0(prefix, spec[[2]]), delim = "\t",
                       quote = "none", escape = "none")
  }

  # metrics: mirror the python detection on the headerless first file — a 2-row
  # layout is header+values (read with header); a 2-column layout is the RNA-SeQC
  # v2 name<TAB>value transposed layout.
  read_noheader <- function(f) as.data.frame(vroom::vroom(f, delim = "\t", col_names = FALSE,
                     col_types = vroom::cols(.default = "c"), show_col_types = FALSE, progress = FALSE))
  df0 <- read_noheader(metrics[1])
  if (nrow(df0) == 2) {
    dfs <- lapply(metrics, read_chr)
  } else if (ncol(df0) == 2) {
    dfs <- lapply(metrics, function(f) {
      d <- read_noheader(f)
      out <- as.data.frame(as.list(d[[2]]), stringsAsFactors = FALSE, check.names = FALSE)
      colnames(out) <- d[[1]]; out
    })
  } else {
    stop(sprintf("Unrecognized RNA-SeQC metrics format (shape %d x %d).", nrow(df0), ncol(df0)))
  }
  metrics_df <- bind_rows(dfs)
  vroom::vroom_write(metrics_df, paste0(prefix, ".rnaseqc.metrics.tsv"), delim = "\t",
                     quote = "none", escape = "none")
}

# ── Step: star_align_3 ───────────────────────────────────────────────────────

strip_extensions <- function(path_like, count) {
  v <- basename(path_like)
  for (i in seq_len(count)) {
    if (!grepl(".", v, fixed = TRUE)) break
    v <- sub("\\.[^.]*$", "", v)
  }
  v
}

run_star_align_3 <- function() {
  inp <- argv$input; n <- length(inp)
  if (n %% 6 != 0) stop("star_align_3 expects inputs in groups of 6")
  coord_bam <- basename(inp[seq(3, n, 6)])
  bigwig    <- basename(inp[seq(5, n, 6)])
  sj  <- paste0(vapply(coord_bam, strip_extensions, "", 5, USE.NAMES = FALSE), ".SJ.out.tab")
  wasp <- if (nzchar(argv$var_vcf_file)) "_wasp_qc" else ""
  trans <- paste0(vapply(coord_bam, strip_extensions, "", 4, USE.NAMES = FALSE),
                  ".toTranscriptome.out", wasp, ".bam")
  out <- tibble::tibble(sample_id = argv$sample_id, strand = argv$strand,
                        coord_bam_list = coord_bam, BW_list = bigwig,
                        SJ_list = sj, trans_bam_list = trans)
  vroom::vroom_write(out, argv$output, delim = "\t", quote = "none", escape = "none")
}

# ── Step: rsem_call_2 ────────────────────────────────────────────────────────

derive_sample_name <- function(path_str) {
  name <- basename(path_str)
  for (sfx in c(".rsem.isoforms.results", ".rsem.genes.results", ".rsem.cnt"))
    if (endsWith(name, sfx)) return(substr(name, 1, nchar(name) - nchar(sfx)))
  name
}

merge_rsem_metric <- function(input_files, metric_name, output_path) {
  first <- read_chr(input_files[1])
  if (!metric_name %in% colnames(first)) stop(sprintf("%s not found in %s", metric_name, input_files[1]))
  id_col <- colnames(first)[1]
  extra_cols <- setdiff(colnames(first)[1:2], metric_name)
  merged <- first[, extra_cols, drop = FALSE]
  for (path in input_files) {
    frame <- read_chr(path)
    sample_name <- derive_sample_name(path)
    sf <- setNames(frame[, c(id_col, metric_name)], c(id_col, sample_name))
    merged <- full_join(merged, sf, by = id_col)
  }
  # pandas outer-merge orders the result by the join key (byte order)
  merged <- merged[order(merged[[id_col]], method = "radix"), , drop = FALSE]
  vroom::vroom_write(merged, output_path, delim = "\t", quote = "none", escape = "none")
}

read_rsem_cnt <- function(paths) {
  if (length(paths) == 0) stop("No RSEM count files provided")
  rows <- lapply(paths, function(path) {
    ln <- readLines(path, n = 3)
    ln <- ln[!startsWith(ln, "#")]
    r0 <- strsplit(trimws(ln[1]), " +")[[1]]
    r1 <- strsplit(trimws(ln[2]), " +")[[1]]
    tibble::tibble(Sample = derive_sample_name(path), File = path,
                   TotalReads = as.numeric(r0[4]), AlignedReads = as.numeric(r0[2]),
                   UniquelyAlignedReads = as.numeric(r1[1]))
  })
  bind_rows(rows)
}

run_rsem_call_2 <- function() {
  dir.create(argv$cwd, recursive = TRUE, showWarnings = FALSE)
  prefix <- file.path(argv$cwd, argv$name)
  inp <- argv$input; n <- length(inp)
  isoform <- inp[seq(1, n, 3)]
  gene    <- inp[seq(2, n, 3)]
  cnt     <- inp[seq(3, n, 3)]

  cat(paste(isoform, collapse = "\n"), file = paste0(prefix, ".rsem.isoforms_output_list"))
  cat(paste(gene,    collapse = "\n"), file = paste0(prefix, ".rsem.genes_output_list"))

  merge_rsem_metric(isoform, "expected_count", paste0(prefix, ".rsem_transcripts_expected_count.txt.gz"))
  merge_rsem_metric(isoform, "TPM",            paste0(prefix, ".rsem_transcripts_tpm.txt.gz"))
  merge_rsem_metric(isoform, "FPKM",           paste0(prefix, ".rsem_transcripts_fpkm.txt.gz"))
  merge_rsem_metric(isoform, "IsoPct",         paste0(prefix, ".rsem_transcripts_isopct.txt.gz"))
  merge_rsem_metric(gene,    "expected_count", paste0(prefix, ".rsem_genes_expected_count.txt.gz"))
  merge_rsem_metric(gene,    "TPM",            paste0(prefix, ".rsem_genes_tpm.txt.gz"))
  merge_rsem_metric(gene,    "FPKM",           paste0(prefix, ".rsem_genes_fpkm.txt.gz"))

  metrics <- read_rsem_cnt(cnt)
  vroom::vroom_write(metrics, paste0(prefix, ".rsem.aggregated_quality.metrics.tsv"),
                     delim = "\t", quote = "none", escape = "none")
}

# ── Step: detect_strand ──────────────────────────────────────────────────────
# From a STAR ReadsPerGene.out.tab, classify library strandedness. Mirrors the
# notebook's strand_detected_1 python: sum the count columns over rows 4+ (the
# .loc[3::] slice keeps N_ambiguous + genes), take ratios to the unstranded
# total, and threshold. The chosen strand is printed to stdout (diagnostics to
# stderr) so the SoS cell can capture it into its shared variable.
run_detect_strand <- function() {
  d <- as.data.frame(vroom::vroom(argv$input, delim = "\t", col_names = FALSE, skip = 3,
                                  show_col_types = FALSE, progress = FALSE))
  num <- d[, vapply(d, is.numeric, logical(1)), drop = FALSE]
  sums <- colSums(num, na.rm = TRUE)
  sp <- sums / sums[1]
  strand <- if (sp[2] > 0.9) "fr" else if (sp[3] > 0.9) "rf" else
            if (max(sp[2], sp[3]) < 0.6) "unstranded" else "strand_missing"
  message(sprintf("strand ratios (unstranded/fr/rf): %s -> %s",
                  paste(sprintf("%.4f", sp), collapse = ", "), strand))
  cat(strand)
}

# ── Step: fastp_manifest ─────────────────────────────────────────────────────
# Rewrite the sample sheet's fq1 (and fq2, paired-end) columns to the trimmed
# fastq paths. Mirrors the notebook's fastp_trim_adaptor_2 python.
run_fastp_manifest <- function() {
  sheet <- as.data.frame(vroom::vroom(argv$sample_list, delim = "\t",
                                      col_types = vroom::cols(.default = "c"),
                                      show_col_types = FALSE, progress = FALSE), check.names = FALSE)
  inp <- argv$input; n <- length(inp)
  if (argv$is_paired_end == 1) {
    sheet$fq1 <- inp[seq(1, n, 2)]
    sheet$fq2 <- inp[seq(2, n, 2)]
  } else {
    sheet$fq1 <- inp
  }
  vroom::vroom_write(sheet, argv$output, delim = "\t", quote = "none", escape = "none")
}

# ── Step: ribosomal_intervals ────────────────────────────────────────────────
# Build a Picard RIBOSOMAL_INTERVALS file: the BAM's SAM header (via Rsamtools,
# byte-identical to `samtools view --no-PG -H`) followed by one row per rRNA
# transcript. Mirrors the python3 heredoc formerly embedded in RNA_calling.sh:
# a transcript line whose raw text contains "rRNA" and whose attributes carry a
# transcript_id becomes chrom<TAB>start<TAB>end<TAB>strand<TAB>transcript_id.
run_ribosomal_intervals <- function() {
  # SAM header from the BAM (replaces the external `samtools view -H`)
  hdr <- Rsamtools::scanBamHeader(argv$bam)[[1]]$text
  header_lines <- vapply(seq_along(hdr),
                         function(i) paste(c(names(hdr)[i], hdr[[i]]), collapse = "\t"), "")
  writeLines(header_lines, argv$output)

  con <- if (grepl("\\.gz$", argv$gtf)) gzfile(argv$gtf, "rt") else file(argv$gtf, "rt")
  lines <- readLines(con); close(con)
  lines <- lines[!startsWith(lines, "#")]
  parts <- strsplit(lines, "\t", fixed = TRUE)
  ok <- grepl("rRNA", lines, fixed = TRUE) &
        vapply(parts, function(p) length(p) >= 9 && p[3] == "transcript" &&
                 grepl("transcript_id", p[9], fixed = TRUE), logical(1))
  sel <- parts[ok]
  if (length(sel) == 0) return(invisible())
  tid <- vapply(sel, function(p) sub('.*transcript_id +"?([^";]+)"?.*', "\\1", p[9]), "")
  rows <- vapply(seq_along(sel), function(i)
    paste(sel[[i]][1], sel[[i]][4], sel[[i]][5], sel[[i]][7], tid[i], sep = "\t"), "")
  cat(paste0(rows, "\n", collapse = ""), file = argv$output, append = TRUE)
}

switch(argv$step,
       aggregate_picard_qc = run_aggregate_picard_qc(),
       rnaseqc_merge       = run_rnaseqc_merge(),
       star_align_3        = run_star_align_3(),
       rsem_call_2         = run_rsem_call_2(),
       detect_strand       = run_detect_strand(),
       fastp_manifest      = run_fastp_manifest(),
       ribosomal_intervals = run_ribosomal_intervals(),
       stop(sprintf("Unknown step: '%s'", argv$step)))
