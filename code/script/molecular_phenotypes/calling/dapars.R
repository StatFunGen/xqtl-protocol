#!/usr/bin/env Rscript
# ============================================================
# dapars.R
# Pure-R port of the DaPars2 helper scripts shipped in
# code/SoS/molecular_phenotypes/calling/apa/ (python2), for apa_calling.ipynb.
# No python2 / bedtools / samtools.
#
# Steps (--step):
#   gtf2bed12    — GTF -> BED12 (per-transcript exon blocks) + transcript_to_geneName
#                  (port of gtf2bed12.py)
#
# Faithful to the python: keep gene_type in {protein_coding, lincRNA, lncRNA} on
# chr1-22/X/Y; transcript key = tid:gid:gname:chr:strand; output in first-seen
# transcript order; thick = start/stop-codon start coords (else transcript span);
# exon blockStarts are relative to transcript start, sorted by start; 1-based GTF
# coords -> 0-based BED chromStart/thick.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("dapars worker (DaPars2 helpers in R; see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "gtf2bed12")
parser <- add_argument(parser, "--gtf", type = "character", default = "", help = "[gtf2bed12] input GTF")
parser <- add_argument(parser, "--skip-rows", type = "numeric", default = 5, help = "[gtf2bed12] header rows to skip")
parser <- add_argument(parser, "--out-bed", type = "character", default = "", help = "[gtf2bed12] output BED12")
parser <- add_argument(parser, "--out-map", type = "character", default = "",
                       help = "[gtf2bed12] output transcript_to_geneName TSV")
parser <- add_argument(parser, "--bed", type = "character", default = "", help = "[extract_anno] BED12 (gtf2bed12 output)")
parser <- add_argument(parser, "--symbol", type = "character", default = "", help = "[extract_anno] transcript_to_geneName")
parser <- add_argument(parser, "--output", type = "character", default = "", help = "[extract_anno] output 3'UTR BED")
parser <- add_argument(parser, "--emit-extracted", flag = TRUE, help = "[extract_anno] write pre-subtract 3'UTRs (validation)")
parser <- add_argument(parser, "--config", type = "character", default = "", help = "[apa_main] DaPars2 config file")
parser <- add_argument(parser, "--chr", type = "character", default = "", help = "[apa_main] chromosome to process")
parser <- add_argument(parser, "--bam", type = "character", default = "", help = "[coverage] input BAM")
parser <- add_argument(parser, "--output-wig", type = "character", default = "", help = "[coverage] output bedGraph .wig")
parser <- add_argument(parser, "--output-depth", type = "character", default = "", help = "[coverage] output mapped-read depth")
parser <- add_argument(parser, "--wig-dir", type = "character", default = "", help = "[apa_config] directory of .wig (+ .depth) files")
parser <- add_argument(parser, "--annotation", type = "character", default = "", help = "[apa_config] 3'UTR annotation")
parser <- add_argument(parser, "--apa-cwd", type = "character", default = "", help = "[apa_config] DaPars2 Output_directory base")
parser <- add_argument(parser, "--coverage-threshold", type = "numeric", default = 10, help = "[apa_config] Coverage_threshold")
parser <- add_argument(parser, "--num-threads", type = "numeric", default = 1, help = "[apa_config] Num_Threads")
parser <- add_argument(parser, "--output-mapping", type = "character", default = "", help = "[apa_config] sample mapping/depth file")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# ---------------------------------------------------------------------------
gtf2bed12 <- function(argv) {
  suppressPackageStartupMessages({ library(vroom); library(stringr) })
  g <- vroom::vroom(argv$gtf, delim = "\t", col_names = FALSE, skip = as.integer(argv$skip_rows),
                    comment = "", col_types = cols(.default = "c"), progress = FALSE)
  a <- g[[9]]
  cap <- function(p) str_match(a, p)[, 2]
  gene_id <- cap('gene_id "(\\w+(?:\\.\\d+)?)"')
  tx_id   <- cap('transcript_id "(\\w+(?:\\.\\d+)?)"')
  gname   <- cap('gene_name "([^"]*)"')
  gtype   <- cap('gene_(?:bio)?type "([^"]*)"')
  gene_id[is.na(gene_id)] <- "NA"; gname[is.na(gname)] <- "NA"; gtype[is.na(gtype)] <- "NA"

  chr <- g[[1]]; strand <- g[[7]]; etype <- g[[3]]
  s4 <- as.integer(g[[4]]); e5 <- as.integer(g[[5]])
  keep_chr <- c(paste0("chr", 1:22), "chrX", "chrY")
  keep_gt  <- c("protein_coding", "lincRNA", "lncRNA")
  base_ok  <- !is.na(tx_id) & gtype %in% keep_gt & chr %in% keep_chr
  key <- paste(tx_id, gene_id, gname, chr, strand, sep = ":")

  ex   <- which(base_ok & etype == "exon")
  ux   <- unique(key[ex])                                   # first-seen transcript order
  ex_by_key <- split(ex, factor(key[ex], levels = ux))

  last_map <- function(idx) {                               # python dict: last write wins
    k <- key[idx]; keep <- !duplicated(k, fromLast = TRUE)
    setNames(idx[keep], k[keep])
  }
  m_tx <- last_map(which(base_ok & etype == "transcript"))
  m_sc <- last_map(which(base_ok & etype == "start_codon"))
  m_ec <- last_map(which(base_ok & etype == "stop_codon"))

  bed <- character(length(ux)); mp <- character(length(ux))
  for (i in seq_along(ux)) {
    t <- ux[i]; p <- strsplit(t, ":", fixed = TRUE)[[1]]
    tid <- p[1]; gn <- p[3]; chrn <- p[4]; str <- p[5]
    ti <- unname(m_tx[t]); t_start <- s4[ti]; t_end <- e5[ti]
    exi <- ex_by_key[[t]]; o <- order(s4[exi])
    es <- s4[exi][o]; ee <- e5[exi][o]
    exon_starts <- es - t_start
    exon_sizes  <- ee - es + 1L
    sc_i <- unname(m_sc[t]); ec_i <- unname(m_ec[t])
    if (!is.na(sc_i) && !is.na(ec_i)) {
      s_codon <- s4[sc_i]; e_codon <- s4[ec_i]
    } else { s_codon <- t_start; e_codon <- t_end }
    bed[i] <- sprintf("%s\t%d\t%d\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%s\t%s",
                      chrn, t_start - 1L, t_end, tid, 0L, str, s_codon - 1L, e_codon - 1L, 0L,
                      length(es), paste0(paste(exon_sizes, collapse = ","), ","),
                      paste0(paste(exon_starts, collapse = ","), ","))
    mp[i] <- sprintf("%s\t%s", tid, gn)
  }
  writeLines(bed, argv$out_bed)
  writeLines(mp, argv$out_map)
  message(sprintf("Written: %s (%d transcripts)", argv$out_bed, length(ux)))
}

# ---------------------------------------------------------------------------
# extract_anno: BED12 -> 3'UTR regions, then remove opposite-strand overlaps.
# Port of DaPars_Extract_Anno.py. NOTE faithfully reproduces the python quirk of
# skipping the FIRST line of the transcript_to_geneName map as a "header" (which
# gtf2bed12 never writes) -> the first transcript gets gene_symbol "NA".
# The bedtools `subtractBed -S` step -> GenomicRanges::setdiff (opposite strand).
# ---------------------------------------------------------------------------
extract_anno <- function(argv) {
  suppressPackageStartupMessages(library(vroom))
  mp <- vroom::vroom(argv$symbol, delim = "\t", col_names = c("tid", "gene"), skip = 1,
                     col_types = "cc", progress = FALSE)                       # skip line 1 (python bug)
  m <- setNames(mp$gene, mp$tid)
  bed <- vroom::vroom(argv$bed, delim = "\t", col_names = FALSE,
                      col_types = cols(.default = "c"), progress = FALSE)
  chr <- bed[[1]]; start <- as.integer(bed[[2]]); end <- bed[[3]]; rid <- bed[[4]]
  strand <- bed[[6]]; bsizes <- bed[[11]]; bstarts <- bed[[12]]

  keep <- !grepl("_", chr, fixed = TRUE)                                        # skip alt/random contigs
  gs <- unname(m[rid]); gs[is.na(gs)] <- "NA"
  bs_last   <- as.integer(sub(".*,", "", sub(",$", "", bstarts)))               # last blockStart
  bsz_first <- as.integer(sub(",.*", "", bsizes))                               # first blockSize
  utr_start <- ifelse(strand == "+", as.character(start + bs_last + 1L), as.character(start + 1L))
  utr_end   <- ifelse(strand == "+", end, as.character(start + bsz_first))
  uid <- paste(rid, gs, chr, strand, sep = "|")
  dedup_key <- paste0(chr, utr_start, utr_end, strand)

  sel <- keep & !duplicated(dedup_key)                                          # first occurrence, in BED order
  ex <- data.frame(chr = chr[sel], start = utr_start[sel], end = utr_end[sel],
                   name = uid[sel], score = "0", strand = strand[sel], stringsAsFactors = FALSE)
  if (argv$emit_extracted) {                                                    # validation: pre-subtract
    writeLines(do.call(paste, c(ex, sep = "\t")), argv$output); return(invisible())
  }

  # subtractBed -a ex -b ex -S in bedtools' 0-based half-open semantics (explicit interval math,
  # NOT GenomicRanges — GRanges is 1-based closed and yields off-by-one boundaries here).
  es <- as.integer(ex$start); ee <- as.integer(ex$end); n <- nrow(ex)
  pc_chr <- character(0); pc_s <- pc_e <- integer(0); pc_name <- pc_score <- pc_strand <- character(0)
  for (i in seq_len(n)) {
    ov <- which(ex$chr == ex$chr[i] & ex$strand != ex$strand[i] & ee > es[i] & es < ee[i])
    rem <- list(c(es[i], ee[i]))
    for (j in ov) {
      bs <- es[j]; be <- ee[j]; nr <- list()
      for (r in rem) {
        rs <- r[1]; re <- r[2]
        if (be <= rs || bs >= re) nr[[length(nr) + 1L]] <- c(rs, re)
        else { if (rs < bs) nr[[length(nr) + 1L]] <- c(rs, bs); if (be < re) nr[[length(nr) + 1L]] <- c(be, re) }
      }
      rem <- nr
    }
    for (r in rem) {
      pc_chr <- c(pc_chr, ex$chr[i]); pc_s <- c(pc_s, r[1]); pc_e <- c(pc_e, r[2])
      pc_name <- c(pc_name, ex$name[i]); pc_score <- c(pc_score, ex$score[i]); pc_strand <- c(pc_strand, ex$strand[i])
    }
  }
  # per transcript (name before '|'): single piece kept, else refine (min left for +, max left for -)
  tid <- sub("\\|.*", "", pc_name)
  out_idx <- integer(0)
  for (t in unique(tid)) {
    idx <- which(tid == t)
    if (length(idx) == 1) { out_idx <- c(out_idx, idx); next }
    pick <- if (pc_strand[idx][1] == "+") idx[which.min(pc_s[idx])] else idx[which.max(pc_s[idx])]
    out_idx <- c(out_idx, pick)
  }
  df <- data.frame(pc_chr[out_idx], pc_s[out_idx], pc_e[out_idx],
                   pc_name[out_idx], pc_score[out_idx], pc_strand[out_idx])
  writeLines(do.call(paste, c(df, sep = "\t")), argv$output)
  message(sprintf("Written: %s (%d 3'UTRs)", argv$output, nrow(df)))
}

# ---------------------------------------------------------------------------
# apa_main: DaPars2 PDUI (port of Dapars2_Multi_Sample.py). Wig(bedGraph) coverage ->
# per-3'UTR two-segment mean-shift change-point -> PDUI (long inclusion ratio).
# Coordinates and slicing mirror the python exactly (0-based via pyslice()).
# ---------------------------------------------------------------------------
pyslice <- function(x, a, b) if (b <= a) x[integer(0)] else x[(a + 1L):b]      # python x[a:b], 0-based

# bedGraph -> step function (positions, values); values is 1 longer (python appends a trailing 0)
load_wig <- function(wig_file, chr) {
  lines <- readLines(wig_file)
  lines <- lines[nzchar(lines) & !startsWith(lines, "#") & !startsWith(lines, "t")]
  m <- do.call(rbind, strsplit(lines, "\t", fixed = TRUE))
  keep <- m[, 1] == chr
  s <- as.integer(m[keep, 2]); e <- as.integer(m[keep, 3]); v <- as.integer(as.numeric(m[keep, 4]))
  n <- length(s)
  positions <- integer(2L * n + 2L); values <- integer(2L * n + 2L)
  positions[1] <- 0L; values[1] <- 0L; p <- 1L
  for (i in seq_len(n)) {
    if (s[i] > positions[p]) { p <- p + 1L; positions[p] <- s[i]; values[p] <- 0L }
    p <- p + 1L; positions[p] <- e[i]; values[p] <- v[i]
  }
  list(positions = positions[1:p], values = c(values[1:p], 0L))
}

# step function over an event -> per-bp coverage (Convert_wig_into_bp_coverage)
convert_wig <- function(cov, reg, strand) {
  n <- reg[length(reg)] - reg[1]
  bp <- numeric(max(n, 0L))
  for (i in seq_along(cov)) {
    a <- reg[i] - reg[1]; b <- reg[i + 1] - reg[1]
    if (b > a) bp[(a + 1L):b] <- cov[i]
  }
  if (strand == "-") rev(bp) else bp
}

# numpy's exact pairwise summation (8-accumulator block <=128, recursive split above) so means
# match np.mean bit-for-bit — matters only where the MSE landscape is degenerate (flat coverage),
# but there np.mean-vs-R-mean summation order would otherwise flip the argmin break point.
np_sum <- function(a) {
  n <- length(a)
  if (n == 0) return(0)
  if (n < 8) return(Reduce(`+`, a))                                            # sequential double
  if (n <= 128) {
    r <- a[1:8]; i <- 8L; lim <- n - (n %% 8L)
    while (i < lim) { r <- r + a[(i + 1L):(i + 8L)]; i <- i + 8L }
    res <- ((r[1] + r[2]) + (r[3] + r[4])) + ((r[5] + r[6]) + (r[7] + r[8]))
    while (i < n) { res <- res + a[i + 1L]; i <- i + 1L }
    return(res)
  }
  n2 <- (n %/% 2L); n2 <- n2 - (n2 %% 8L)
  np_sum(a[1:n2]) + np_sum(a[(n2 + 1L):n])
}
np_mean <- function(a) np_sum(a) / length(a)

estimation_abundance <- function(cov, bp) {                                    # bp = # proximal elements
  L <- length(cov)
  distal <- cov[(bp + 1L):L]; prox <- cov[1:bp]
  Long <- np_mean(distal); Short <- np_mean(prox - Long); if (Short < 0) Short <- 0
  d <- c(prox - Long - Short, distal - Long)
  list(mse = np_mean(d^2), long = Long, short = Short)
}

de_novo <- function(all_cov, ustart, uend, strand, weights, cov_thresh, least_pass) {
  search_start <- 150L; end_pt <- as.integer(abs(uend - ustart) * 0.05)
  ns <- length(all_cov); region_covs <- list(); pass_idx <- integer(0)
  for (i in seq_len(ns)) {
    raw <- all_cov[[i]]
    if (np_mean(raw[seq_len(min(99L, length(raw)))]) > cov_thresh) {
      pass_idx <- c(pass_idx, i); region_covs[[length(region_covs) + 1L]] <- raw / weights[i]
    }
  }
  if (length(pass_idx) > ns * least_pass && (uend - ustart) >= 150) {
    sr_end <- uend - ustart - end_pt
    cps <- if (sr_end >= search_start) search_start:sr_end else integer(0)     # python range(150, sr_end+1)
    if (length(cps) > 1) {
      mse_list <- numeric(length(cps)); abund_list <- vector("list", length(cps))
      for (j in seq_along(cps)) {
        res <- lapply(region_covs, estimation_abundance, bp = cps[j])
        mse_list[j] <- np_mean(vapply(res, `[[`, numeric(1), "mse"))
        abund_list[[j]] <- list(vapply(res, `[[`, numeric(1), "long"), vapply(res, `[[`, numeric(1), "short"))
      }
      k <- which.min(mse_list)                                                 # first minimum
      coord <- if (strand == "+") ustart + cps[k] else uend - cps[k]           # == search_region[k]
      lf <- rep(NA_real_, ns); sf <- rep(NA_real_, ns)
      lf[pass_idx] <- abund_list[[k]][[1]]; sf[pass_idx] <- abund_list[[k]][[2]]
      return(list(mse = mse_list[k], bp = coord, long = lf, short = sf))
    }
  }
  list(mse = "Na")
}

apa_main <- function(argv) {
  cfg <- readLines(argv$config); chr <- argv$chr
  getv <- function(k) { l <- cfg[startsWith(cfg, paste0(k, "="))]; if (length(l)) sub(paste0("^", k, "="), "", l[1]) else "" }
  wig_files <- strsplit(getv("Aligned_Wig_files"), ",", fixed = TRUE)[[1]]
  cov_thresh <- suppressWarnings(as.integer(getv("Coverage_threshold"))); if (is.na(cov_thresh)) cov_thresh <- 1L
  sample_names <- sub("\\.[^.]*$", "", wig_files); ns <- length(wig_files)
  out_dir <- paste0(sub("/$", "", getv("Output_directory")), "_", chr, "/")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_file <- paste0(out_dir, getv("Output_result_file"), "_result_temp.", chr, ".txt")
  depths <- vapply(readLines(getv("sequencing_depth_file")),
                   function(l) { f <- strsplit(l, "\t", fixed = TRUE)[[1]]; as.integer(f[length(f)]) }, integer(1))
  weights <- depths / np_mean(depths)

  # UTR events (20% end shift + 1bp inset + length filter), in annotation order for this chr
  ann <- do.call(rbind, strsplit(readLines(getv("Annotated_3UTR")), "\t", fixed = TRUE))
  ev_id <- character(0); ev <- list()
  for (i in which(ann[, 1] == chr)) {
    rs <- as.integer(ann[i, 2]); re <- as.integer(ann[i, 3]); st <- ann[i, ncol(ann)]
    pos <- sprintf("%s:%d-%d", chr, rs, re)
    shift <- as.integer(round(abs(rs - re) * 0.2))
    if (st == "+") re <- re - shift else rs <- rs + shift
    rs <- rs + 1L; re <- re - 1L
    if (rs + 50L < re && !(ann[i, 4] %in% ev_id)) { ev_id <- c(ev_id, ann[i, 4]); ev[[ann[i, 4]]] <- list(chr, rs, re, st, pos) }
  }

  con <- file(out_file, "w")
  writeLines(paste(c("Gene", "fit_value", "Predicted_Proximal_APA", "Loci", paste0(sample_names, "_PDUI")), collapse = "\t"), con)
  wigs <- lapply(wig_files, load_wig, chr = chr)
  for (id in ev_id) {
    e <- ev[[id]]; rs <- e[[2]]; re <- e[[3]]; strand <- e[[4]]
    bp_cov <- lapply(seq_len(ns), function(i) {
      li <- findInterval(rs, wigs[[i]]$positions); ri <- findInterval(re, wigs[[i]]$positions)
      cov <- pyslice(wigs[[i]]$values, li, ri + 1L)
      reg <- c(rs, pyslice(wigs[[i]]$positions, li, ri), re)
      convert_wig(cov, reg, strand)
    })
    est <- de_novo(bp_cov, rs, re, strand, weights, cov_thresh, 0.3)
    if (!identical(est$mse, "Na")) {
      line <- c(id, sprintf("%.1f", est$mse), as.character(est$bp), e[[5]])
      for (i in seq_len(ns)) line <- c(line, if (!is.na(est$long[i]))
        sprintf("%.2f", est$long[i] / (est$long[i] + est$short[i])) else "NA")
      writeLines(paste(line, collapse = "\t"), con)
    }
  }
  close(con)
  message(sprintf("Written: %s", out_file))
}

# ---------------------------------------------------------------------------
# coverage: BAM -> bedGraph .wig (replaces `bedtools genomecov -bga -split`) + mapped-read
# depth (replaces `samtools flagstat`). CIGAR-aware coverage() splits at N like -split.
# ---------------------------------------------------------------------------
coverage_step <- function(argv) {
  suppressPackageStartupMessages({ library(GenomicAlignments); library(Rsamtools) })
  ga <- readGAlignments(argv$bam)
  gr <- as(coverage(ga), "GRanges")
  df <- data.frame(chr = as.character(seqnames(gr)), start = start(gr) - 1L,
                   end = end(gr), val = as.integer(gr$score))
  con <- file(argv$output_wig, "w"); writeLines("track type=bedGraph", con)
  write.table(df, con, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE); close(con)
  if (nzchar(argv$output_depth)) writeLines(as.character(sum(idxstatsBam(argv$bam)$mapped)), argv$output_depth)
  message(sprintf("Written: %s", argv$output_wig))
}

# ---------------------------------------------------------------------------
# apa_config: assemble the DaPars2 sample-mapping (depth) file + configuration file
# from a directory of <name>.wig (+ <name>.depth) files. Port of [APAconfig].
# ---------------------------------------------------------------------------
apa_config <- function(argv) {
  wigs <- sort(list.files(argv$wig_dir, pattern = "wig$", full.names = TRUE))
  depths <- vapply(wigs, function(w) readLines(sub("\\.wig$", ".depth", w))[1], character(1))
  writeLines(paste(wigs, depths, sep = "\t"), argv$output_mapping)
  cfg <- c(
    sprintf("Annotated_3UTR=%s", argv$annotation),
    sprintf("Aligned_Wig_files=%s", paste(wigs, collapse = ",")),
    sprintf("Output_directory=%s/apa", sub("/$", "", argv$apa_cwd)),
    "Output_result_file=Dapars_result",
    sprintf("Coverage_threshold=%d", as.integer(argv$coverage_threshold)),
    sprintf("Num_Threads=%d", as.integer(argv$num_threads)),
    sprintf("sequencing_depth_file=%s", argv$output_mapping))
  writeLines(cfg, argv$config)
  message(sprintf("Written: %s", argv$config))
}

# ---------------------------------------------------------------------------
switch(argv$step,
  gtf2bed12   = gtf2bed12(argv),
  extract_anno = extract_anno(argv),
  coverage    = coverage_step(argv),
  apa_config  = apa_config(argv),
  apa_main    = apa_main(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
