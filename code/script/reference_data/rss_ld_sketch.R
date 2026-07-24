#!/usr/bin/env Rscript
# rss_ld_sketch.R — worker for rss_ld_sketch.ipynb (RSS LD random-projection sketch).
#
# Two steps:
#   generate_w     Draw the shared projection matrix W ~ N(0, 1/sqrt(n)), shape (n, B).
#                  Run ONCE; every chromosome/block reuses the same W so the per-block
#                  stochastic genotype sketches are arithmetically mergeable.
#   process_block  For one LD block: read the VCF region, filter variants
#                  (biallelic / polymorphic / MAF / MAC / missingness), then emit
#                  the random-projection dosage sketch U = W^T G plus .map/.afreq/.meta.
#
# VCF parsing is done on raw tabix text (Rsamtools::scanTabix) rather than
# VariantAnnotation::readVcf: the latter DNA-encodes ALT and dies on '*' (spanning
# deletion) alleles, whereas the pipeline counts genotype allele indices as strings.
#
# NOTE (W reproducibility): W is drawn with R's RNG, so the *dosage* sketch is not
# byte-identical to the numpy-generated MWE. The deterministic outputs (.map, .meta,
# and the CHROM/ID/REF/ALT/ALT_FREQS/OBS_CT columns of .afreq) ARE byte-identical.

suppressPackageStartupMessages({
  library(argparser)
  library(Rsamtools)
})

# ---- helpers ---------------------------------------------------------------

# Diploid dosage per sample from a VCF genotype column; missing ("." allele) -> NA.
extract_dosage <- function(gt_field) {
  alleles <- strsplit(gt_field, "[/|]")
  vapply(alleles, function(a) if (any(a == ".")) NA_real_ else sum(as.integer(a)), numeric(1))
}

do_generate_w <- function(argv) {
  n <- as.integer(argv$n_samples); B <- as.integer(argv$B); seed <- as.integer(argv$seed)
  set.seed(seed)
  W <- matrix(rnorm(n * B, mean = 0, sd = 1 / sqrt(n)), nrow = n, ncol = B)
  dir.create(dirname(normalizePath(argv$output, mustWork = FALSE)), recursive = TRUE, showWarnings = FALSE)
  saveRDS(W, argv$output)
  cat(sprintf("Generated W ~ N(0, 1/sqrt(%d)), shape (%d, %d), seed=%d -> %s\n",
              n, n, B, seed, argv$output))
}

do_process_block <- function(argv) {
  chrom <- argv$chrom
  bs <- as.integer(argv$block_start); be <- as.integer(argv$block_end)
  maf_min <- as.numeric(argv$maf_min); mac_min <- as.numeric(argv$mac_min)
  msng_min <- as.numeric(argv$msng_min); B <- as.integer(argv$B)
  cohort_id <- argv$cohort_id
  block_tag <- sprintf("%s_%d_%d", chrom, bs, be)
  out_dir <- file.path(argv$output_dir, chrom, block_tag)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # optional sample subset
  keep_samples <- NULL
  if (nzchar(argv$sample_list)) keep_samples <- readLines(argv$sample_list)

  # -- scan the region and filter (BED 0-based half-open; VCF 1-based) --------
  # resolve VCF shard(s): explicit --vcf, else discover in --vcf-base by prefix+chrom
  vcf_files <- argv$vcf[!is.na(argv$vcf)]
  if (!length(vcf_files)) {
    if (is.na(argv$vcf_base)) stop("process_block needs --vcf or --vcf-base")
    all_files <- list.files(argv$vcf_base, full.names = TRUE)
    pat <- paste0("^", argv$vcf_prefix, chrom, "[:.]")
    vcf_files <- sort(all_files[grepl("\\.bgz$", all_files) &
                                grepl(pat, basename(all_files))])
    if (!length(vcf_files)) stop(sprintf("No VCF files for %s in %s", chrom, argv$vcf_base))
  }
  # first shard supplies the sample header; scan every shard for the region
  hdr <- headerTabix(TabixFile(vcf_files[1]))$header
  chrom_line <- hdr[grepl("^#CHROM", hdr)][1]
  hdr_samples <- strsplit(chrom_line, "\t", fixed = TRUE)[[1]][-(1:9)]
  lines <- unlist(lapply(vcf_files, function(vf)
    scanTabix(TabixFile(vf), param = GRanges(chrom, IRanges(bs + 1, be)))[[1]]),
    use.names = FALSE)
  smp_idx <- seq_along(hdr_samples)
  if (!is.null(keep_samples)) {
    smp_idx <- which(hdr_samples %in% keep_samples)
    if (!length(smp_idx)) stop("No sample_list samples present in VCF")
  }

  cnt <- c(total = 0, multiallelic = 0, monomorphic = 0, all_na = 0,
           high_msng = 0, low_maf = 0, low_mac = 0)
  var_info <- list(); dosage_rows <- list()
  for (ln in lines) {
    f <- strsplit(ln, "\t", fixed = TRUE)[[1]]
    pos <- as.integer(f[2])
    if (!(bs <= pos - 1 && pos - 1 < be)) next
    cnt["total"] <- cnt["total"] + 1
    ref <- f[4]; alt <- f[5]
    if (grepl(",", alt, fixed = TRUE)) { cnt["multiallelic"] <- cnt["multiallelic"] + 1; next }
    fmt <- strsplit(f[9], ":", fixed = TRUE)[[1]]
    gti <- match("GT", fmt); if (is.na(gti)) next
    smp <- f[10:length(f)][smp_idx]
    gtf <- vapply(strsplit(smp, ":", fixed = TRUE), `[`, character(1), gti)
    dos <- extract_dosage(gtf)
    v <- var(dos, na.rm = TRUE)
    if (is.na(v) || v == 0) { cnt["monomorphic"] <- cnt["monomorphic"] + 1; next }
    nn <- sum(!is.na(dos)); nna <- sum(is.na(dos))
    if (nn == 0) { cnt["all_na"] <- cnt["all_na"] + 1; next }
    alt_sum <- sum(dos, na.rm = TRUE)
    mac <- min(2 * nn - alt_sum, alt_sum); maf <- mac / (2 * nn); af <- alt_sum / (2 * nn)
    if (nna / length(dos) > msng_min) { cnt["high_msng"] <- cnt["high_msng"] + 1; next }
    if (maf < maf_min) { cnt["low_maf"] <- cnt["low_maf"] + 1; next }
    if (mac < mac_min) { cnt["low_mac"] <- cnt["low_mac"] + 1; next }
    var_info[[length(var_info) + 1]] <- list(
      chr = chrom, pos = pos, ref = ref, alt = alt,
      af = round(af, 6), id = sprintf("%s:%d:%s:%s", chrom, pos, ref, alt), obs_ct = 2 * nn)
    dosage_rows[[length(dosage_rows) + 1]] <- dos
  }
  n_passed <- length(var_info)
  if (!n_passed) stop(sprintf("No passing variants in %s [%d, %d)", chrom, bs, be))
  n_samples <- length(smp_idx)

  # -- load W and compute U = W^T G ------------------------------------------
  W <- readRDS(argv$w_matrix)
  if (!all(dim(W) == c(n_samples, B))) {
    stop(sprintf("W shape mismatch: (%d, %d) vs (%d, %d)", nrow(W), ncol(W), n_samples, B))
  }
  G <- do.call(cbind, dosage_rows)                    # n_samples x n_variants
  # mean-fill missing per column
  col_means <- colMeans(G, na.rm = TRUE)
  na_ij <- which(is.na(G), arr.ind = TRUE)
  if (nrow(na_ij)) G[na_ij] <- col_means[na_ij[, "col"]]
  # standardize columns (avoid /0)
  cmu <- colMeans(G); csd <- apply(G, 2, sd); csd[csd == 0] <- 1
  G <- sweep(sweep(G, 2, cmu, "-"), 2, csd, "/")
  U <- crossprod(W, G)                                # B x n_variants (BLAS)
  col_min <- apply(U, 2, min); col_max <- apply(U, 2, max)
  denom <- col_max - col_min; denom[denom == 0] <- 1
  U <- 2 * sweep(sweep(U, 2, col_min, "-"), 2, denom, "/")
  U <- round(U, 4)

  ids <- vapply(var_info, `[[`, character(1), "id")
  poss <- vapply(var_info, `[[`, numeric(1), "pos")
  refs <- vapply(var_info, `[[`, character(1), "ref")
  alts <- vapply(var_info, `[[`, character(1), "alt")
  afs <- vapply(var_info, `[[`, numeric(1), "af")
  obs <- vapply(var_info, `[[`, numeric(1), "obs_ct")

  pfx <- file.path(out_dir, sprintf("%s.%s", cohort_id, block_tag))
  # .map : CHROM ID 0 POS
  writeLines(sprintf("%s\t%s\t0\t%d", chrom, ids, poss), paste0(pfx, ".map"))
  # .meta
  writeLines(c(
    sprintf("source_n_samples=%d", n_samples), sprintf("B=%d", B),
    sprintf("chrom=%s", chrom), sprintf("block_start=%d", bs), sprintf("block_end=%d", be),
    sprintf("n_total=%d", cnt["total"]), sprintf("n_passed=%d", n_passed),
    sprintf("n_multiallelic=%d", cnt["multiallelic"]), sprintf("n_monomorphic=%d", cnt["monomorphic"]),
    sprintf("n_all_na=%d", cnt["all_na"]), sprintf("n_high_msng=%d", cnt["high_msng"]),
    sprintf("n_low_maf=%d", cnt["low_maf"]), sprintf("n_low_mac=%d", cnt["low_mac"])),
    paste0(pfx, ".meta"))
  # .afreq : #CHROM ID REF ALT ALT_FREQS OBS_CT U_MIN U_MAX
  afreq <- c("#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\tU_MIN\tU_MAX",
             sprintf("%s\t%s\t%s\t%s\t%.6f\t%d\t%.6f\t%.6f",
                     chrom, ids, refs, alts, afs, obs, col_min, col_max))
  writeLines(afreq, paste0(pfx, ".afreq"))
  # .dosage.gz : ID ALT REF u1 ... uB   (format=1, plink2-importable)
  gz <- gzfile(paste0(pfx, ".dosage.gz"), "wt", compression = 4)
  for (j in seq_len(n_passed)) {
    writeLines(sprintf("%s %s %s %s", ids[j], alts[j], refs[j],
                       paste(sprintf("%.4f", U[, j]), collapse = " ")), gz)
  }
  close(gz)
  cat(sprintf("Wrote %d variants -> %s\n", n_passed, basename(paste0(pfx, ".dosage.gz"))))
}

# ---- CLI -------------------------------------------------------------------
p <- arg_parser("RSS LD random-projection sketch (generate_w / process_block)")
p <- add_argument(p, "--step", help = "generate_w | process_block")
p <- add_argument(p, "--n-samples", help = "generate_w: total sample size n")
p <- add_argument(p, "--B", help = "sketch dimension B", default = "10000")
p <- add_argument(p, "--seed", help = "generate_w: RNG seed", default = "123")
p <- add_argument(p, "--vcf", help = "process_block: explicit VCF shard(s)", nargs = Inf)
p <- add_argument(p, "--vcf-base", help = "process_block: dir to discover VCF shards in")
p <- add_argument(p, "--vcf-prefix", help = "process_block: VCF filename prefix", default = "")
p <- add_argument(p, "--chrom", help = "process_block: chromosome (e.g. chr22)")
p <- add_argument(p, "--block-start", help = "process_block: BED start (0-based)")
p <- add_argument(p, "--block-end", help = "process_block: BED end")
p <- add_argument(p, "--w-matrix", help = "process_block: W .rds from generate_w")
p <- add_argument(p, "--cohort-id", help = "process_block: cohort id", default = "cohort")
p <- add_argument(p, "--output-dir", help = "process_block: base output dir")
p <- add_argument(p, "--maf-min", help = "process_block: min MAF", default = "0.0005")
p <- add_argument(p, "--mac-min", help = "process_block: min MAC", default = "5")
p <- add_argument(p, "--msng-min", help = "process_block: max missingness", default = "0.05")
p <- add_argument(p, "--sample-list", help = "process_block: optional sample subset file", default = "")
p <- add_argument(p, "--output", help = "generate_w: output W .rds")
argv <- parse_args(p)

if (identical(argv$step, "generate_w")) {
  do_generate_w(argv)
} else if (identical(argv$step, "process_block")) {
  do_process_block(argv)
} else {
  stop("--step must be 'generate_w' or 'process_block'")
}
