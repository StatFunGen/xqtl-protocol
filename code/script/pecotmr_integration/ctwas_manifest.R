#!/usr/bin/env Rscript
# ctwas_manifest.R
#
# Enumerate the LD-block grid cTWAS runs over. cTWAS models the whole set of
# LD-reference blocks jointly (every block is a SNP-background "region"; only a
# few carry gene weights), so this worker simply lists the blocks — one row per
# LD block with the per-block GwasSumStats RDS path the caller is expected to
# build (the `region` column drives that fan-out).
#
# Gene -> home-LD-block PLACEMENT is no longer done here: pecotmr's
# assembleCtwasInputs() now places each gene into its block internally from the
# `region` provenance carried on the TwasWeights / FineMappingResult (matching
# cTWAS's own p0 assignment rule). So this manifest is pecotmr-free and does NOT
# read the weights or the xQTL meta table — the weights are handed to
# ctwas_assemble.R as a single FLAT set, not bucketed per block.
#
# NOTE: no data-layout path is hardcoded. The LD-meta table and the GwasSumStats
# output directory come from arguments.
#
# Inputs:
#   --ld-meta            LD-meta TSV (#chr/start/end[/path]); rows are LD blocks
#   --gwas-sumstats-dir  Directory the per-block GwasSumStats RDS live in
#                        (path = <dir>/<region_id>.gwas_sumstats.rds)
#   --chrom              Optional chromosome filter (e.g. "22" or "chr22");
#                        default: every chromosome in --ld-meta
#   --output             Output manifest TSV
#
# Output columns: region_id, region, gwas_sumstats_rds

suppressPackageStartupMessages({
  library(argparser)
})

p <- arg_parser("Enumerate the per-LD-block cTWAS grid")
p <- add_argument(p, "--ld-meta", type = "character",
                  help = "LD-meta TSV (#chr/start/end)")
p <- add_argument(p, "--gwas-sumstats-dir", type = "character",
                  help = "Directory holding the per-block GwasSumStats RDS")
p <- add_argument(p, "--chrom", type = "character", default = "",
                  help = "Optional chromosome filter (default: all chroms in --ld-meta)")
p <- add_argument(p, "--output", type = "character", help = "Output manifest TSV")
argv <- parse_args(p)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "manifest_common.R"))


# ---- enumerate LD blocks ---------------------------------------------------
ld <- readMeta(argv$ld_meta)
ldChrCol <- intersect(c("#chr", "#chrom", "chr", "chrom"), names(ld))[1L]
if (is.na(ldChrCol))
  stop("--ld-meta needs a chromosome column; got: ",
       paste(names(ld), collapse = ", "))
ld$.chr   <- chromStrip(ld[[ldChrCol]])
ld$.start <- suppressWarnings(as.integer(ld$start))
ld$.end   <- suppressWarnings(as.integer(ld$end))
if (nzchar(argv$chrom))
  ld <- ld[ld$.chr %in% chromStrip(argv$chrom), , drop = FALSE]
ld <- ld[!is.na(ld$.start) & !is.na(ld$.end), , drop = FALSE]
ld <- ld[!duplicated(ld[, c(".chr", ".start", ".end")]), , drop = FALSE]
ld <- ld[order(ld$.chr, ld$.start), , drop = FALSE]
if (nrow(ld) < 2L)
  stop("Fewer than two LD blocks in scope (got ", nrow(ld),
       "); cTWAS's EM needs multi-block context.")

region    <- sprintf("chr%s:%d-%d", ld$.chr, ld$.start, ld$.end)
region_id <- gsub("[:-]", "_", region)
gwas_rds  <- file.path(argv$gwas_sumstats_dir,
                       paste0(region_id, ".gwas_sumstats.rds"))

out <- data.frame(region_id = region_id, region = region,
                  gwas_sumstats_rds = gwas_rds,
                  stringsAsFactors = FALSE)
writeManifest(out, argv$output)
cat(sprintf("Wrote cTWAS block grid: %d LD block(s) to %s\n",
            nrow(out), argv$output))
