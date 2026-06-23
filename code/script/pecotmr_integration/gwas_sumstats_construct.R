#!/usr/bin/env Rscript
# gwas_sumstats_construct.R
#
# Build one pecotmr::GwasSumStats over a single LD block from a GWAS
# summary-statistics TSV and an LD-reference meta file, run
# summaryStatsQc(), and serialize to RDS. The resulting RDS is the
# per-(study, block) input to twasWeightsPipeline downstream consumers,
# causalInferencePipeline, and ctwasPipeline.
#
# Inputs:
#   --study           Single study identifier
#   --gwas-tsv        Path to a tabix-indexed (or plain) GWAS TSV
#                     Standard columns (with hardcoded aliases when no
#                     --column-mapping is supplied):
#                       #chrom|chrom|chr, pos, variant_id|SNP,
#                       A1, A2, z|Z, n_sample|N
#                     Optional: effect_allele_frequency, p, beta, se
#   --column-mapping  Optional YAML file mapping standard column names
#                     to this study's actual column names. Keys are
#                     the standard names (chrom, pos, variant_id, A1,
#                     A2, z, n_sample, and optionally beta, se, p, maf,
#                     info); values are the column name as it appears
#                     in the TSV. Required entries: chrom, pos,
#                     variant_id, A1, A2, z, n_sample. Used in place
#                     of the hardcoded alias fallback.
#   --ld-block        Genomic interval as chr:start-end (one LD block)
#   --ld-meta         LD-meta TSV (#chr, start, end, path)
#   --genome          Genome build label (e.g. "GRCh38"). Default "GRCh38"
#   --qc-method       LD-mismatch QC method passed to
#                     summaryStatsQc(zMismatchQc = ...). One of "none"
#                     (default), "slalom", "dentist".
#   --impute          Flag: run RAISS sumstat imputation in
#                     summaryStatsQc(impute = TRUE) against the LD
#                     sketch.
#   --qc-args         Optional JSON object of extra named kwargs
#                     spliced into summaryStatsQc(). Anything in the
#                     summaryStatsQc() signature is reachable here
#                     (e.g. '{"mafCutoff":0.01,"imputeOpts":{"rcond":0.005}}').
#   --output          Output RDS path

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
  library(GenomicRanges)
  library(IRanges)
  library(S4Vectors)
  library(jsonlite)
  library(yaml)
})

parser <- arg_parser("Build a per-LD-block GwasSumStats RDS")
parser <- add_argument(parser, "--study",
                       help = "Single study identifier",
                       type = "character")
parser <- add_argument(parser, "--gwas-tsv",
                       help = "Path to GWAS summary-statistics TSV",
                       type = "character")
parser <- add_argument(parser, "--ld-block",
                       help = "LD block as chr:start-end",
                       type = "character")
parser <- add_argument(parser, "--ld-meta",
                       help = "Path to LD-meta TSV",
                       type = "character")
parser <- add_argument(parser, "--genome",
                       help = "Genome build label",
                       type = "character", default = "GRCh38")
parser <- add_argument(parser, "--column-mapping",
                       help = "Optional YAML file mapping standard column names to this study's column names",
                       type = "character", default = "")
parser <- add_argument(parser, "--qc-method",
                       help = "LD-mismatch QC: 'none' (default), 'slalom', or 'dentist'",
                       type = "character", default = "none")
parser <- add_argument(parser, "--impute",
                       help = "Enable RAISS sumstat imputation in summaryStatsQc()",
                       flag = TRUE)
parser <- add_argument(parser, "--qc-args",
                       help = "JSON object of extra named kwargs for summaryStatsQc()",
                       type = "character", default = "")
parser <- add_argument(parser, "--skip-qc",
                       help = "Skip summaryStatsQc() (debug-only; the output GwasSumStats will have qcInfo = list() and fineMappingPipeline / twasWeightsPipeline will reject it)",
                       flag = TRUE)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path", type = "character")
argv <- parse_args(parser)

# ----- Resolve column-name lookup -------------------------------------------
# Either pull from a user-supplied YAML mapping (per-study explicit names)
# or fall back to the hardcoded alias lists. The mapping must cover every
# required standard name; optional names (BETA/SE/P/MAF/INFO) pass through
# when present, regardless of source.
column_mapping <- if (nzchar(argv$column_mapping) && argv$column_mapping != ".") {
  if (!file.exists(argv$column_mapping))
    stop("--column-mapping file not found: ", argv$column_mapping)
  yaml::read_yaml(argv$column_mapping)
} else {
  NULL
}

# ----- Parse --qc-args JSON -------------------------------------------------
qc_extra <- if (nzchar(argv$qc_args) && argv$qc_args != "." &&
                argv$qc_args != "{}") {
  parsed <- tryCatch(jsonlite::fromJSON(argv$qc_args, simplifyVector = FALSE),
                     error = function(e) stop(
                       "--qc-args must be a JSON object string (got: ",
                       argv$qc_args, "). Error: ", conditionMessage(e)))
  if (!is.list(parsed) || is.null(names(parsed)) || any(names(parsed) == ""))
    stop("--qc-args must be a JSON object with named fields, e.g. ",
         '\'{"mafCutoff":0.01,"removeStrandAmbiguous":false}\'.')
  parsed
} else {
  list()
}
# Reject collisions between explicit flags and --qc-args (zMismatchQc /
# impute) to avoid silent override surprises.
clash <- intersect(names(qc_extra), c("zMismatchQc", "impute"))
if (length(clash) > 0L)
  stop("--qc-args sets ", paste(clash, collapse = ", "),
       " which is also controlled by a dedicated flag (--qc-method / ",
       "--impute). Pass it via the dedicated flag.")

qc_method <- match.arg(argv$qc_method, c("none", "slalom", "dentist"))

parse_region <- function(s) {
  m <- regmatches(s, regexec("^([^:]+):([0-9]+)-([0-9]+)$", s))[[1L]]
  if (length(m) != 4L)
    stop("--ld-block must be in chr:start-end format (got: ", s, ")")
  list(chr = m[[2L]], start = as.integer(m[[3L]]), end = as.integer(m[[4L]]))
}

block <- parse_region(argv$ld_block)

# ----- Read GWAS TSV and subset to the LD block -----------------------------
gwas <- read.table(if (grepl("\\.gz$", argv$gwas_tsv)) gzfile(argv$gwas_tsv)
                   else argv$gwas_tsv,
                   header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE, check.names = FALSE,
                   comment.char = "")

pick <- function(opts, where) intersect(opts, names(where))[1L]

# When the user provides a column mapping, use it directly; otherwise
# fall back to the hardcoded alias lists below.
resolve_col <- function(std, fallback) {
  if (!is.null(column_mapping) && !is.null(column_mapping[[std]])) {
    named <- column_mapping[[std]]
    if (!(named %in% names(gwas)))
      stop("--column-mapping['", std, "'] = '", named,
           "' is not a column in ", argv$gwas_tsv)
    return(named)
  }
  pick(fallback, gwas)
}
chr_col <- resolve_col("chrom",      c("#chrom", "chrom", "chr"))
pos_col <- resolve_col("pos",        c("pos", "position", "BP"))
snp_col <- resolve_col("variant_id", c("variant_id", "SNP", "rsid"))
a1_col  <- resolve_col("A1",         c("A1", "a1"))
a2_col  <- resolve_col("A2",         c("A2", "a2"))
z_col   <- resolve_col("z",          c("z", "Z"))
n_col   <- resolve_col("n_sample",   c("n_sample", "N", "n"))

if (any(is.na(c(chr_col, pos_col, snp_col, a1_col, a2_col, z_col, n_col))))
  stop("--gwas-tsv missing one of required columns ",
       "(chrom/pos/variant_id/A1/A2/z/n_sample) in: ", argv$gwas_tsv,
       if (!is.null(column_mapping))
         " — check that every required key is present in --column-mapping"
       else "")

# Normalise chromosome label to match the LD block's
chrom_vals <- as.character(gwas[[chr_col]])
if (!startsWith(chrom_vals[[1L]], "chr"))
  chrom_vals <- paste0("chr", chrom_vals)

pos_vals <- as.integer(gwas[[pos_col]])
keep <- chrom_vals == block$chr & pos_vals >= block$start & pos_vals <= block$end
sub <- gwas[keep, , drop = FALSE]
if (nrow(sub) == 0L)
  stop("No GWAS variants fall in LD block ", argv$ld_block,
       " (after chromosome normalisation).")

# ----- Build the LD-reference GenotypeHandle for the block ------------------
# Resolve the LD-meta row covering this block directly: pecotmr's
# GenotypeHandle(ldMeta=…, region=…) currently round-trips the meta file
# path as the data path (an upstream bug), so we read the meta TSV
# ourselves and call the appropriate GenotypeHandle constructor.
ld_meta_dir <- dirname(normalizePath(argv$ld_meta))
ld_meta_df <- read.table(argv$ld_meta, header = TRUE, sep = "\t",
                          stringsAsFactors = FALSE, check.names = FALSE,
                          comment.char = "")
# Header may be "#chr", "#chrom", "chr", or "chrom".
ld_chr_col <- intersect(c("#chr", "#chrom", "chr", "chrom"),
                        names(ld_meta_df))[1L]
if (is.na(ld_chr_col))
  stop("Could not find a chromosome column in ", argv$ld_meta,
       " (expected one of '#chr' / '#chrom' / 'chr' / 'chrom'); got: ",
       paste(names(ld_meta_df), collapse = ", "))
# Normalize both sides to the bare chromosome label (no "chr" prefix).
ld_chr_norm    <- sub("^chr", "",
                      as.character(ld_meta_df[[ld_chr_col]]),
                      ignore.case = TRUE)
block_chr_norm <- sub("^chr", "", block$chr, ignore.case = TRUE)
# `start = 0, end = 0` is the meta convention for "whole chromosome" — any
# block on that chromosome is covered.
ld_start <- suppressWarnings(as.integer(ld_meta_df$start))
ld_end   <- suppressWarnings(as.integer(ld_meta_df$end))
whole_chrom <- !is.na(ld_start) & !is.na(ld_end) &
               ld_start == 0L & ld_end == 0L
covers <- which(ld_chr_norm == block_chr_norm &
                (whole_chrom |
                 (ld_start <= block$start & ld_end >= block$end)))
if (length(covers) == 0L)
  stop("No LD-meta row in ", argv$ld_meta, " fully covers ", argv$ld_block)
if (length(covers) > 1L)
  stop("Multiple LD-meta rows cover ", argv$ld_block,
       "; restrict to a single LD block.")
ld_prefix <- ld_meta_df$path[covers[[1L]]]
if (!startsWith(ld_prefix, "/"))
  ld_prefix <- file.path(ld_meta_dir, ld_prefix)

# Detect data format from companion file extensions
if (file.exists(paste0(ld_prefix, ".pgen"))) {
  ld_handle <- GenotypeHandle(plink2Prefix = ld_prefix)
} else if (file.exists(paste0(ld_prefix, ".bed"))) {
  ld_handle <- GenotypeHandle(plink1Prefix = ld_prefix)
} else if (file.exists(paste0(ld_prefix, ".gds"))) {
  ld_handle <- GenotypeHandle(path = paste0(ld_prefix, ".gds"))
} else if (file.exists(paste0(ld_prefix, ".vcf.gz"))) {
  ld_handle <- GenotypeHandle(path = paste0(ld_prefix, ".vcf.gz"))
} else {
  stop("Could not find a recognised genotype payload at LD-meta prefix: ", ld_prefix,
       " (looked for .pgen / .bed / .gds / .vcf.gz)")
}

# ----- Build GwasSumStats entry GRanges -------------------------------------
entry_gr <- GRanges(
  seqnames = chrom_vals[keep],
  ranges   = IRanges(start = pos_vals[keep], width = 1L))
mcols(entry_gr) <- DataFrame(
  SNP = as.character(sub[[snp_col]]),
  A1  = as.character(sub[[a1_col]]),
  A2  = as.character(sub[[a2_col]]),
  Z   = as.numeric(sub[[z_col]]),
  N   = as.integer(sub[[n_col]]))
# Optional columns when present
opt_cols <- c(beta = "beta", se = "se", p = c("p", "pvalue"),
              maf = c("effect_allele_frequency", "maf", "MAF"),
              info = c("info", "INFO"))
for (slot in c("BETA", "SE", "P", "MAF", "INFO")) {
  src <- intersect(c(slot, tolower(slot),
                     if (slot == "MAF") c("effect_allele_frequency"),
                     if (slot == "P")   c("pvalue", "p")),
                   names(sub))[1L]
  if (!is.na(src)) {
    mcols(entry_gr)[[slot]] <- if (slot == "N") as.integer(sub[[src]])
                                else as.numeric(sub[[src]])
  }
}

# ----- Construct + QC + save ------------------------------------------------
gss <- GwasSumStats(
  study    = argv$study,
  entry    = list(entry_gr),
  genome   = argv$genome,
  ldSketch = ld_handle)
gss_out <- if (argv$skip_qc) {
  message("--skip-qc set; serialising raw GwasSumStats without summaryStatsQc().")
  gss
} else {
  qc_call_args <- c(list(gss,
                          zMismatchQc = qc_method,
                          impute      = isTRUE(argv$impute)),
                    qc_extra)
  do.call(summaryStatsQc, qc_call_args)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(gss_out, argv$output)
cat(sprintf("Wrote %sGwasSumStats for study '%s' over %s (%d variants in) to %s\n",
            if (argv$skip_qc) "(skip-QC) " else "QC'd ",
            argv$study, argv$ld_block, length(entry_gr), argv$output))
