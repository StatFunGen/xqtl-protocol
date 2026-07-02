#!/usr/bin/env Rscript
# gwas_sumstats_construct.R
#
# Build a pecotmr::GwasSumStats over a single analysis region from one or
# more GWAS summary-statistics TSVs and an LD-reference meta file, run
# summaryStatsQc(), and serialize to RDS. The resulting RDS is the
# per-(region) GWAS input to twasWeightsPipeline downstream consumers,
# causalInferencePipeline, and ctwasPipeline.
#
# Multiple studies: pass --study / --gwas-tsv (and optionally
# --column-mapping) as comma-separated lists of equal length. One
# GwasSumStats entry is built per study; all studies share the region's
# LD sketch (a population LD panel is not study-specific).
# causalInferencePipeline then loops the studies and emits one row per
# (qtl tuple, gwasStudy).
#
# LD resolution: the region need NOT coincide with a single LD-meta row.
# All LD-meta rows overlapping the region are collected and their genotype
# payloads de-duplicated. The common one-file-per-chromosome layout (every
# overlapping row points at the same prefix) resolves to a single handle
# that already spans the region; genuinely separate per-block payloads are
# a hard error (multi-file LD merge is not done here).
#
# Inputs:
#   --study           Study identifier(s), comma-separated
#   --gwas-tsv        GWAS TSV path(s), comma-separated, one per study.
#                     Standard columns (with hardcoded aliases when no
#                     --column-mapping is supplied):
#                       #chrom|chrom|chr, pos, variant_id|SNP,
#                       A1, A2, z|Z, n_sample|N
#                     Optional: effect_allele_frequency, p, beta, se
#   --column-mapping  Optional YAML file(s) mapping standard column names
#                     to a study's actual column names. Comma-separated:
#                     empty (none), one (applied to every study), or one
#                     per study. Keys are the standard names (chrom, pos,
#                     variant_id, A1, A2, z, n_sample, and optionally beta,
#                     se, p, maf, info); values are the column name as it
#                     appears in the TSV. Required: chrom, pos, variant_id,
#                     A1, A2, z, n_sample.
#   --ld-block        Analysis region as chr:start-end
#   --ld-meta         LD-meta TSV (#chr, start, end, path)
#   --genome          Genome build label (e.g. "GRCh38"). Default "GRCh38"
#   --qc-method       LD-mismatch QC method passed to
#                     summaryStatsQc(zMismatchQc = ...). One of "none"
#                     (default), "slalom", "dentist".
#   --impute          Flag: run RAISS sumstat imputation in
#                     summaryStatsQc(impute = TRUE) against the LD sketch.
#   --maf             MAF cutoff (summaryStatsQc mafCutoff). Default 0.0025.
#   --skip-region     Comma-separated chr:start-end window(s) whose variants
#                     are dropped (summaryStatsQc skipRegion).
#   --qc-args         Optional JSON object of extra named kwargs spliced
#                     into summaryStatsQc(). May not set a key already
#                     controlled by a dedicated flag (mafCutoff / skipRegion /
#                     pipCutoffToSkip / zMismatchQc / impute).
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

parser <- arg_parser("Build a per-region (multi-study) GwasSumStats RDS")
parser <- add_argument(parser, "--study",
                       help = "Study identifier(s), comma-separated",
                       type = "character")
parser <- add_argument(parser, "--gwas-tsv",
                       help = "GWAS TSV path(s), comma-separated (one per study)",
                       type = "character")
parser <- add_argument(parser, "--ld-block",
                       help = "Analysis region as chr:start-end",
                       type = "character")
parser <- add_argument(parser, "--ld-meta",
                       help = "Path to LD-meta TSV",
                       type = "character")
parser <- add_argument(parser, "--genome",
                       help = "Genome build label",
                       type = "character", default = "GRCh38")
parser <- add_argument(parser, "--column-mapping",
                       help = "Optional YAML mapping file(s), comma-separated (none / one-for-all / one-per-study)",
                       type = "character", default = "")
parser <- add_argument(parser, "--n-case",
                       help = "Optional per-study case counts, comma-separated (one per study; NA for quantitative). Stored on GwasSumStats for case/control effective-N downstream.",
                       type = "character", default = "")
parser <- add_argument(parser, "--n-control",
                       help = "Optional per-study control counts, comma-separated (one per study; NA for quantitative).",
                       type = "character", default = "")
parser <- add_argument(parser, "--pip-cutoff-to-skip",
                       help = "Skip a study whose single-trait max PIP is below this cutoff (summaryStatsQc pipCutoffToSkip); 0 disables, <0 uses 3/n_variants.",
                       type = "numeric", default = 0)
parser <- add_argument(parser, "--maf",
                       help = "Minor-allele-frequency cutoff (summaryStatsQc mafCutoff); drop variants below it. 0 disables.",
                       type = "numeric", default = 0.0025)
parser <- add_argument(parser, "--skip-region",
                       help = "Comma-separated chr:start-end window(s) whose overlapping variants are dropped (summaryStatsQc skipRegion). Empty = none.",
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

splitCsv <- function(s) {
  if (is.null(s) || !nzchar(s)) return(character(0))
  trimws(strsplit(s, ",", fixed = TRUE)[[1L]])
}

# ----- Resolve study / gwas-tsv / column-mapping lists ----------------------
studies  <- splitCsv(argv$study)
gwasTsvs <- splitCsv(argv$gwas_tsv)
if (length(studies) == 0L || length(gwasTsvs) == 0L)
  stop("--study and --gwas-tsv are both required.")
if (length(gwasTsvs) != length(studies))
  stop("--gwas-tsv supplies ", length(gwasTsvs), " path(s) but --study lists ",
       length(studies), " study/ies; they must match.")
if (anyDuplicated(studies))
  stop("--study has duplicate identifiers: ",
       paste(studies[duplicated(studies)], collapse = ", "))

mappings <- splitCsv(argv$column_mapping)
mappings <- if (length(mappings) == 0L) {
  rep("", length(studies))
} else if (length(mappings) == 1L) {
  rep(mappings, length(studies))
} else if (length(mappings) == length(studies)) {
  mappings
} else {
  stop("--column-mapping must be empty, a single file (applied to every ",
       "study), or one file per study (got ", length(mappings),
       " for ", length(studies), " studies).")
}

# ----- Optional per-study case/control counts -------------------------------
# Comma-separated, one per study; "NA"/"" entries allowed for quantitative
# studies in a mixed collection. NULL (unset) leaves the columns off entirely.
parseCounts <- function(s, nm) {
  v <- splitCsv(s)
  if (length(v) == 0L) return(NULL)
  num <- suppressWarnings(as.numeric(v))
  if (length(num) == 1L) num <- rep(num, length(studies))
  if (length(num) != length(studies))
    stop("--", nm, " must supply one value per study (got ", length(num),
         " for ", length(studies), " studies).")
  num
}
nCase    <- parseCounts(argv$n_case, "n-case")
nControl <- parseCounts(argv$n_control, "n-control")

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
# Reject collisions between explicit flags and --qc-args. Dedicated flags win;
# passing the same key via --qc-args is an error so behavior is unambiguous.
clash <- intersect(names(qc_extra),
                   c("zMismatchQc", "impute", "mafCutoff", "skipRegion",
                     "pipCutoffToSkip"))
if (length(clash) > 0L)
  stop("--qc-args sets ", paste(clash, collapse = ", "),
       " which is also controlled by a dedicated flag (--qc-method / ",
       "--impute / --maf / --skip-region / --pip-cutoff-to-skip). Pass it via ",
       "the dedicated flag.")

# --skip-region: comma-separated chr:start-end windows -> character vector
# (NULL when unset so summaryStatsQc's skipRegion default applies).
skip_region_vec <- splitCsv(argv$skip_region)
if (length(skip_region_vec) == 0L) skip_region_vec <- NULL

qc_method <- match.arg(argv$qc_method, c("none", "slalom", "dentist"))

parse_region <- function(s) {
  m <- regmatches(s, regexec("^([^:]+):([0-9]+)-([0-9]+)$", s))[[1L]]
  if (length(m) != 4L)
    stop("--ld-block must be in chr:start-end format (got: ", s, ")")
  list(chr = m[[2L]], start = as.integer(m[[3L]]), end = as.integer(m[[4L]]))
}

block <- parse_region(argv$ld_block)

# ----- Build the LD-reference GenotypeHandle spanning the region ------------
# Resolve every LD-meta row overlapping the region and de-duplicate the
# genotype payloads. The one-file-per-chromosome layout (all overlapping
# rows share a prefix) gives a single handle that already covers the region;
# pecotmr's GenotypeHandle(ldMeta=…, region=…) round-trips the meta path as
# the data path (an upstream bug) and rejects multi-row regions, so we read
# the meta TSV ourselves and point a constructor at the resolved payload.
buildLdHandle <- function(ld_meta, block) {
  ld_meta_dir <- dirname(normalizePath(ld_meta))
  ld_meta_df <- read.table(ld_meta, header = TRUE, sep = "\t",
                           stringsAsFactors = FALSE, check.names = FALSE,
                           comment.char = "")
  ld_chr_col <- intersect(c("#chr", "#chrom", "chr", "chrom"),
                          names(ld_meta_df))[1L]
  if (is.na(ld_chr_col))
    stop("Could not find a chromosome column in ", ld_meta,
         " (expected one of '#chr' / '#chrom' / 'chr' / 'chrom'); got: ",
         paste(names(ld_meta_df), collapse = ", "))
  ld_chr_norm    <- sub("^chr", "", as.character(ld_meta_df[[ld_chr_col]]),
                        ignore.case = TRUE)
  block_chr_norm <- sub("^chr", "", block$chr, ignore.case = TRUE)
  ld_start <- suppressWarnings(as.integer(ld_meta_df$start))
  ld_end   <- suppressWarnings(as.integer(ld_meta_df$end))
  # `start = 0, end = 0` is the meta convention for "whole chromosome".
  whole_chrom <- !is.na(ld_start) & !is.na(ld_end) &
                 ld_start == 0L & ld_end == 0L
  # Strict overlap (no shared-endpoint match) so adjacent blocks that abut
  # the region are not pulled in.
  overlaps <- which(ld_chr_norm == block_chr_norm &
                    (whole_chrom |
                     (ld_start < block$end & ld_end > block$start)))
  if (length(overlaps) == 0L)
    stop("No LD-meta row overlaps ", argv$ld_block, " in ", ld_meta, ".")
  ld_prefixes <- unique(ld_meta_df$path[overlaps])
  if (length(ld_prefixes) > 1L)
    stop("Region ", argv$ld_block, " overlaps LD-meta rows pointing at ",
         length(ld_prefixes), " distinct genotype payloads (",
         paste(ld_prefixes, collapse = ", "), "). Multi-file LD merge is ",
         "not supported here; restrict the region to a single payload.")
  ld_prefix <- ld_prefixes[[1L]]
  if (!startsWith(ld_prefix, "/"))
    ld_prefix <- file.path(ld_meta_dir, ld_prefix)

  # Detect data format from companion file extensions.
  if (file.exists(paste0(ld_prefix, ".pgen"))) {
    GenotypeHandle(plink2Prefix = ld_prefix)
  } else if (file.exists(paste0(ld_prefix, ".bed"))) {
    GenotypeHandle(plink1Prefix = ld_prefix)
  } else if (file.exists(paste0(ld_prefix, ".gds"))) {
    GenotypeHandle(path = paste0(ld_prefix, ".gds"))
  } else if (file.exists(paste0(ld_prefix, ".vcf.gz"))) {
    GenotypeHandle(path = paste0(ld_prefix, ".vcf.gz"))
  } else {
    stop("Could not find a recognised genotype payload at LD-meta prefix: ",
         ld_prefix, " (looked for .pgen / .bed / .gds / .vcf.gz)")
  }
}

ld_handle <- buildLdHandle(argv$ld_meta, block)

# ----- Build one GwasSumStats entry GRanges per study ----------------------
buildEntryGr <- function(gwas_tsv, mapping_path, block) {
  column_mapping <- if (nzchar(mapping_path) && mapping_path != ".") {
    if (!file.exists(mapping_path))
      stop("--column-mapping file not found: ", mapping_path)
    yaml::read_yaml(mapping_path)
  } else {
    NULL
  }

  gwas <- read.table(if (grepl("\\.gz$", gwas_tsv)) gzfile(gwas_tsv)
                     else gwas_tsv,
                     header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE, check.names = FALSE,
                     comment.char = "")

  pick <- function(opts, where) intersect(opts, names(where))[1L]
  resolve_col <- function(std, fallback) {
    if (!is.null(column_mapping) && !is.null(column_mapping[[std]])) {
      named <- column_mapping[[std]]
      if (!(named %in% names(gwas)))
        stop("--column-mapping['", std, "'] = '", named,
             "' is not a column in ", gwas_tsv)
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
         "(chrom/pos/variant_id/A1/A2/z/n_sample) in: ", gwas_tsv,
         if (!is.null(column_mapping))
           " — check that every required key is present in --column-mapping"
         else "")

  # Normalise chromosome label to match the region's.
  chrom_vals <- as.character(gwas[[chr_col]])
  if (!startsWith(chrom_vals[[1L]], "chr"))
    chrom_vals <- paste0("chr", chrom_vals)
  pos_vals <- as.integer(gwas[[pos_col]])
  keep <- chrom_vals == block$chr & pos_vals >= block$start &
          pos_vals <= block$end
  sub <- gwas[keep, , drop = FALSE]
  if (nrow(sub) == 0L)
    stop("No GWAS variants from ", gwas_tsv, " fall in region ",
         argv$ld_block, " (after chromosome normalisation).")

  entry_gr <- GRanges(
    seqnames = chrom_vals[keep],
    ranges   = IRanges(start = pos_vals[keep], width = 1L))
  mcols(entry_gr) <- DataFrame(
    SNP = as.character(sub[[snp_col]]),
    A1  = as.character(sub[[a1_col]]),
    A2  = as.character(sub[[a2_col]]),
    Z   = as.numeric(sub[[z_col]]),
    N   = as.integer(sub[[n_col]]))
  # Optional columns when present (hardcoded aliases).
  for (slot in c("BETA", "SE", "P", "MAF", "INFO")) {
    src <- intersect(c(slot, tolower(slot),
                       if (slot == "MAF") c("effect_allele_frequency"),
                       if (slot == "P")   c("pvalue", "p")),
                     names(sub))[1L]
    if (!is.na(src))
      mcols(entry_gr)[[slot]] <- as.numeric(sub[[src]])
  }
  entry_gr
}

entries <- lapply(seq_along(studies), function(k)
  buildEntryGr(gwasTsvs[[k]], mappings[[k]], block))

# ----- Construct + QC + save ------------------------------------------------
gss <- GwasSumStats(
  study    = studies,
  entry    = entries,
  genome   = argv$genome,
  ldSketch = ld_handle,
  nCase    = nCase,
  nControl = nControl)
gss_out <- if (argv$skip_qc) {
  message("--skip-qc set; serialising raw GwasSumStats without summaryStatsQc().")
  gss
} else {
  qc_call_args <- c(list(gss,
                          zMismatchQc     = qc_method,
                          impute          = isTRUE(argv$impute),
                          pipCutoffToSkip = argv$pip_cutoff_to_skip,
                          mafCutoff       = argv$maf),
                    if (!is.null(skip_region_vec))
                      list(skipRegion = skip_region_vec) else list(),
                    qc_extra)
  do.call(summaryStatsQc, qc_call_args)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(gss_out, argv$output)
cat(sprintf("Wrote %sGwasSumStats (%d stud%s: %s) over %s to %s\n",
            if (argv$skip_qc) "(skip-QC) " else "QC'd ",
            length(studies), if (length(studies) == 1L) "y" else "ies",
            paste(studies, collapse = ","), argv$ld_block, argv$output))
