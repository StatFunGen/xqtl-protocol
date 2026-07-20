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
  library(jsonlite)
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
parser <- add_argument(parser, "--ld-sketch",
                       help = paste("LD reference panel (ldSketch): a genome-wide genotype",
                                    "prefix/path (.bed/.pgen/.vcf[.gz]/.bcf/.gds), OR a",
                                    "per-chromosome mapping file with a chromosome column",
                                    "and a 'path' column - one genotype payload per",
                                    "chromosome (other columns, e.g. start/end, are",
                                    "ignored). No sub-chromosomal block selection."),
                       type = "character", default = "")
parser <- add_argument(parser, "--ld-meta",
                       help = "Deprecated alias for --ld-sketch (kept for backward compatibility).",
                       type = "character", default = "")
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
parser <- add_argument(parser, "--allele-flip-kriging",
                       help = "Enable kriging allele-flip QC (summaryStatsQc alleleFlipKriging): sign-flip allele-switched z (logLR>2 & |z|>2) in place, retaining the variants. Off by default.",
                       flag = TRUE)
parser <- add_argument(parser, "--effective-n",
                       help = "TRUE/FALSE. For case/control GWAS (n_case + n_control columns), use the effective sample size 4/(1/n_case + 1/n_control) as N (summaryStatsQc effectiveN). Default TRUE; FALSE keeps the raw N column (or total when absent).",
                       type = "character", default = "TRUE")
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
                     "pipCutoffToSkip", "alleleFlipKriging", "effectiveN"))
if (length(clash) > 0L)
  stop("--qc-args sets ", paste(clash, collapse = ", "),
       " which is also controlled by a dedicated flag (--qc-method / ",
       "--impute / --maf / --skip-region / --pip-cutoff-to-skip / ",
       "--allele-flip-kriging / --effective-n). Pass it via the dedicated flag.")

# --skip-region: comma-separated chr:start-end windows -> character vector
# (NULL when unset so summaryStatsQc's skipRegion default applies).
skip_region_vec <- splitCsv(argv$skip_region)
if (length(skip_region_vec) == 0L) skip_region_vec <- NULL

qc_method <- match.arg(argv$qc_method, c("none", "slalom", "dentist"))

# ----- Resolve the LD sketch spec (genome-wide OR per-chromosome) ------------
# LD sketches are never region/block-sharded: they are either a single
# genome-wide genotype file/prefix or one genotype file per chromosome. The raw
# --ld-sketch spec is handed straight to the loader's `ldSketch`, which sniffs a
# genotype path/prefix (-> genome-wide handle) versus a per-chromosome mapping
# file (-> genoMeta handle). In the mapping file the chromosome and path columns
# are matched by NAME (#chr/#chrom/chr/chrom + path/payload/prefix/genotype), so
# extra columns (e.g. legacy start/end) are tolerated, and each chromosome must
# map to exactly one payload (no sub-chromosomal blocks).
# The same ld-meta table may ALSO be cTWAS's LD-block grid (one row per block,
# many rows sharing a chromosome's single genotype panel). The sketch only cares
# about the genotype reference, so collapse a block-sharded table to its unique
# per-chromosome (chrom -> panel) mapping before handing it to the loader: the
# loader stays strict (one payload per chromosome) and the notebook can pass the
# block grid to both the block enumerator and the sketch with no separate
# fixture. A genotype path/prefix, or a table that is already one row per
# chromosome, is returned unchanged (the latter as an equivalent named vector).
.collapseSketchMeta <- function(spec) {
  if (!(is.character(spec) && length(spec) == 1L)) return(spec)
  lower <- tolower(spec)
  if (grepl("\\.(pgen|bed|vcf(\\.b?gz)?|bcf|gds)$", lower) ||
      file.exists(paste0(spec, ".pgen")) || file.exists(paste0(spec, ".bed")))
    return(spec)                                    # genotype file/prefix
  if (!file.exists(spec) || dir.exists(spec)) return(spec)
  meta <- tryCatch(
    read.table(spec, header = TRUE, sep = "", comment.char = "",
               stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) NULL)
  if (is.null(meta) || ncol(meta) < 2L) return(spec)
  chromCol <- intersect(c("#chr", "#chrom", "chr", "chrom"), names(meta))[1L]
  pathCol  <- intersect(c("path", "payload", "prefix", "genotype"), names(meta))[1L]
  if (is.na(chromCol) || is.na(pathCol)) return(spec)   # not a chrom->path table
  base  <- dirname(normalizePath(spec))
  chrom <- as.character(meta[[chromCol]])
  pth   <- as.character(meta[[pathCol]])
  pth <- vapply(pth, function(p)
    if (grepl("^(/|[A-Za-z]:)", p) || file.exists(p)) p else file.path(base, p),
    character(1), USE.NAMES = FALSE)
  keep  <- !duplicated(paste(chrom, pth, sep = "\t"))
  chrom <- chrom[keep]; pth <- pth[keep]
  if (anyDuplicated(chrom))
    stop("--ld-sketch '", spec, "': chromosome(s) ",
         paste(unique(chrom[duplicated(chrom)]), collapse = ", "),
         " map to multiple genotype payloads; each chromosome must reference ",
         "exactly one LD panel.")
  # A single genotype panel (one chromosome / one shared payload) is returned as
  # the bare prefix so the GenotypeHandle keeps a real on-disk @path -- cTWAS's
  # .ctwasLdPanelKey resolves the LD file from it. Only a genuinely multi-panel
  # per-chromosome reference needs the named chrom->path vector.
  if (length(pth) == 1L) return(unname(pth))
  stats::setNames(pth, chrom)
}
ld_sketch_spec <- if (nzchar(argv$ld_sketch)) argv$ld_sketch else argv$ld_meta
if (!nzchar(ld_sketch_spec))
  stop("--ld-sketch is required (the LD reference panel for the ldSketch).")
ld_sketch_spec <- .collapseSketchMeta(ld_sketch_spec)

# ----- Build the GwasSumStats via pecotmr's manifest loader -----------------
# Assemble an in-memory one-row-per-study manifest and hand it to
# loadGwasSumStatsFromManifest(), which reads each sumstats file, resolves the
# columns (its built-in aliases match the hardcoded set this script used, plus
# per-study columnMapping YAML), reconciles chr-prefix conventions, restricts
# to `region`, and constructs the GwasSumStats. The raw LD-sketch spec is passed
# through as `ldSketch` (a genotype path/prefix -> genome-wide handle, or a
# per-chromosome mapping file -> genoMeta handle; resolved inside the loader).
#
# NOTE: region restriction of delimited-text sumstats now requires a
# `<path>.tbi` tabix sidecar; a non-indexed TSV is read whole (with a warning).
# The pipeline's GWAS sumstats are bgzipped + tabix-indexed.
manifest_df <- data.frame(
  study        = studies,
  sumStatsPath = gwasTsvs,
  stringsAsFactors = FALSE)
if (any(nzchar(mappings)))
  manifest_df$columnMapping <- ifelse(nzchar(mappings), mappings, NA_character_)
if (!is.null(nCase))    manifest_df$nCase    <- nCase
if (!is.null(nControl)) manifest_df$nControl <- nControl

gss <- loadGwasSumStatsFromManifest(
  manifest = manifest_df,
  genome   = argv$genome,
  ldSketch = ld_sketch_spec,
  region   = argv$ld_block)
gss_out <- if (argv$skip_qc) {
  message("--skip-qc set; serialising raw GwasSumStats without summaryStatsQc().")
  gss
} else {
  effN <- as.logical(argv$effective_n)
  if (is.na(effN))
    stop("--effective-n must be TRUE or FALSE (got: ", argv$effective_n, ")")
  qc_call_args <- c(list(gss,
                          zMismatchQc     = qc_method,
                          impute          = isTRUE(argv$impute),
                          pipCutoffToSkip = argv$pip_cutoff_to_skip,
                          mafCutoff       = argv$maf,
                          effectiveN      = effN),
                    if (isTRUE(argv$allele_flip_kriging))
                      list(alleleFlipKriging = TRUE) else list(),
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

# ----- Per-entry QC diagnostics summary (to the step log) -------------------
# Surface the qcInfo audit so the SoS log shows what QC did, not just the path.
if (!argv$skip_qc) {
  audit <- tryCatch(getQcInfo(gss_out)$entryAudit, error = function(e) NULL)
  for (i in seq_along(audit)) {
    a <- audit[[i]]
    if (is.null(a)) next
    study_i <- if (!is.null(names(audit))) names(audit)[i] else studies[[i]]
    cf <- a$contentFilters
    seg <- c(
      if (!is.null(a$variantsIn) && !is.null(a$variantsOut))
        sprintf("variants %s->%s", a$variantsIn, a$variantsOut),
      if (!is.null(a$nSource) && !is.na(a$nSource)) paste0("N=", a$nSource),
      if (!is.null(cf$mafDropped) && cf$mafDropped > 0L) paste0("maf-drop ", cf$mafDropped),
      if (!is.null(cf$nDropped)   && cf$nDropped   > 0L) paste0("N-drop ", cf$nDropped),
      if (!is.null(a$krigingFlipped)) paste0("kriging-flip ", a$krigingFlipped),
      if (!is.null(a$ldMismatchOutliersDropped) && a$ldMismatchOutliersDropped > 0L)
        paste0("mismatch-drop ", a$ldMismatchOutliersDropped),
      if (!is.null(a$raissImputedVariants) && a$raissImputedVariants > 0L)
        paste0("imputed ", a$raissImputedVariants),
      if (isTRUE(a$pipScreenSkipped)) "PIP-screen SKIPPED")
    cat(sprintf("[QC %s] %s\n", study_i, paste(seg, collapse = " | ")))
  }
}
