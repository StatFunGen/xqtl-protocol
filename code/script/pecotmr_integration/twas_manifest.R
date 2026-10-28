#!/usr/bin/env Rscript
# twas_manifest.R
#
# Resolve twas_ctwas.ipynb's per-region TWAS analysis units into a manifest TSV,
# so the [twas] / [ctwas] / [quantile_twas] steps can fan out over its rows with
# inline csv.DictReader (no notebook-local Python / pandas). This replaces the
# in-notebook get_analysis_regions / extract_regional_data pandas machinery.
#
# One row per analysis region that carries >=1 overlapping xQTL weight. A
# "region" is a coarse genomic window (an LD-reference block, or a
# region_name entry); the [twas] step builds ONE GwasSumStats per region and
# reuses it across the region's genes (a fan-out convenience, not cTWAS
# placement -- gene -> home-LD-block placement is done inside pecotmr's
# assembleCtwasInputs from the weight `region` provenance).
#
# Gene <-> region mapping here is the legacy TSS-overlap binning
# (extract_regional_data): an xQTL weight belongs to a region when it is on the
# region's chromosome and its TSS falls within [start, stop]. Meta-referenced
# file paths are resolved with the legacy adapt_file_path rule (exists as given
# -> basename in cwd -> basename in the meta's dir -> meta_dir/path).
#
# NOTE: no data-layout path is hardcoded; every path comes from arguments or is
# resolved relative to the meta file that referenced it.
#
# Inputs:
#   --gwas-meta   GWAS metadata TSV (study_id, chrom, file_path[, column_mapping_file]).
#                 chrom 0 expands to chromosomes 1..22.
#   --xqtl-meta   xQTL weight metadata TSV (region_id, #chr, start, end, TSS,
#                 original_data[, contexts]); original_data is a comma-separated
#                 list of weight RDS paths; region_id is the gene name.
#   --regions     Optional BED-like TSV of analysis regions (chr, start, stop).
#   --region-name Optional comma-separated "chr_start_stop" regions. When given,
#                 these REPLACE --regions (matching the legacy behaviour).
#   --gwas-name / --gwas-data / --column-mapping  Optional parallel vectors of
#                 inline GWAS studies not listed in --gwas-meta (study name,
#                 sumstats path, optional column-mapping path). Applied to every
#                 region (covering all chromosomes).
#   --output      Output manifest TSV path.
#
# Output columns (one row per region): region_id, chrom, start, stop, genes,
#   weight_files, gwas_studies, gwas_files, gwas_mappings. The comma-separated
#   `genes` and `weight_files` are position-parallel (a gene repeats once per
#   weight file); `gwas_studies`, `gwas_files`, `gwas_mappings` are likewise
#   position-parallel.

suppressPackageStartupMessages({
  library(argparser)
})

p <- arg_parser("Resolve per-region TWAS analysis units into a manifest TSV")
p <- add_argument(p, "--gwas-meta", type = "character",
                  help = "GWAS metadata TSV (study_id, chrom, file_path)")
p <- add_argument(p, "--xqtl-meta", type = "character",
                  help = "xQTL weight metadata TSV (region_id, #chr, start, end, TSS, original_data)")
p <- add_argument(p, "--regions", type = "character", default = "",
                  help = "Optional BED-like TSV of analysis regions (chr, start, stop)")
p <- add_argument(p, "--region-name", type = "character", default = "",
                  help = "Optional comma-separated 'chr_start_stop' regions (replace --regions)")
p <- add_argument(p, "--gwas-name", type = "character", nargs = Inf,
                  default = NA_character_, help = "Optional inline GWAS study names")
p <- add_argument(p, "--gwas-data", type = "character", nargs = Inf,
                  default = NA_character_, help = "Optional inline GWAS sumstats paths")
p <- add_argument(p, "--column-mapping", type = "character", nargs = Inf,
                  default = NA_character_, help = "Optional inline GWAS column-mapping paths")
p <- add_argument(p, "--unit", type = "character", default = "region",
  help = "Analysis unit: 'region' (default; genes grouped by --regions) or 'gene' (one unit per gene spanning its weight window, no LD blocks; twas_ctwas.ipynb passes this by default)")
p <- add_argument(p, "--min-cv-rsq", type = "numeric", default = NA,
  help = "gene unit only: skip genes where no context/method reaches this CV r2 (with --max-cv-pval); NA = no prefilter")
p <- add_argument(p, "--max-cv-pval", type = "numeric", default = NA,
  help = "gene unit only: CV p-value that must also be passed (Inf = no p-value check)")
p <- add_argument(p, "--threads", type = "integer", default = 1L,
  help = "Cores for reading weight files in the prefilter")
p <- add_argument(p, "--output", type = "character", help = "Output manifest TSV path")
argv <- parse_args(p)

.d <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]))
source(file.path(.d, "manifest_common.R"))

checkCols <- function(df, required, what) {
  missing <- setdiff(required, names(df))
  if (length(missing) > 0L)
    stop(what, " missing required column(s): ", paste(missing, collapse = ", "),
         " (got: ", paste(names(df), collapse = ", "), ")")
}

splitVec <- function(x) {
  # normalise an nargs=Inf argument (NA / length-0 / whitespace) to a clean vector
  if (length(x) == 0L) return(character(0))
  x <- x[!is.na(x)]
  x <- trimws(x)
  x[nzchar(x)]
}

# ---- inline GWAS vectors (retain the legacy CLI) ---------------------------
gwasName <- splitVec(argv$gwas_name)
gwasData <- splitVec(argv$gwas_data)
colMap   <- splitVec(argv$column_mapping)
if (length(gwasName) != length(gwasData))
  stop("--gwas-name and --gwas-data must have equal length")
if (length(colMap) > 0L && length(colMap) != length(gwasName))
  stop("--column-mapping, if given, must match --gwas-name / --gwas-data length")

# ---- GWAS metadata: (study, chrom) -> [file, mapping] ----------------------
if (is.null(argv$gwas_meta) || !nzchar(argv$gwas_meta) || !file.exists(argv$gwas_meta))
  stop("--gwas-meta is required and must exist: ", argv$gwas_meta)
gwasDf <- readMeta(argv$gwas_meta)
checkCols(gwasDf, c("study_id", "chrom", "file_path"), "--gwas-meta")
hasMapCol <- "column_mapping_file" %in% names(gwasDf)

# gwasByChrom[[chrom]] = data.frame(study, file, mapping) of studies on that chrom
gwasByChrom <- list()
addGwas <- function(chrom, study, file, mapping) {
  chrom <- chromAdd(chrom)
  row <- data.frame(study = study, file = file,
                    mapping = if (is.na(mapping) || is.null(mapping)) "" else mapping,
                    stringsAsFactors = FALSE)
  gwasByChrom[[chrom]] <<- rbind(gwasByChrom[[chrom]], row)
}
for (i in seq_len(nrow(gwasDf))) {
  filePath <- adaptFilePath(gwasDf$file_path[[i]], argv$gwas_meta)
  mapping <- if (hasMapCol && nzchar(as.character(gwasDf$column_mapping_file[[i]])))
    adaptFilePath(gwasDf$column_mapping_file[[i]], argv$gwas_meta) else NA_character_
  chromVal <- suppressWarnings(as.integer(gwasDf$chrom[[i]]))
  chroms <- if (!is.na(chromVal) && chromVal == 0L) 1:22 else gwasDf$chrom[[i]]
  for (ch in chroms) addGwas(ch, gwasDf$study_id[[i]], filePath, mapping)
}

# ---- analysis regions: --regions BED and/or --region-name ------------------
regions <- data.frame(chr = character(0), start = integer(0), stop = integer(0),
                      stringsAsFactors = FALSE)
if (nzchar(argv$regions) && argv$regions != "." && file.exists(argv$regions)) {
  rg <- readMeta(argv$regions)
  names(rg) <- trimws(names(rg))
  checkCols(rg, c("chr", "start", "stop"), "--regions")
  regions <- data.frame(chr = chromAdd(trimws(rg$chr)),
                        start = as.integer(rg$start), stop = as.integer(rg$stop),
                        stringsAsFactors = FALSE)
}
rn <- splitVec(strsplit(argv$region_name, ",")[[1L]])
if (length(rn) > 0L) {
  # region_name entries "chr_start_stop" REPLACE the file regions (legacy behaviour)
  parts <- strsplit(rn, "_")
  regions <- data.frame(
    chr   = chromAdd(vapply(parts, `[`, character(1), 1L)),
    start = as.integer(vapply(parts, `[`, character(1), 2L)),
    stop  = as.integer(vapply(parts, `[`, character(1), 3L)),
    stringsAsFactors = FALSE)
}
regions <- unique(regions)
if (!argv$unit %in% c("gene", "region"))
  stop("--unit must be 'gene' or 'region', got: ", argv$unit)
if (argv$unit == "region" && nrow(regions) == 0L)
  stop("--unit region needs --regions and/or --region-name.")

# ---- xQTL weight metadata --------------------------------------------------
if (is.null(argv$xqtl_meta) || !nzchar(argv$xqtl_meta) || !file.exists(argv$xqtl_meta))
  stop("--xqtl-meta is required and must exist: ", argv$xqtl_meta)
xqtlDf <- readMeta(argv$xqtl_meta)
checkCols(xqtlDf, c("region_id", "#chr", "start", "end", "TSS", "original_data"),
          "--xqtl-meta")
xqtlChr <- chromAdd(xqtlDf[["#chr"]])
xqtlTss <- suppressWarnings(as.integer(xqtlDf$TSS))

# ---- per-region: overlapping genes/weights + covering GWAS -----------------
# Analysis units. gene: one unit per xQTL meta row, spanning that
# gene's weight window (meta start..end), so every weight variant is scored
# with GWAS and LD read from the per-chromosome files; --regions /
# --region-name, if given, only filter genes by TSS. region (default): genes
# binned into --regions by TSS, one unit per region.
inRegions <- function(j) nrow(regions) == 0L ||
  any(regions$chr == xqtlChr[[j]] & !is.na(xqtlTss[[j]]) &
      regions$start <= xqtlTss[[j]] & regions$stop >= xqtlTss[[j]])
units <- if (argv$unit == "gene") {
  keep <- Filter(inRegions, seq_len(nrow(xqtlDf)))
  # Optional prefilter: drop genes that can have no imputable model, so the
  # twas step does not build their GWAS input. Lenient on purpose: a gene is
  # kept if ANY context/method row passes, a superset of what pecotmr's own
  # best-method selection can keep; pecotmr still applies the exact rule.
  if (!is.na(argv$min_cv_rsq)) {
    maxP <- if (is.na(argv$max_cv_pval)) Inf else argv$max_cv_pval
    passes <- function(j) {
      f <- trimws(strsplit(xqtlDf$original_data[[j]], ",")[[1L]])
      f <- vapply(f[nzchar(f)], adaptFilePath, character(1), referenceFile = argv$xqtl_meta, USE.NAMES = FALSE)
      for (x in f) {
        w <- readRDS(x)
      # Legacy list-layout weights carry no per-row cvResult column: keep the
      # gene and let pecotmr apply the CV rule.
      if (!methods::is(w, "TwasWeights")) return(TRUE)
      cv <- as.list(S4Vectors::mcols(w)$cvResult)
        for (r in cv) {
          v <- r$metrics
          if (is.null(v) || is.na(v["rsq"])) return(TRUE)  # no CV info: let pecotmr decide
          pv <- if (!is.na(v["adj_rsq_pval"])) v["adj_rsq_pval"] else v["pval"]
          if (v["rsq"] >= argv$min_cv_rsq && (is.infinite(maxP) || (!is.na(pv) && pv < maxP))) return(TRUE)
        }
      }
      FALSE
    }
    ok <- unlist(parallel::mclapply(keep, passes, mc.cores = max(1L, argv$threads)))
    message(sprintf("CV prefilter (r2 >= %g, p < %g): kept %d of %d genes.", argv$min_cv_rsq, maxP, sum(ok), length(keep)))
    keep <- keep[ok]
  }
  lapply(keep, function(j) list(chrom = xqtlChr[[j]],
    start = as.integer(xqtlDf$start[[j]]), stop = as.integer(xqtlDf$end[[j]]), hit = j))
} else {
  lapply(seq_len(nrow(regions)), function(i) {
    chrom <- regions$chr[[i]]; start <- regions$start[[i]]; stop <- regions$stop[[i]]
    list(chrom = chrom, start = start, stop = stop,
         hit = which(xqtlChr == chrom & !is.na(xqtlTss) & xqtlTss >= start & xqtlTss <= stop))
  })
}
rows <- list()
skipped <- 0L
for (u in units) {
  chrom <- u$chrom; start <- u$start; stop <- u$stop; hit <- u$hit
  genes <- character(0); weightFiles <- character(0)
  for (j in hit) {
    files <- trimws(strsplit(xqtlDf$original_data[[j]], ",")[[1L]])
    files <- files[nzchar(files)]
    files <- vapply(files, adaptFilePath, character(1), referenceFile = argv$xqtl_meta,
                    USE.NAMES = FALSE)
    weightFiles <- c(weightFiles, files)
    genes <- c(genes, rep(as.character(xqtlDf$region_id[[j]]), length(files)))
  }
  if (length(weightFiles) == 0L) { skipped <- skipped + 1L; next }

  # GWAS studies covering this chromosome (inline vectors apply to every region)
  g <- gwasByChrom[[chrom]]
  studies  <- character(0); gfiles <- character(0); gmaps <- character(0)
  if (length(gwasName) > 0L) {
    studies <- gwasName
    gfiles  <- vapply(gwasData, adaptFilePath, character(1),
                      referenceFile = argv$gwas_meta, USE.NAMES = FALSE)
    gmaps   <- if (length(colMap) > 0L)
      vapply(colMap, adaptFilePath, character(1),
             referenceFile = argv$gwas_meta, USE.NAMES = FALSE) else rep("", length(gwasName))
  }
  if (!is.null(g)) {
    studies <- c(studies, g$study); gfiles <- c(gfiles, g$file); gmaps <- c(gmaps, g$mapping)
  }

  rows[[length(rows) + 1L]] <- data.frame(
    region_id     = if (argv$unit == "gene") genes[[1L]] else sprintf("%s_%d_%d", chrom, start, stop),
    chrom         = chrom, start = start, stop = stop,
    genes         = paste(genes, collapse = ","),
    weight_files  = paste(weightFiles, collapse = ","),
    gwas_studies  = paste(studies, collapse = ","),
    gwas_files    = paste(gfiles, collapse = ","),
    gwas_mappings = paste(gmaps, collapse = ","),
    stringsAsFactors = FALSE)
}
if (skipped > 0L)
  message(sprintf("Skipping %d of %d unit(s): no overlapping xQTL weights.",
                  skipped, length(units)))
if (length(rows) == 0L)
  stop("No regions with overlapping xQTL weights; nothing to analyse.")
out <- do.call(rbind, rows)

writeManifest(out, argv$output)
cat(sprintf("Wrote TWAS manifest with %d region(s) to %s\n", nrow(out), argv$output))
