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
if (nrow(regions) == 0L)
  stop("No analysis regions: provide --regions and/or --region-name.")

# ---- xQTL weight metadata --------------------------------------------------
if (is.null(argv$xqtl_meta) || !nzchar(argv$xqtl_meta) || !file.exists(argv$xqtl_meta))
  stop("--xqtl-meta is required and must exist: ", argv$xqtl_meta)
xqtlDf <- readMeta(argv$xqtl_meta)
checkCols(xqtlDf, c("region_id", "#chr", "start", "end", "TSS", "original_data"),
          "--xqtl-meta")
xqtlChr <- chromAdd(xqtlDf[["#chr"]])
xqtlTss <- suppressWarnings(as.integer(xqtlDf$TSS))

# ---- per-region: overlapping genes/weights + covering GWAS -----------------
rows <- list()
skipped <- 0L
for (i in seq_len(nrow(regions))) {
  chrom <- regions$chr[[i]]; start <- regions$start[[i]]; stop <- regions$stop[[i]]
  hit <- which(xqtlChr == chrom & !is.na(xqtlTss) &
               xqtlTss >= start & xqtlTss <= stop)
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
    region_id     = sprintf("%s_%d_%d", chrom, start, stop),
    chrom         = chrom, start = start, stop = stop,
    genes         = paste(genes, collapse = ","),
    weight_files  = paste(weightFiles, collapse = ","),
    gwas_studies  = paste(studies, collapse = ","),
    gwas_files    = paste(gfiles, collapse = ","),
    gwas_mappings = paste(gmaps, collapse = ","),
    stringsAsFactors = FALSE)
}
if (skipped > 0L)
  message(sprintf("Skipping %d of %d region(s): no overlapping xQTL weights.",
                  skipped, nrow(regions)))
if (length(rows) == 0L)
  stop("No regions with overlapping xQTL weights; nothing to analyse.")
out <- do.call(rbind, rows)

writeManifest(out, argv$output)
cat(sprintf("Wrote TWAS manifest with %d region(s) to %s\n", nrow(out), argv$output))
