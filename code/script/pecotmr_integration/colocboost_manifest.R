#!/usr/bin/env Rscript
# colocboost_manifest.R
#
# Resolve `colocboost_mnm.ipynb`'s per-gene analysis units into a single
# manifest TSV, so the downstream [colocboost] step can fan out over its
# rows via inline csv.DictReader without any notebook-local Python parsing.
# One row per gene (= one colocboost focal-trait task).
#
# Gene coordinates come from the QtlDataset phenotype manifest (the same
# file passed to qtl_dataset_construct.R), aggregated across its context
# rows. The GWAS analysis window (`ld_block`) defaults to the gene body
# +/- --cis-window; an optional --customized-association-windows BED
# overrides it per gene. GWAS studies whose chromosome matches the gene's
# (a meta `chrom` of 0 matches every chromosome) are grouped onto the
# gene's row as comma-separated lists, ready for one multi-study
# gwas_sumstats_construct.R call.
#
# Inputs:
#   --pheno-manifest   QtlDataset phenotype manifest TSV. Columns:
#                        ID            gene / trait identifier
#                        #chr|chrom    chromosome
#                        start, end    gene-body coordinates
#                      (cond/path/cov_path are ignored here). Multiple
#                      rows per gene (one per context) are aggregated:
#                      start = min(start), end = max(end).
#   --cis-window       bp added on each side of the gene body to form the
#                      default ld_block. Default 1000000.
#   --customized-association-windows
#                      Optional BED (#chr start end ID). When a gene is
#                      present, its row defines ld_block directly,
#                      overriding --cis-window.
#   --region-name      Optional comma-separated gene IDs to restrict to.
#   --region-list      Optional file whose LAST column lists gene IDs to
#                      restrict to (header / comment lines skipped).
#   --gwas-meta        Optional colocboost GWAS meta TSV. Columns:
#                        study_id, chrom, file_path; optional n_sample,
#                        n_case, n_control, ld_meta_data,
#                        column_mapping_file. Relative paths resolve
#                        against the meta file's own directory.
#   --ld-meta          Optional default LD-meta path used when a study's
#                      ld_meta_data is absent/empty.
#   --output           Output manifest TSV path.
#
# Output TSV columns (one row per gene):
#   region_id, gene_id, chr, start, end, ld_block, studies, gwas_tsvs,
#   column_mappings, n_cases, n_controls, ld_meta
# The grouped GWAS columns are empty when no study covers the gene's
# chromosome (xQTL-only coloc).

suppressPackageStartupMessages({
  library(argparser)
})

parser <- arg_parser("Resolve colocboost per-gene analysis units into a manifest TSV")
parser <- add_argument(parser, "--pheno-manifest",
                       help = "QtlDataset phenotype manifest TSV (ID, #chr, start, end)",
                       type = "character")
parser <- add_argument(parser, "--cis-window",
                       help = "bp added on each side of the gene body for the default ld_block",
                       type = "numeric", default = 1000000)
parser <- add_argument(parser, "--customized-association-windows",
                       help = "Optional BED (#chr start end ID) overriding ld_block per gene",
                       type = "character", default = "")
parser <- add_argument(parser, "--region-name",
                       help = "Optional comma-separated gene IDs to restrict to",
                       type = "character", default = "")
parser <- add_argument(parser, "--region-list",
                       help = "Optional file whose last column lists gene IDs to restrict to",
                       type = "character", default = "")
parser <- add_argument(parser, "--gwas-meta",
                       help = "Optional colocboost GWAS meta TSV",
                       type = "character", default = "")
parser <- add_argument(parser, "--ld-meta",
                       help = "Optional default LD-meta path",
                       type = "character", default = "")
parser <- add_argument(parser, "--output",
                       help = "Output manifest TSV path", type = "character")
argv <- parse_args(parser)

# ----- Small helpers --------------------------------------------------------
norm_chr <- function(x) {
  x <- as.character(x)
  ifelse(is.na(x) | !nzchar(x), x,
         ifelse(startsWith(x, "chr"), x, paste0("chr", x)))
}
resolve_against <- function(p, dir) {
  if (is.na(p) || !nzchar(p)) return("")
  if (startsWith(p, "/")) return(p)
  file.path(dir, p)
}
read_tsv <- function(path) {
  read.table(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
             check.names = FALSE, comment.char = "")
}

# ----- Gene coordinates from the phenotype manifest -------------------------
if (is.null(argv$pheno_manifest) || !nzchar(argv$pheno_manifest) ||
    !file.exists(argv$pheno_manifest))
  stop("--pheno-manifest is required and must exist: ", argv$pheno_manifest)
pm <- read_tsv(argv$pheno_manifest)
id_col    <- intersect(c("ID", "gene_id", "phenotype_id"), names(pm))[1L]
chr_col   <- intersect(c("#chr", "chrom", "chr"),          names(pm))[1L]
start_col <- intersect(c("start", "Start"),                names(pm))[1L]
end_col   <- intersect(c("end", "End"),                    names(pm))[1L]
if (any(is.na(c(id_col, chr_col, start_col, end_col))))
  stop("--pheno-manifest needs ID, #chr/chrom, start, end columns (got: ",
       paste(names(pm), collapse = ", "), ").")

genes  <- as.character(pm[[id_col]])
gchr   <- norm_chr(pm[[chr_col]])
gstart <- suppressWarnings(as.integer(pm[[start_col]]))
gend   <- suppressWarnings(as.integer(pm[[end_col]]))
uniqGenes <- unique(genes)
coords <- lapply(uniqGenes, function(g) {
  idx <- which(genes == g)
  list(chr = gchr[idx][[1L]],
       start = min(gstart[idx], na.rm = TRUE),
       end   = max(gend[idx],   na.rm = TRUE))
})
names(coords) <- uniqGenes

# ----- Optional customized association windows ------------------------------
custom <- list()
if (nzchar(argv$customized_association_windows) &&
    argv$customized_association_windows != "." &&
    file.exists(argv$customized_association_windows)) {
  caw <- read.table(argv$customized_association_windows, header = FALSE,
                    sep = "", stringsAsFactors = FALSE, comment.char = "#")
  # columns: chr start end ID
  for (i in seq_len(nrow(caw))) {
    g <- as.character(caw[[4L]][[i]])
    custom[[g]] <- list(chr = norm_chr(caw[[1L]][[i]]),
                        start = as.integer(caw[[2L]][[i]]),
                        end   = as.integer(caw[[3L]][[i]]))
  }
}

# ----- Resolve the gene list ------------------------------------------------
selected <- character(0)
rn <- trimws(strsplit(argv$region_name, ",")[[1L]])
rn <- rn[nzchar(rn)]
if (length(rn) > 0L) {
  selected <- rn
} else if (nzchar(argv$region_list) && argv$region_list != "." &&
           file.exists(argv$region_list)) {
  for (line in readLines(argv$region_list)) {
    line <- trimws(line)
    if (!nzchar(line) || startsWith(line, "#")) next
    parts <- strsplit(line, "\\s+")[[1L]]
    selected <- c(selected, parts[[length(parts)]])
  }
  selected <- unique(selected)
} else {
  selected <- uniqGenes
}

# Coordinate lookup: customized window wins, else gene body +/- cis-window.
cis <- as.integer(argv$cis_window)
gene_window <- function(g) {
  if (!is.null(custom[[g]])) {
    w <- custom[[g]]
    return(list(chr = w$chr, gstart = NA_integer_, gend = NA_integer_,
                wstart = max(w$start, 0L), wend = w$end))
  }
  if (!is.null(coords[[g]])) {
    c0 <- coords[[g]]
    return(list(chr = c0$chr, gstart = c0$start, gend = c0$end,
                wstart = max(c0$start - cis, 0L), wend = c0$end + cis))
  }
  stop("Gene '", g, "' not found in --pheno-manifest or ",
       "--customized-association-windows; cannot determine its window.")
}

# ----- GWAS studies from the meta -------------------------------------------
gwasRows <- list()
if (nzchar(argv$gwas_meta) && argv$gwas_meta != "." &&
    file.exists(argv$gwas_meta)) {
  gm <- read_tsv(argv$gwas_meta)
  req <- c("study_id", "chrom", "file_path")
  miss <- setdiff(req, names(gm))
  if (length(miss) > 0L)
    stop("--gwas-meta missing required column(s): ",
         paste(miss, collapse = ", "), " (got: ",
         paste(names(gm), collapse = ", "), ").")
  metaDir <- dirname(normalizePath(argv$gwas_meta))
  getcol <- function(nm) if (nm %in% names(gm)) as.character(gm[[nm]]) else NULL
  cmCol  <- getcol("column_mapping_file")
  ncCol  <- getcol("n_case")
  nnCol  <- getcol("n_control")
  ldCol  <- getcol("ld_meta_data")
  for (i in seq_len(nrow(gm))) {
    gwasRows[[length(gwasRows) + 1L]] <- list(
      study   = as.character(gm$study_id[[i]]),
      chrom   = as.character(gm$chrom[[i]]),  # "0" = all chromosomes
      tsv     = resolve_against(as.character(gm$file_path[[i]]), metaDir),
      cmap    = if (!is.null(cmCol)) resolve_against(cmCol[[i]], metaDir) else "",
      ncase   = if (!is.null(ncCol)) ncCol[[i]] else "",
      ncontrol= if (!is.null(nnCol)) nnCol[[i]] else "",
      ldmeta  = if (!is.null(ldCol)) resolve_against(ldCol[[i]], metaDir) else "")
  }
}
default_ld <- if (nzchar(argv$ld_meta) && argv$ld_meta != ".") argv$ld_meta else ""

studies_for_chr <- function(chr) {
  Filter(function(r) r$chrom == "0" || norm_chr(r$chrom) == chr, gwasRows)
}

# ----- Build the per-gene manifest ------------------------------------------
rows <- list()
for (g in selected) {
  w <- gene_window(g)
  hits <- studies_for_chr(w$chr)
  joinf <- function(field) paste(vapply(hits, function(r) {
    v <- r[[field]]; if (is.null(v) || is.na(v)) "" else as.character(v)
  }, character(1)), collapse = ",")
  # One LD-meta per region (gwas_sumstats_construct.R takes a single panel):
  # first non-empty study ld_meta, else the --ld-meta default.
  ldset <- unique(Filter(nzchar, vapply(hits, function(r) r$ldmeta, character(1))))
  ldmeta <- if (length(ldset) >= 1L) ldset[[1L]] else default_ld
  if (length(ldset) > 1L)
    message("NOTE: gene '", g, "' has ", length(ldset),
            " distinct study LD-meta files; using the first (",
            ldmeta, ") for the shared region panel.")
  rows[[length(rows) + 1L]] <- data.frame(
    region_id       = gsub("[^A-Za-z0-9_]", "_", g),
    gene_id         = g,
    chr             = w$chr,
    start           = if (is.na(w$gstart)) w$wstart else w$gstart,
    end             = if (is.na(w$gend))   w$wend   else w$gend,
    ld_block        = sprintf("%s:%d-%d", w$chr, w$wstart, w$wend),
    studies         = joinf("study"),
    gwas_tsvs       = joinf("tsv"),
    column_mappings = joinf("cmap"),
    n_cases         = joinf("ncase"),
    n_controls      = joinf("ncontrol"),
    ld_meta         = if (length(hits) > 0L) ldmeta else "",
    stringsAsFactors = FALSE)
}
if (length(rows) == 0L)
  stop("No genes selected; check --region-name / --region-list against the ",
       "phenotype manifest.")
out <- do.call(rbind, rows)

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
write.table(out, file = argv$output, sep = "\t", quote = FALSE,
            row.names = FALSE, na = "")
nWithGwas <- sum(nzchar(out$studies))
cat(sprintf("Wrote colocboost manifest with %d gene(s) (%d with GWAS) to %s\n",
            nrow(out), nWithGwas, argv$output))
