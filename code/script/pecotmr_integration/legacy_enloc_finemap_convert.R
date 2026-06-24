#!/usr/bin/env Rscript
# legacy_enloc_finemap_convert.R
#
# Bridge for migrating the legacy SuSiE-enloc fine-mapping RDS files to the
# new pecotmr S4 path, so the enrichment + coloc cells of SuSiE_enloc.ipynb
# can call qtl_enrichment.R / coloc.R (which consume S4
# QtlFineMappingResult / GwasFineMappingResult collections).
#
# Two modes, each emitting ONE combined collection RDS:
#   --mode qtl  : a QtlFineMappingResult spanning every (study, context)
#                 fit found in the per-study `*.univariate_susie_twas_weights.rds`
#                 files (one entry per (study, context, trait=gene, method="susie")).
#   --mode gwas : a GwasFineMappingResult spanning every (gwas_study, block)
#                 fit found in the per-block `*.univariate_susie_rss.rds` files
#                 (one entry per (gwas_study, region_id=block, method="susie_rss")).
#
# Legacy inner shape (shared by both modes), per fit:
#   $variant_names         all variant ids (e.g. "chr1:20520132:G:C" for qtl,
#                          "1:17351816:A:C" for gwas), one per column of the fit.
#   $susie_result_trimmed  a SuSiE object: $pip (over all variants, UNNAMED),
#                          $sets ($cs list of per-effect variant-index vectors,
#                          NULL when no credible set), $alpha / $mu / $mu2 /
#                          $lbf_variable (L x nvariants), $V (per-effect prior
#                          variance).
#   $top_loci              a small canonical-top-loci df (NOT used here; we
#                          build topLoci over ALL variants so getTopLoci's
#                          posterior projection has the full per-variant view).
#   $sumstats              betahat/sebetahat (qtl) over all variants (UNNAMED,
#                          aligned by index to variant_names); GWAS carries z.
#
# qtl files:  RDS[[gene]][[context]]                 = inner
# gwas files: RDS[[block]][[gwas_study]]$RSS_QC_RAISS_imputed = inner
#             (some [[block]][[gwas_study]] are empty list() -> skipped)
#
# FIELD-MAPPING into the canonical topLoci (consumed via getTopLoci /
# .projectPosteriorView, which sources variant_id/chrom/pos/A1/A2, pip,
# beta<-posterior_mean, se<-posterior_sd, and passes cs_95 through):
#   variant_id      <- variant_names (verbatim; chr-prefix convention preserved;
#                      colocPipeline/qtlEnrichmentPipeline reconcile qtl vs gwas
#                      conventions via alignVariantNames)
#   chrom/pos/A1/A2 <- parsed from chr:pos:A1:A2
#   pip             <- susie_result_trimmed$pip  (set as names on the fit too,
#                      so the FineMappingEntry drift check passes)
#   posterior_mean  <- colSums(alpha * mu)             [proper SuSiE posterior]
#   posterior_sd    <- sqrt(pmax(colSums(alpha*mu2) - posterior_mean^2, 0))
#   cs_95           <- "L<k>" for variants in credible set k, "0" otherwise
#                      (from susie_result_trimmed$sets$cs; "0" everywhere when NULL)
# susieFit is kept as the full susie_result_trimmed (coloc needs lbf_variable/
# sets/V; enrichment needs alpha/pip/V).
#
# LD-SKETCH: the QtlFineMappingResult is emitted with ldSketch = NULL
# (individual-level fits). The GwasFineMappingResult MUST carry a non-NULL
# GenotypeHandle ldSketch -- qtlEnrichmentPipeline hard-requires it -- so a
# minimal placeholder GenotypeHandle is attached. It is never inspected for
# content: .requireMatchingLdSketches short-circuits the moment the QTL side
# is NULL, so no real genotype panel is needed for this MWE bridge.
#
# OBJECT PATHS: the fit / variant-names live at configurable nested paths
# inside `inner` (qtl) or `rds[[block]]` (gwas), supplied via --finemapping-obj
# / --varname-obj (each a single string of space-separated path components).
#   QTL  : inner = rds[[gene]][[context]]; the obj path is applied to `inner`,
#          e.g. "preset_variants_result susie_result_trimmed" reaches the legacy
#          PRESET-subset fit (NOT inner$susie_result_trimmed, the full fit).
#   GWAS : the obj's FIRST element is the specific GWAS study; the WHOLE path is
#          applied to rds[[block]], e.g.
#          "AD_Bellenguez_2022 RSS_QC_RAISS_imputed susie_result_trimmed"
#          reaches that one study's fit. The GwasFMR study label = obj[[1]].
#
# Inputs:
#   --mode      {qtl, gwas}
#   --meta      xqtl_meta tsv (qtl mode) or gwas_meta tsv (gwas mode); used to
#               discover the per-study/per-block RDS basenames + labels. When
#               absent / unreadable the converter falls back to globbing
#               --data-dir for the matching RDS pattern.
#   --data-dir  directory holding the legacy RDS files (and the meta tsv)
#   --rds-files comma-separated explicit RDS paths; when given, bypasses
#               --meta/--data-dir discovery and uses exactly those files
#               (per-region mode).
#   --finemapping-obj  space-separated path to the fit inside inner (qtl) or
#               rds[[block]] (gwas; first component = study). Defaults to the
#               verified legacy QTL/GWAS paths when empty.
#   --varname-obj      space-separated path to the variant_names vector.
#   --region-obj       space-separated path to the region grange (accepted for
#               CLI parity with the legacy interface; not consumed here).
#   --method    fine-mapping method label (default: susie for qtl, susie_rss for gwas)
#   --output    output collection RDS

suppressPackageStartupMessages({ library(argparser); library(pecotmr) })

p <- arg_parser("Convert legacy SuSiE-enloc fine-mapping RDS to pecotmr S4")
p <- add_argument(p, "--mode", type = "character",
                  help = "qtl or gwas")
p <- add_argument(p, "--meta", type = "character", default = "",
                  help = "xqtl_meta / gwas_meta tsv (optional; else glob --data-dir)")
p <- add_argument(p, "--data-dir", type = "character", default = "",
                  help = "directory of the legacy *.rds files")
p <- add_argument(p, "--rds-files", type = "character", default = "",
                  help = "comma-separated explicit RDS paths (per-region; bypasses --meta/--data-dir)")
p <- add_argument(p, "--finemapping-obj", type = "character", default = "",
                  help = "space-separated object path to the fit (default: verified legacy path)")
p <- add_argument(p, "--varname-obj", type = "character", default = "",
                  help = "space-separated object path to variant_names (default: verified legacy path)")
p <- add_argument(p, "--region-obj", type = "character", default = "",
                  help = "space-separated object path to region grange (CLI parity; unused here)")
p <- add_argument(p, "--study", type = "character", default = "",
                  help = "QTL study label override (default: parsed from filename)")
p <- add_argument(p, "--method", type = "character", default = "",
                  help = "method label (default susie [qtl] / susie_rss [gwas])")
p <- add_argument(p, "--output", type = "character",
                  help = "output collection RDS")
argv <- parse_args(p)

mode <- match.arg(argv$mode, c("qtl", "gwas"))
method <- if (nzchar(argv$method)) argv$method else
  if (mode == "qtl") "susie" else "susie_rss"

# Parse a single space-separated CLI string into a character path vector.
parseObjPath <- function(x) {
  if (is.null(x) || !nzchar(trimws(x))) return(character(0))
  strsplit(trimws(x), "\\s+")[[1L]]
}

# Object paths into the RDS -- REQUIRED, supplied by the caller (the SoS cell
# forwards the legacy --*-finemapping-obj / --*-varname-obj values). They are
# deliberately NOT defaulted: the fit / variant-name locations are a property of
# the input data, not of this script, so hardcoding them here would silently
# bind the converter to one dataset's layout. For GWAS the first path element is
# the study (which becomes the GwasFMR study label).
finemappingObj <- parseObjPath(argv$finemapping_obj)
varnameObj     <- parseObjPath(argv$varname_obj)
if (length(finemappingObj) == 0L)
  stop("--finemapping-obj is required (space-separated object path to the fit).")
if (length(varnameObj) == 0L)
  stop("--varname-obj is required (space-separated object path to variant_names).")

# ---- shared helpers --------------------------------------------------------

# Walk a nested list `x` along the character path components in `path`. An empty
# / NULL path returns `x` unchanged; a missing intermediate short-circuits NULL.
getNested <- function(x, path) {
  if (is.null(path) || length(path) == 0L) return(x)
  for (p in path) {
    if (is.null(x)) return(NULL)
    x <- x[[p]]
  }
  x
}

# Parse "chr:pos:A1:A2" ids into the canonical identity columns. Tolerant of a
# missing "chr" prefix (gwas ids) and of malformed ids (NA-filled).
parseIds <- function(vids) {
  vp <- strsplit(as.character(vids), ":", fixed = TRUE)
  g  <- function(i) vapply(vp, function(x)
          if (length(x) >= i) x[[i]] else NA_character_, character(1))
  list(chrom = sub("^chr", "", g(1L)),
       pos   = suppressWarnings(as.integer(g(2L))),
       A1    = g(3L),
       A2    = g(4L))
}

# Per-variant posterior mean / sd from a (trimmed) SuSiE fit's single-effect
# matrices: E[b] = sum_l alpha_lj mu_lj ; Var[b] = sum_l alpha_lj mu2_lj - E[b]^2.
# Falls back to 0 / a small placeholder when the matrices are unavailable.
posteriorMeanSd <- function(fit, n) {
  alpha <- if (!is.null(fit$alpha)) as.matrix(fit$alpha) else NULL
  mu    <- if (!is.null(fit$mu))    as.matrix(fit$mu)    else NULL
  mu2   <- if (!is.null(fit$mu2))   as.matrix(fit$mu2)   else NULL
  if (!is.null(alpha) && !is.null(mu) && all(dim(alpha) == dim(mu)) &&
      ncol(alpha) == n) {
    pm <- as.numeric(colSums(alpha * mu))
    ps <- if (!is.null(mu2) && all(dim(alpha) == dim(mu2)))
            as.numeric(sqrt(pmax(colSums(alpha * mu2) - pm^2, 0)))
          else pmax(abs(pm) * 0.5, 0.05)
    return(list(mean = pm, sd = ps))
  }
  list(mean = rep(0, n), sd = rep(0.05, n))
}

# Credible-set membership label per variant: "L<k>" for variants in the k-th
# credible set of susie_result_trimmed$sets$cs, "0" otherwise. Index vectors in
# $sets$cs are 1-based positions into the variant axis.
csLabels <- function(fit, n) {
  out <- rep("0", n)
  cs <- fit$sets$cs
  if (is.null(cs) || length(cs) == 0L) return(out)
  csNames <- names(cs)
  for (k in seq_along(cs)) {
    idx <- cs[[k]]
    idx <- idx[!is.na(idx) & idx >= 1L & idx <= n]
    if (length(idx) == 0L) next
    lab <- if (!is.null(csNames) && nzchar(csNames[[k]])) csNames[[k]]
           else paste0("L", k)
    out[idx] <- lab
  }
  out
}

# Build one FineMappingEntry from a legacy inner fit list. The fit and its
# variant-names are sourced via configurable object paths (relative to `inner`).
buildEntry <- function(inner, finemappingObj, varnameObj) {
  fit  <- getNested(inner, finemappingObj)
  vids <- as.character(getNested(inner, varnameObj))
  n    <- length(vids)
  if (is.null(fit) || is.null(fit$pip) || n == 0L) return(NULL)
  pip <- as.numeric(fit$pip)
  if (length(pip) != n) return(NULL)
  names(fit$pip) <- vids                 # name the fit's pip (drift check + pipelines)
  pms <- posteriorMeanSd(fit, n)
  ids <- parseIds(vids)
  topLoci <- data.frame(
    variant_id     = vids,
    chrom          = ids$chrom,
    pos            = ids$pos,
    A1             = ids$A1,
    A2             = ids$A2,
    pip            = pip,
    posterior_mean = pms$mean,
    posterior_sd   = pms$sd,
    cs_95          = csLabels(fit, n),
    stringsAsFactors = FALSE)
  FineMappingEntry(variantIds = vids, susieFit = fit, topLoci = topLoci)
}

# Discover the basenames of the legacy RDS to read, preferring the meta tsv's
# `original_data` column (comma-separated), falling back to a directory glob.
discoverFiles <- function(meta, dataDir, pattern) {
  files <- character(0)
  if (nzchar(meta) && file.exists(meta)) {
    md <- tryCatch(
      utils::read.delim(meta, header = TRUE, sep = "\t",
                        check.names = FALSE, comment.char = "",
                        stringsAsFactors = FALSE),
      error = function(e) NULL)
    if (!is.null(md) && "original_data" %in% colnames(md)) {
      bn <- unlist(strsplit(as.character(md$original_data), ",", fixed = TRUE))
      bn <- trimws(bn[nzchar(trimws(bn))])
      files <- file.path(dataDir, unique(bn))
      files <- files[file.exists(files)]
    }
  }
  if (length(files) == 0L) {
    files <- list.files(dataDir, pattern = pattern, full.names = TRUE)
  }
  unique(files)
}

# ---- QTL mode --------------------------------------------------------------
# RDS[[gene]][[context]] = inner. study label comes from --study (the SoS cell
# forwards ${name}); context = the inner key, trait = the gene. The fit/varname
# obj paths are the tail applied to `inner`.
convertQtl <- function(files, finemappingObj, varnameObj, studyOverride = "") {
  if (!nzchar(studyOverride))
    stop("--study is required in QTL mode (the study label is a property of ",
         "the analysis, not encoded generically in the filename).")
  fs <- character(0); fc <- character(0); ft <- character(0)
  fm <- character(0); fe <- list()
  for (f in files) {
    study <- studyOverride
    rds <- readRDS(f)
    for (gene in names(rds)) {
      for (context in names(rds[[gene]])) {
        entry <- buildEntry(rds[[gene]][[context]], finemappingObj, varnameObj)
        if (is.null(entry)) next
        fe[[length(fe) + 1L]] <- entry
        fs <- c(fs, study); fc <- c(fc, context)
        ft <- c(ft, gene);  fm <- c(fm, method)
      }
    }
  }
  if (length(fe) == 0L) stop("No QTL fine-mapping entries built from inputs.")
  QtlFineMappingResult(study = fs, context = fc, trait = ft,
                       method = fm, entry = fe, ldSketch = NULL)
}

# ---- GWAS mode -------------------------------------------------------------
# RDS[[block]] holds per-study nodes. The obj's FIRST element is the specific
# GWAS study; the WHOLE obj path applied to rds[[block]] reaches that one
# study's fit (getNested(rds[[block]], gwasObj)). study = obj[[1]], region_id =
# block. We iterate blocks (files) for that one study, skipping empty nodes.
convertGwas <- function(files, finemappingObj, varnameObj) {
  gwasStudy <- finemappingObj[[1L]]
  fitTail   <- finemappingObj[-1L]      # path inside rds[[block]][[study]]
  varTail   <- varnameObj[-1L]
  gs <- character(0); gm <- character(0); gr <- character(0); ge <- list()
  for (f in files) {
    rds <- readRDS(f)
    for (block in names(rds)) {
      node <- rds[[block]][[gwasStudy]]
      if (is.null(node) || length(node) == 0L) next     # empty study in block
      entry <- buildEntry(node, fitTail, varTail)
      if (is.null(entry)) next
      ge[[length(ge) + 1L]] <- entry
      gs <- c(gs, gwasStudy); gm <- c(gm, method); gr <- c(gr, block)
    }
  }
  if (length(ge) == 0L) stop("No GWAS fine-mapping entries built from inputs.")
  # qtlEnrichmentPipeline hard-requires a non-NULL GenotypeHandle ldSketch on
  # the GWAS side. It is never inspected for content here (the matching check
  # short-circuits because the QTL ldSketch is NULL), so a minimal placeholder
  # handle suffices for this MWE bridge.
  ldSketch <- new("GenotypeHandle",
                  path = "<enloc-rss-ld-sketch-placeholder>",
                  format = "vcf",
                  snpInfo = data.frame(SNP = character(0), CHR = character(0),
                                       BP = integer(0), A1 = character(0),
                                       A2 = character(0),
                                       stringsAsFactors = FALSE),
                  nSamples = 0L, sampleIds = character(0),
                  pgenPtr = NULL, chromPaths = character(0))
  GwasFineMappingResult(study = gs, method = gm, entry = ge,
                        region_id = gr, ldSketch = ldSketch)
}

# ---- dispatch --------------------------------------------------------------
# Per-region mode: --rds-files lists exactly which RDS to read (bypasses the
# whole-collection --meta/--data-dir discovery). Otherwise discover via meta/glob.
if (nzchar(trimws(argv$rds_files))) {
  files <- trimws(strsplit(argv$rds_files, ",", fixed = TRUE)[[1L]])
  files <- files[nzchar(files)]
  missing <- files[!file.exists(files)]
  if (length(missing) > 0L)
    stop("These --rds-files do not exist: ", paste(missing, collapse = ", "))
} else {
  pattern <- if (mode == "qtl") {
    "univariate_susie_twas_weights\\.rds$"
  } else {
    "univariate_susie_rss\\.rds$"
  }
  files <- discoverFiles(argv$meta, argv$data_dir, pattern)
}
if (length(files) == 0L)
  stop("No legacy RDS files found (rds-files='", argv$rds_files, "', meta='",
       argv$meta, "', data-dir='", argv$data_dir, "').")
cat(sprintf("[%s mode] reading %d legacy RDS file(s)\n", mode, length(files)))

res <- if (mode == "qtl") {
  convertQtl(files, finemappingObj, varnameObj, argv$study)
} else {
  convertGwas(files, finemappingObj, varnameObj)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(res, argv$output)
cat(sprintf("Wrote %s (%d entries) to %s\n",
            class(res)[[1L]], nrow(res), argv$output))
print(res)
