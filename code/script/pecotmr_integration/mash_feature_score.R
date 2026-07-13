#!/usr/bin/env Rscript
# mash_feature_score.R
#
# Feature scores from per-region posterior-contrast results -- unified backing
# for the four feature-score steps of mash_posterior.ipynb. Each --method fixes
# one scorer (all implemented in pecotmr, meta via metafor):
#
#   --method meta       calculateFeatureScores  (deviation-contrast RE meta -> Z)
#   --method nsig       nSignificantScore       (fraction of sig deviation contrasts)
#   --method pval_pair  metaAnalysisPerCondition (pairwise-contrast RE meta -> p)
#   --method finemap    scoreFromCs             (credible-set-based score)
#
# One or more --contrast RDS files (from mash_posterior_contrast.R) are scored
# and stacked into a long table (gene, condition, [contrast], score, scoreType).
# Optional LD pruning of the contrast variants uses pecotmr::ldPruneByCorrelation
# (correlation-based; needs a --genotype file + --region), NOT external plink.
#
# Inputs:
#   --method            meta | nsig | pval_pair | finemap.
#   --contrast f1 [...]  per-region contrast RDS files.
#   --gene-ids id,...    gene labels (one per --contrast; default: file basename).
#   --p-cutoff X         nsig significance cutoff. Default 1e-5.
#   --meta-method M      meta / pval_pair estimator (metafor). Default REML.
#   --se-cutoff X        pval_pair SE cutoff. Default 1e-3.
#   --fine-mapping f     finemap: table with cs_order / pip / variants columns.
#   --conditions c,...   finemap: conditions to score (default: all contexts).
#   --genotype f         optional PLINK/VCF/GDS prefix for LD pruning.
#   --region chr:s-e     region for --genotype extraction.
#   --cor-threshold X    LD-prune |correlation| threshold. Default 0.45 (~r2 0.2).
#   --output <RDS>       output long-format score table.

suppressPackageStartupMessages({
  library(argparser)
  library(pecotmr)
})

p <- arg_parser("MASH feature scores from posterior contrasts")
p <- add_argument(p, "--method", type = "character",
                  help = "meta | nsig | pval_pair | finemap")
p <- add_argument(p, "--contrast", type = "character", nargs = Inf,
                  help = "per-region contrast RDS files")
p <- add_argument(p, "--gene-ids", type = "character", default = "",
                  help = "comma-separated gene labels (one per --contrast)")
p <- add_argument(p, "--p-cutoff", type = "numeric", default = 1e-5,
                  help = "nsig significance cutoff")
p <- add_argument(p, "--meta-method", type = "character", default = "REML",
                  help = "meta / pval_pair estimator (metafor)")
p <- add_argument(p, "--se-cutoff", type = "numeric", default = 1e-3,
                  help = "pval_pair SE cutoff")
p <- add_argument(p, "--fine-mapping", type = "character", default = "",
                  help = "finemap: table with cs_order/pip/variants")
p <- add_argument(p, "--conditions", type = "character", default = "",
                  help = "finemap: conditions to score")
p <- add_argument(p, "--genotype", type = "character", default = "",
                  help = "optional genotype prefix for LD pruning")
p <- add_argument(p, "--region", type = "character", default = "",
                  help = "region for --genotype extraction")
p <- add_argument(p, "--cor-threshold", type = "numeric", default = 0.45,
                  help = "LD-prune |correlation| threshold")
p <- add_argument(p, "--output", type = "character", help = "output score RDS")
argv <- parse_args(p)

splitCsv <- function(x) if (nzchar(x)) trimws(strsplit(x, ",", fixed = TRUE)[[1L]]) else character(0)

files <- as.character(argv$contrast)
if (length(files) == 0L) stop("--contrast requires at least one RDS file.")
geneIds <- splitCsv(argv$gene_ids)
if (length(geneIds) == 0L) geneIds <- tools::file_path_sans_ext(basename(files))
if (length(geneIds) != length(files))
  stop("--gene-ids length must match --contrast.")

# Optional LD pruning: keep the contrast rows whose variants survive
# ldPruneByCorrelation on their genotype dosages. No-op without --genotype.
ldPruneContrast <- function(cr) {
  if (!nzchar(argv$genotype) || !nzchar(argv$region)) return(cr)
  X <- loadGenotypeRegion(genotype = argv$genotype, region = argv$region)
  # orient SNPs as columns; keep only the contrast variants that are present
  if (is.null(colnames(X)) && !is.null(rownames(X))) X <- t(X)
  common <- intersect(rownames(cr), colnames(X))
  if (length(common) < 2L) return(cr)
  keep <- ldPruneByCorrelation(X[, common, drop = FALSE],
                               corThres = argv$cor_threshold)$X.new
  cr[colnames(keep), , drop = FALSE]
}

scoreOne <- function(file, gene) {
  cr <- ldPruneContrast(as.data.frame(readRDS(file)))
  if (argv$method == "meta") {
    fs <- calculateFeatureScores(cr, metaMethod = argv$meta_method)
    if (nrow(fs) == 0L) return(NULL)
    data.frame(gene = gene, condition = fs$condition, contrast = NA_character_,
               score = fs$zScore, scoreType = "meta_z", stringsAsFactors = FALSE)
  } else if (argv$method == "nsig") {
    ns <- nSignificantScore(cr, pCutoff = argv$p_cutoff)
    if (nrow(ns) == 0L) return(NULL)
    data.frame(gene = gene, condition = ns$condition, contrast = NA_character_,
               score = ns$ratio, scoreType = "nsig_ratio", stringsAsFactors = FALSE)
  } else if (argv$method == "pval_pair") {
    eff <- as.matrix(cr[, grep("mean_contrast.*_vs_", names(cr)), drop = FALSE])
    se  <- as.matrix(cr[, grep("se_contrast.*_vs_",   names(cr)), drop = FALSE])
    colnames(se) <- colnames(eff)  # metaAnalysisPerCondition needs matching names
    mp <- metaAnalysisPerCondition(eff, se, seCutoff = argv$se_cutoff,
                                   metaMethod = argv$meta_method)
    if (nrow(mp) == 0L) return(NULL)
    data.frame(gene = gene, condition = mp$condition, contrast = mp$contrast,
               score = mp$meta_pvalue, scoreType = "pval_pair", stringsAsFactors = FALSE)
  } else if (argv$method == "finemap") {
    fm <- as.data.frame(readRDS(argv$fine_mapping))
    conds <- splitCsv(argv$conditions)
    if (length(conds) == 0L)
      conds <- unique(sub("_deviation$", "",
                          sub("^mean_contrast_", "",
                              grep("mean_contrast.*deviation", names(cr), value = TRUE))))
    do.call(rbind, lapply(conds, function(cond) {
      data.frame(gene = gene, condition = cond, contrast = NA_character_,
                 score = scoreFromCs(fm, cr, cond), scoreType = "finemap",
                 stringsAsFactors = FALSE)
    }))
  } else {
    stop("--method must be one of meta | nsig | pval_pair | finemap.")
  }
}

out <- do.call(rbind, Filter(Negate(is.null),
                             Map(scoreOne, files, geneIds)))
if (is.null(out)) out <- data.frame(gene = character(0), condition = character(0),
  contrast = character(0), score = numeric(0), scoreType = character(0))

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(out, argv$output, compress = "xz")
cat(sprintf("Wrote %s feature scores (%d rows over %d region(s)) to %s\n",
            argv$method, nrow(out), length(files), argv$output))
