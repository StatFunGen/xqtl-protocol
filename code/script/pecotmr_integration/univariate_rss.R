#!/usr/bin/env Rscript
# ============================================================
# univariate_rss.R
# Mirrors: pipeline/rss_analysis.ipynb [univariate_rss]
#
# Allele QC, optional RAISS imputation, and RSS-based fine-mapping
# (SuSiE-RSS / single-effect / BCR) for one or more GWAS studies
# over a single genomic region.
#
# Usage:
#   Rscript univariate_rss.R --ld-meta-data ld.tsv \
#       --studies study1,study2 \
#       --sumstat-paths s1.tsv,s2.tsv \
#       --column-file-paths None,None \
#       --n-samples 0,0 --n-cases 0,0 --n-controls 0,0 \
#       --ld-meta-data-paths None,None \
#       --region chr22_49355984-50799822 \
#       --qc-method slalom --impute \
#       --finemapping-method susie_rss \
#       --output out.rds
# ============================================================

suppressPackageStartupMessages({
  library(optparse)
  library(pecotmr)
})

# --------------- CLI ---------------

opt_list <- list(
  # required
  make_option("--ld-meta-data",       type = "character", default = NULL,
              help = "Global LD metadata TSV path"),
  make_option("--studies",            type = "character", default = NULL,
              help = "Comma-separated study names"),
  make_option("--sumstat-paths",      type = "character", default = NULL,
              help = "Comma-separated sumstat file paths"),
  make_option("--column-file-paths",  type = "character", default = NULL,
              help = "Comma-separated column-mapping file paths (use 'None' to omit)"),
  make_option("--n-samples",          type = "character", default = "0",
              help = "Comma-separated sample sizes (0 = read from file)"),
  make_option("--n-cases",            type = "character", default = "0",
              help = "Comma-separated case counts (0 = read from file)"),
  make_option("--n-controls",         type = "character", default = "0",
              help = "Comma-separated control counts (0 = read from file)"),
  make_option("--ld-meta-data-paths", type = "character", default = NULL,
              help = "Comma-separated per-study LD metadata paths (use 'None' to fall back to --ld-meta-data)"),
  make_option("--region",             type = "character", default = NULL,
              help = "Region string in format chrN_START-END (colons replaced by underscores by SoS)"),
  make_option("--output",             type = "character", default = NULL,
              help = "Output RDS file path"),

  # optional region filtering
  make_option("--skip-regions",        type = "character", default = NULL,
              help = "Comma-separated regions to skip (chrN:start-end)"),
  make_option("--extract-region-name", type = "character", default = NULL,
              help = "Gene/phenotype name to subset from sumstats"),
  make_option("--region-name-col",     type = "character", default = NULL,
              help = "Column to filter for --extract-region-name"),

  # variant filters
  make_option("--imiss", type = "double", default = 1.0,
              help = "Individual missingness cutoff [default %default]"),
  make_option("--maf",   type = "double", default = 0.0025,
              help = "MAF cutoff [default %default]"),

  # fine-mapping
  make_option("--L",                        type = "integer", default = 15L,
              help = "Max SuSiE components [default %default]"),
  make_option("--L-greedy",                 type = "integer", default = 5L,
              help = "Greedy SuSiE components [default %default]"),
  make_option("--pip-cutoff",               type = "double",  default = 0.025,
              help = "Signal PIP cutoff [default %default]"),
  make_option("--skip-analysis-pip-cutoff", type = "double",  default = 0.025,
              help = "PIP threshold for early-exit [default %default]"),
  make_option("--coverage", type = "character", default = "0.95,0.7,0.5",
              help = "Comma-separated credible-set coverages [default %default]"),
  make_option("--finemapping-method", type = "character", default = "single_effect",
              help = "Fine-mapping method: susie_rss | single_effect | bayesian_conditional_regression [default %default]"),

  # LD
  make_option("--ld-reference-size",      type = "integer", default = 10000L,
              help = "Reference-panel size for finite-sample LD correction [default %default]"),
  make_option("--ld-mismatch-correction", action = "store_true", default = FALSE,
              help = "Enable LD mismatch correction"),

  # imputation
  make_option("--impute",       action = "store_true", default = FALSE,
              help = "Impute missing variants via RAISS"),
  make_option("--rcond",        type = "double",  default = 0.01,
              help = "RAISS rcond [default %default]"),
  make_option("--lamb",         type = "double",  default = 0.01,
              help = "RAISS lambda [default %default]"),
  make_option("--r2-threshold", type = "double",  default = 0.6,
              help = "RAISS R2 threshold [default %default]"),
  make_option("--minimum-ld",   type = "integer", default = 5L,
              help = "RAISS minimum LD variants [default %default]"),

  # QC
  make_option("--qc-method",      type = "character", default = "",
              help = "QC method: slalom | dentist | '' (none) [default none]"),
  make_option("--comment-string", type = "character", default = "#",
              help = "Comment character in sumstat files [default %default]"),

  # diagnostics / output
  make_option("--diagnostics",   action = "store_true", default = FALSE,
              help = "Run diagnostic fine-mapping mode"),
  make_option("--output-prefix", type = "character", default = NULL,
              help = "Output prefix (without extension, optional)")
)

opt <- parse_args(OptionParser(option_list = opt_list))

# --------------- Validate required args ---------------
# Note: R optparse keeps the original hyphenated name as the list key.

for (arg in c("ld-meta-data", "studies", "sumstat-paths",
              "column-file-paths", "n-samples", "n-cases",
              "n-controls", "ld-meta-data-paths", "region", "output")) {
  if (is.null(opt[[arg]])) stop(sprintf("--%s is required", arg))
}

# --------------- Parse region string ---------------
# SoS replaces ':' with '_' in region strings before passing to Rscript, so
# '22:49355984-50799822' becomes 'chr22_49355984_50799822' (both separators
# are underscores).  Convert back to 'chr22:49355984-50799822'.
region_raw <- opt[["region"]]
if (grepl("^chr[^_]+_[0-9]+_[0-9]+$", region_raw)) {
  # chrN_START_END  →  chrN:START-END
  region_std <- sub("^(chr[^_]+)_([0-9]+)_([0-9]+)$", "\\1:\\2-\\3", region_raw)
} else if (grepl("^chr[^_]+_[0-9]+-[0-9]+$", region_raw)) {
  # chrN_START-END  →  chrN:START-END
  region_std <- sub("^(chr[^_]+)_", "\\1:", region_raw)
} else {
  region_std <- region_raw   # assume already in correct format
}
message("Region: ", region_std)

# --------------- Helper functions ---------------

split_field  <- function(x) trimws(strsplit(x, ",")[[1]])
null_if_none <- function(v) if (v %in% c("None", "none", "NULL", "null", "")) NULL else v

# Return NULL when the bgzipped sumstat file has no #-prefixed tabix header,
# so that load_rss_data falls back to full-file reading (which preserves the
# real column names).  When the tabix header exists, return region as-is.
region_for_sumstat <- function(sumstat_path, region) {
  if (!grepl("\\.gz$", sumstat_path) || !file.exists(sumstat_path)) return(region)
  tbx <- tryCatch(Rsamtools::TabixFile(sumstat_path), error = function(e) NULL)
  if (is.null(tbx)) return(region)
  hdr <- tryCatch(Rsamtools::headerTabix(tbx)$header, error = function(e) character(0))
  if (length(hdr) > 0) return(region)
  message("  [note] No #-header in tabix index for ", basename(sumstat_path),
          "; loading full file so column names are preserved.")
  NULL
}

# --------------- Parse comma-separated fields ---------------

studies        <- split_field(opt[["studies"]])
sumstat_paths  <- split_field(opt[["sumstat-paths"]])
col_file_paths <- split_field(opt[["column-file-paths"]])
n_samples      <- as.integer(split_field(opt[["n-samples"]]))
n_cases        <- as.integer(split_field(opt[["n-cases"]]))
n_controls     <- as.integer(split_field(opt[["n-controls"]]))
ld_meta_paths  <- split_field(opt[["ld-meta-data-paths"]])
coverage       <- as.numeric(split_field(opt[["coverage"]]))
skip_regions_vec <- if (!is.null(opt[["skip-regions"]])) split_field(opt[["skip-regions"]]) else NULL

n_studies <- length(studies)
if (!all(lengths(list(sumstat_paths, col_file_paths, n_samples,
                      n_cases, n_controls, ld_meta_paths)) == n_studies)) {
  stop("--studies, --sumstat-paths, --column-file-paths, --n-samples, ",
       "--n-cases, --n-controls, --ld-meta-data-paths must all have the same number of entries")
}

global_ld_meta <- opt[["ld-meta-data"]]
qc_method_arg  <- if (nzchar(opt[["qc-method"]])) opt[["qc-method"]] else NULL
R_mismatch_arg <- if (isTRUE(opt[["ld-mismatch-correction"]])) "simple" else NULL
R_finite_arg   <- if (opt[["ld-reference-size"]] > 0) opt[["ld-reference-size"]] else NULL

finemapping_opts <- list(
  L             = opt[["L"]],
  L_greedy      = opt[["L-greedy"]],
  coverage      = coverage,
  signal_cutoff = opt[["pip-cutoff"]],
  min_abs_corr  = 0.8
)

impute_opts <- list(
  rcond        = opt[["rcond"]],
  R2_threshold = opt[["r2-threshold"]],
  minimum_ld   = opt[["minimum-ld"]],
  lamb         = opt[["lamb"]]
)

# --------------- Per-study analysis ---------------

results <- vector("list", n_studies)
names(results) <- studies

for (i in seq_len(n_studies)) {
  study <- studies[i]
  message("\n=== Study: ", study, " (", i, "/", n_studies, ") ===")

  ld_meta_i  <- null_if_none(ld_meta_paths[i])
  if (is.null(ld_meta_i)) ld_meta_i <- global_ld_meta

  col_file_i <- null_if_none(col_file_paths[i])

  message("Loading LD matrix for region ", region_std, " ...")
  LD_data <- tryCatch(
    load_LD_matrix(ld_meta_i, region = region_std),
    error = function(e) {
      message("ERROR loading LD matrix: ", conditionMessage(e))
      NULL
    }
  )

  if (is.null(LD_data)) {
    message("Skipping study ", study, " (LD load failed).")
    results[[i]] <- list(rss_data_analyzed = data.frame())
    next
  }

  region_i <- region_for_sumstat(sumstat_paths[i], region_std)

  message("Running rss_analysis_pipeline ...")
  res <- tryCatch(
    rss_analysis_pipeline(
      sumstat_path        = sumstat_paths[i],
      column_file_path    = col_file_i,
      LD_data             = LD_data,
      n_sample            = n_samples[i],
      n_case              = n_cases[i],
      n_control           = n_controls[i],
      region              = region_i,
      skip_region         = skip_regions_vec,
      extract_region_name = if (!is.null(opt[["extract-region-name"]])) null_if_none(opt[["extract-region-name"]]) else NULL,
      region_name_col     = if (!is.null(opt[["region-name-col"]]))     null_if_none(opt[["region-name-col"]])     else NULL,
      qc_method           = qc_method_arg,
      finemapping_method  = null_if_none(opt[["finemapping-method"]]),
      finemapping_opts    = finemapping_opts,
      impute              = isTRUE(opt[["impute"]]),
      impute_opts         = impute_opts,
      pip_cutoff_to_skip  = opt[["skip-analysis-pip-cutoff"]],
      R_finite            = R_finite_arg,
      R_mismatch          = R_mismatch_arg,
      comment_string      = opt[["comment-string"]],
      diagnostics         = isTRUE(opt[["diagnostics"]])
    ),
    error = function(e) {
      message("ERROR in rss_analysis_pipeline: ", conditionMessage(e))
      list(rss_data_analyzed = data.frame())
    }
  )

  results[[i]] <- res
}

# --------------- Save ---------------

dir.create(dirname(opt[["output"]]), recursive = TRUE, showWarnings = FALSE)
saveRDS(results, file = opt[["output"]])
message("\nSaved: ", opt[["output"]])
