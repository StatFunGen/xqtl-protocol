#!/usr/bin/env Rscript
# susie_twas.R — Univariate SuSiE fine-mapping and TWAS weights
# Called directly by route3 SoS notebook task blocks.
#
# Usage:
#   Rscript susie_twas.R \
#     --genotype geno.bed \
#     --phenotype pheno1.bed.gz,pheno2.bed.gz \
#     --covariate cov1.gz,cov2.gz \
#     --region chr12:752578-752579 \
#     --window chr12:0-1000000 \
#     --region-name ENSG00000060237 \
#     --extract-region-names "ENSG00000060237" \
#     --conditions cond1,cond2 \
#     --output-prefix /path/to/output/name.chrN_GENE \
#     --cwd /path/to/output

suppressPackageStartupMessages(library(optparse))

option_list <- list(
  # inputs
  make_option("--genotype",               type="character"),
  make_option("--phenotype",              type="character", help="Comma-separated phenotype file paths"),
  make_option("--covariate",              type="character", help="Comma-separated covariate file paths"),
  make_option("--region",                 type="character", help="Region coordinate, e.g. chr12:752578-752579"),
  make_option("--window",                 type="character", help="Association window, e.g. chr12:0-1000000"),
  make_option("--region-name",            type="character", help="Primary gene/region ID"),
  make_option("--extract-region-names",   type="character", default="",
              help="Pipe-separated groups of comma-separated gene IDs, e.g. 'gene1|iso1,iso2'"),
  make_option("--conditions",             type="character", default="",
              help="Comma-separated condition names"),
  make_option("--skip-analysis-pip-cutoff", type="character", default="",
              help="Comma-separated cond=value pairs for per-condition PIP cutoff to skip"),
  # variant/sample filters
  make_option("--maf",        type="double",  default=0.0025),
  make_option("--mac",        type="integer", default=0),
  make_option("--imiss",      type="double",  default=1.0),
  make_option("--indel",      action="store_true", default=FALSE),
  make_option("--keep-samples",   type="character", default="."),
  make_option("--keep-variants",  type="character", default="."),
  # analysis modes
  make_option("--save-data",           action="store_true", default=FALSE),
  make_option("--skip-fine-mapping",   action="store_true", default=FALSE),
  make_option("--skip-twas-weights",   action="store_true", default=FALSE),
  make_option("--trans-analysis",      action="store_true", default=FALSE),
  # fine-mapping params
  make_option("--pip-cutoff",   type="double",    default=0.025),
  make_option("--coverage",     type="character", default="0.95,0.7,0.5"),
  make_option("--init-l",       type="integer",   default=1),
  make_option("--max-l",        type="integer",   default=10),
  make_option("--seed",         type="integer",   default=999),
  # TWAS params
  make_option("--max-cv-variants",  type="integer", default=5000),
  make_option("--twas-cv-folds",    type="integer", default=5),
  make_option("--twas-cv-threads",  type="integer", default=1),
  make_option("--min-twas-maf",     type="double",  default=0.01),
  make_option("--min-twas-xvar",    type="double",  default=0.01),
  make_option("--ld-reference-meta-file", type="character", default="."),
  # output
  make_option("--output-prefix",  type="character"),
  make_option("--cwd",            type="character", default="output")
)

opt <- parse_args(OptionParser(option_list=option_list))

library(pecotmr)

phenotype_files <- strsplit(opt$phenotype, ",")[[1]]
covariate_files <- strsplit(opt$covariate, ",")[[1]]
conditions      <- if (nchar(opt$conditions) > 0) strsplit(opt$conditions, ",")[[1]] else character(0)
coverage        <- as.numeric(strsplit(opt$coverage, ",")[[1]])

# Parse extract_region_names: pipe-separated groups, each group comma-separated
extract_region_name <- if (nchar(opt[["extract-region-names"]]) > 0) {
  lapply(strsplit(opt[["extract-region-names"]], "|", fixed=TRUE)[[1]],
         function(g) strsplit(g, ",")[[1]])
} else {
  list(opt[["region-name"]])
}

# Parse per-condition PIP skip cutoffs
pip_cutoff_to_skip <- NULL
if (nchar(opt[["skip-analysis-pip-cutoff"]]) > 0) {
  pairs <- strsplit(opt[["skip-analysis-pip-cutoff"]], ",")[[1]]
  vals  <- setNames(
    as.numeric(sub(".*=", "", pairs)),
    gsub("'", "", sub("=.*", "", pairs))
  )
  pip_cutoff_to_skip <- vals[conditions]
}

# Optional files
keep_samples <- if (file.exists(opt[["keep-samples"]])) {
  message(paste("Loading keep_samples from", opt[["keep-samples"]]))
  unlist(strsplit(readLines(opt[["keep-samples"]]), "\\s+"))
} else NULL

keep_variants <- if (file.exists(opt[["keep-variants"]])) opt[["keep-variants"]] else NULL

ld_ref <- if (file.exists(opt[["ld-reference-meta-file"]])) opt[["ld-reference-meta-file"]] else NULL

# Infer phenotype_header from region (if end coord > 0, use 4-column BED header)
region_end  <- as.integer(sub(".*-", "", opt$region))
pheno_header <- if (!is.na(region_end) && region_end > 0) 4L else 1L

region_str <- if (!opt[["trans-analysis"]]) opt$region else NULL

tryCatch({
  fdat <- load_regional_univariate_data(
    genotype          = opt$genotype,
    phenotype         = phenotype_files,
    covariate         = covariate_files,
    region            = region_str,
    association_window = opt$window,
    conditions        = conditions,
    maf_cutoff        = opt$maf,
    mac_cutoff        = opt$mac,
    imiss_cutoff      = opt$imiss,
    keep_indel        = opt$indel,
    keep_samples      = keep_samples,
    keep_variants     = keep_variants,
    extract_region_name = extract_region_name,
    phenotype_header  = pheno_header,
    region_name_col   = pheno_header,
    scale_residuals   = FALSE
  )
}, NoSNPsError = function(e) {
  message("Error: ", paste(e$message, paste0(opt[["region-name"]], "@", opt$window)))
  data_dir <- file.path(opt$cwd, "data")
  dir.create(data_dir, recursive=TRUE, showWarnings=FALSE)
  region_name_sym <- opt[["region-name"]]
  result <- list()
  result[[region_name_sym]] <- e$message
  saveRDS(result, file.path(data_dir, paste0(basename(opt[["output-prefix"]]), ".univariate_data.rds")), compress="xz")
  quit(save="no")
})

# Optionally save raw data
if (opt[["save-data"]]) {
  data_dir <- file.path(opt$cwd, "data")
  dir.create(data_dir, recursive=TRUE, showWarnings=FALSE)
  region_name_sym <- opt[["region-name"]]
  result <- list()
  result[[region_name_sym]] <- fdat
  saveRDS(result, file.path(data_dir, paste0(basename(opt[["output-prefix"]]), ".univariate_data.rds")), compress="xz")
}

if (opt[["skip-fine-mapping"]] && opt[["skip-twas-weights"]]) quit(save="no")

# Build region_name vector (primary gene + any isoforms/sub-regions)
region_name_vec <- if (opt[["region-name"]] != as.character(extract_region_name[[1]][1])) {
  c(opt[["region-name"]], unlist(extract_region_name))
} else {
  opt[["region-name"]]
}

region_info <- list(
  region_coord = parse_region(opt$region),
  grange       = parse_region(opt$window),
  region_name  = region_name_vec
)

finemapping_result   <- list()
preset_variants_result <- list()
condition_names      <- character(0)
empty_elements_cnt   <- 0L

for (r in seq_along(fdat$residual_Y)) {
  dropped_samples <- list(
    X     = fdat$dropped_sample$dropped_samples_X[[r]],
    y     = fdat$dropped_sample$dropped_samples_Y[[r]],
    covar = fdat$dropped_sample$dropped_samples_covar[[r]]
  )
  new_names     <- names(fdat$residual_Y)[r]
  new_col_names <- extract_region_name[[r]]
  if (is.null(new_col_names)) new_col_names <- seq_len(ncol(fdat$residual_Y[[r]]))
  if (!identical(new_names, new_col_names))
    new_names <- paste(new_names, new_col_names, sep="_")

  out <- list()

  if (!opt[["skip-fine-mapping"]]) {
    out$finemapping <- lapply(seq_len(ncol(fdat$residual_Y[[r]])), function(i) {
      set.seed(opt$seed)
      univariate_analysis_pipeline(
        X              = fdat$residual_X[[r]],
        Y              = fdat$residual_Y[[r]][, i, drop=FALSE],
        maf            = fdat$maf[[r]],
        X_scalar       = fdat$residual_X_scalar[[r]],
        Y_scalar       = if (identical(fdat$residual_Y_scalar[[r]], 1)) 1 else fdat$residual_Y_scalar[[r]][, i, drop=FALSE],
        X_variance     = fdat$X_variance[[r]],
        other_quantities = list(dropped_samples=dropped_samples),
        imiss_cutoff   = opt$imiss,
        maf_cutoff     = NULL,
        xvar_cutoff    = 0,
        ld_reference_meta_file = NULL,
        pip_cutoff_to_skip = if (!is.null(pip_cutoff_to_skip)) pip_cutoff_to_skip[r] else NULL,
        init_L         = opt[["init-l"]],
        max_L          = opt[["max-l"]],
        l_step         = 5,
        signal_cutoff  = opt[["pip-cutoff"]],
        coverage       = coverage,
        twas_weights   = FALSE,
        max_cv_variants = opt[["max-cv-variants"]],
        cv_folds       = opt[["twas-cv-folds"]],
        cv_threads     = opt[["twas-cv-threads"]]
      )
    })
  }

  if (!opt[["skip-twas-weights"]]) {
    common_cols <- intersect(colnames(fdat$X), colnames(fdat$residual_X[[r]]))
    X_r  <- fdat$X[rownames(fdat$residual_X[[r]]), common_cols, drop=FALSE]
    maf_r <- fdat$maf[[r]][common_cols]
    out$twas_models <- lapply(seq_len(ncol(fdat$residual_Y[[r]])), function(i) {
      set.seed(opt$seed)
      univariate_analysis_pipeline(
        X              = X_r,
        Y              = fdat$residual_Y[[r]][, i, drop=FALSE],
        maf            = maf_r,
        X_scalar       = fdat$residual_X_scalar[[r]],
        Y_scalar       = if (identical(fdat$residual_Y_scalar[[r]], 1)) 1 else fdat$residual_Y_scalar[[r]][, i, drop=FALSE],
        X_variance     = fdat$X_variance[[r]],
        other_quantities = list(dropped_samples=dropped_samples),
        imiss_cutoff   = opt$imiss,
        maf_cutoff     = opt[["min-twas-maf"]],
        xvar_cutoff    = opt[["min-twas-xvar"]],
        ld_reference_meta_file = ld_ref,
        pip_cutoff_to_skip = if (!is.null(pip_cutoff_to_skip)) pip_cutoff_to_skip[r] else NULL,
        init_L         = opt[["init-l"]],
        max_L          = opt[["max-l"]],
        l_step         = 5,
        signal_cutoff  = opt[["pip-cutoff"]],
        coverage       = coverage,
        twas_weights   = TRUE,
        max_cv_variants = opt[["max-cv-variants"]],
        cv_folds       = opt[["twas-cv-folds"]],
        cv_threads     = opt[["twas-cv-threads"]]
      )
    })
  }

  empty_idx <- unique(unlist(lapply(out, function(res)
    which(sapply(res, function(x) is.list(x) && length(x) == 0)))))
  if (length(empty_idx) > 0) {
    empty_elements_cnt <- empty_elements_cnt + length(empty_idx)
    if (!is.null(out$finemapping))  out$finemapping  <- out$finemapping[-empty_idx]
    if (!is.null(out$twas_models))  out$twas_models  <- out$twas_models[-empty_idx]
    new_names <- new_names[-empty_idx]
  }

  if (!is.null(out$finemapping))  finemapping_result   <- c(finemapping_result, out$finemapping)
  if (!is.null(out$twas_models))  preset_variants_result <- c(preset_variants_result, out$twas_models)
  condition_names <- c(condition_names, new_names)
  if (length(new_names) > 0) message("Analysis completed for: ", paste(new_names, collapse=","))

  fdat$residual_X[[r]] <- NA
  fdat$residual_Y[[r]] <- NA
}

twas_output       <- list()
finemapping_output <- list()

if (length(preset_variants_result) > 0) {
  names(preset_variants_result) <- condition_names
  for (r in condition_names) {
    twas_output[[r]] <- preset_variants_result[[r]]$twas_weights_result
    preset_variants_result[[r]]$twas_weights_result <- NULL
    twas_output[[r]]$variant_names <- preset_variants_result[[r]]$variant_names
    twas_output[[r]]$region_info   <- region_info
    preset_variants_result[[r]]$region_info <- region_info
  }
}

if (length(finemapping_result) > 0) names(finemapping_result) <- condition_names

for (r in condition_names) {
  if (r %in% names(finemapping_result)) {
    finemapping_output[[r]] <- finemapping_result[[r]]
    finemapping_output[[r]]$region_info  <- region_info
    finemapping_output[[r]]$susie_fitted <- NULL
  }
  if (r %in% names(preset_variants_result))
    finemapping_output[[r]]$preset_variants_result <- preset_variants_result[[r]]
}

if (empty_elements_cnt > 0) {
  msg <- paste0(empty_elements_cnt, " analysis are skipped for failing to pass initial screen for potential association signals")
  message(msg)
  if (length(finemapping_output) == 0) finemapping_output <- msg
  if (length(twas_output) == 0)        twas_output        <- msg
}

region_name_sym <- opt[["region-name"]]
prefix <- opt[["output-prefix"]]

if (!opt[["skip-fine-mapping"]]) {
  fm_dir <- file.path(opt$cwd, "fine_mapping")
  dir.create(fm_dir, recursive=TRUE, showWarnings=FALSE)
  result <- list(); result[[region_name_sym]] <- finemapping_output
  saveRDS(result, file.path(fm_dir, paste0(basename(prefix), ".univariate_bvsr.rds")), compress="xz")
}

if (!opt[["skip-twas-weights"]]) {
  tw_dir <- file.path(opt$cwd, "twas_weights")
  dir.create(tw_dir, recursive=TRUE, showWarnings=FALSE)
  result <- list(); result[[region_name_sym]] <- twas_output
  saveRDS(result, file.path(tw_dir, paste0(basename(prefix), ".univariate_twas_weights.rds")), compress="xz")
}
