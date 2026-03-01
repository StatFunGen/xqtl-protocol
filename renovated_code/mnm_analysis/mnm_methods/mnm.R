#!/usr/bin/env Rscript
# mnm.R — Multivariate mvSuSiE fine-mapping and mr.mash TWAS weights
# Called directly by route3 SoS notebook task blocks.
#
# Usage:
#   Rscript mnm.R \
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
  make_option("--region",                 type="character"),
  make_option("--window",                 type="character"),
  make_option("--region-name",            type="character"),
  make_option("--extract-region-names",   type="character", default=""),
  make_option("--conditions",             type="character", default=""),
  make_option("--skip-analysis-pip-cutoff", type="character", default=""),
  # variant/sample filters
  make_option("--maf",        type="double",  default=0.0025),
  make_option("--mac",        type="integer", default=0),
  make_option("--imiss",      type="double",  default=1.0),
  make_option("--xvar-cutoff",type="double",  default=0),
  make_option("--indel",      action="store_true", default=FALSE),
  make_option("--keep-samples",   type="character", default="."),
  make_option("--keep-variants",  type="character", default="."),
  # analysis modes
  make_option("--save-data",           action="store_true", default=FALSE),
  make_option("--skip-fine-mapping",   action="store_true", default=FALSE),
  make_option("--skip-twas-weights",   action="store_true", default=FALSE),
  # prior model files
  make_option("--mixture-prior",       type="character", default="."),
  make_option("--mixture-prior-cv",    type="character", default="."),
  make_option("--prior-weights-min",   type="double",    default=5e-4),
  make_option("--prior-canonical-matrices", action="store_true", default=FALSE),
  make_option("--sample-partition",    type="character", default="."),
  # fine-mapping params
  make_option("--pip-cutoff",     type="double",    default=0.025),
  make_option("--coverage",       type="character", default="0.95,0.7,0.5"),
  make_option("--mvsusie-max-iter",type="integer",  default=200),
  make_option("--mrmash-max-iter", type="integer",  default=5000),
  make_option("--seed",           type="integer",   default=999),
  # TWAS params
  make_option("--max-cv-variants",  type="integer", default=5000),
  make_option("--twas-cv-folds",    type="integer", default=5),
  make_option("--twas-cv-threads",  type="integer", default=1),
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

extract_region_name <- if (nchar(opt[["extract-region-names"]]) > 0) {
  lapply(strsplit(opt[["extract-region-names"]], "|", fixed=TRUE)[[1]],
         function(g) strsplit(g, ",")[[1]])
} else {
  list(opt[["region-name"]])
}

pip_cutoff_to_skip <- NULL
if (nchar(opt[["skip-analysis-pip-cutoff"]]) > 0) {
  pairs <- strsplit(opt[["skip-analysis-pip-cutoff"]], ",")[[1]]
  vals  <- setNames(
    as.numeric(sub(".*=", "", pairs)),
    gsub("'", "", sub("=.*", "", pairs))
  )
  pip_cutoff_to_skip <- vals[conditions]
}

keep_samples <- if (file.exists(opt[["keep-samples"]])) {
  unlist(strsplit(readLines(opt[["keep-samples"]]), "\\s+"))
} else NULL

keep_variants <- if (file.exists(opt[["keep-variants"]])) opt[["keep-variants"]] else NULL
ld_ref        <- if (file.exists(opt[["ld-reference-meta-file"]])) opt[["ld-reference-meta-file"]] else NULL

region_end   <- as.integer(sub(".*-", "", opt$region))
pheno_header <- if (!is.na(region_end) && region_end > 0) 4L else 1L

data_dir <- file.path(opt$cwd, "data")
dir.create(data_dir, recursive=TRUE, showWarnings=FALSE)

region_name_sym <- opt[["region-name"]]
prefix <- opt[["output-prefix"]]

tryCatch({
  fdat <- load_regional_multivariate_data(
    genotype          = opt$genotype,
    phenotype         = phenotype_files,
    covariate         = covariate_files,
    region            = opt$region,
    conditions        = conditions,
    association_window = opt$window,
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
  message("Error: ", paste(e$message, paste0(region_name_sym, "@", opt$window)))
  result <- list(); result[[region_name_sym]] <- e$message
  saveRDS(result, file.path(data_dir, paste0(basename(prefix), ".multicontext_data.rds")), compress="xz")
  quit(save="no")
})

if (opt[["save-data"]]) {
  saveRDS(fdat, file.path(data_dir, paste0(basename(prefix), ".multicontext_data.rds")), compress="xz")
}

if (opt[["skip-fine-mapping"]] && opt[["skip-twas-weights"]]) quit(save="no")

# Build region_name and region_info
region_name_vec <- if (region_name_sym != as.character(extract_region_name[[1]][1])) {
  c(region_name_sym, unlist(extract_region_name))
} else region_name_sym

context_regions <- setNames(extract_region_name, colnames(fdat$residual_Y))
region_info <- list(
  region_coord = parse_region(opt$region),
  grange       = parse_region(opt$window),
  region_name  = region_name_vec
)

# Load optional prior models
dd_prior    <- if (file.exists(opt[["mixture-prior"]]))    readRDS(opt[["mixture-prior"]])    else NULL
dd_prior_cv <- if (file.exists(opt[["mixture-prior-cv"]])) readRDS(opt[["mixture-prior-cv"]]) else NULL
use_canonical_prior <- (opt[["prior-canonical-matrices"]] && !is.null(dd_prior)) || is.null(dd_prior)

sample_part <- if (file.exists(opt[["sample-partition"]])) opt[["sample-partition"]] else NULL

set.seed(opt$seed)
result <- multivariate_analysis_pipeline(
  X                             = fdat$X,
  Y                             = fdat$residual_Y,
  maf                           = fdat$maf,
  X_variance                    = fdat$X_variance,
  other_quantities              = list(dropped_samples=fdat$dropped_samples),
  imiss_cutoff                  = opt$imiss,
  maf_cutoff                    = opt$maf,
  xvar_cutoff                   = opt[["xvar-cutoff"]],
  ld_reference_meta_file        = ld_ref,
  pip_cutoff_to_skip            = pip_cutoff_to_skip,
  max_L                         = -1,
  data_driven_prior_matrices    = dd_prior,
  data_driven_prior_matrices_cv = dd_prior_cv,
  data_driven_prior_weights_cutoff = opt[["prior-weights-min"]],
  canonical_prior_matrices      = use_canonical_prior,
  mvsusie_max_iter              = opt[["mvsusie-max-iter"]],
  mrmash_max_iter               = opt[["mrmash-max-iter"]],
  signal_cutoff                 = opt[["pip-cutoff"]],
  coverage                      = coverage,
  twas_weights                  = !opt[["skip-twas-weights"]],
  sample_partition              = sample_part,
  max_cv_variants               = opt[["max-cv-variants"]],
  cv_folds                      = opt[["twas-cv-folds"]],
  cv_threads                    = opt[["twas-cv-threads"]]
)

result$region_info <- region_info

if (!is.null(result$twas_weights_result)) {
  tw_dir <- file.path(opt$cwd, "multivariate_twas_weights")
  dir.create(tw_dir, recursive=TRUE, showWarnings=FALSE)
  for (ctx in names(result$twas_weights_result)) {
    result$twas_weights_result[[ctx]]$region_info <- region_info
    ctx_region <- context_regions[[ctx]]
    if (is.null(ctx_region)) ctx_region <- ctx
    new_ctx <- if (!identical(ctx, ctx_region)) paste(ctx, ctx_region, sep="_") else ctx
    names(result$twas_weights_result)[match(ctx, names(result$twas_weights_result))] <- new_ctx
  }
  tw_result <- list(); tw_result[[region_name_sym]] <- result$twas_weights_result
  saveRDS(tw_result, file.path(tw_dir, paste0(basename(prefix), ".multicontext_twas_weights.rds")), compress="xz")
  result$twas_weights_result <- NULL
}

fm_dir <- file.path(opt$cwd, "multivariate_fine_mapping")
dir.create(fm_dir, recursive=TRUE, showWarnings=FALSE)
fm_result <- list(); fm_result[[region_name_sym]] <- result
saveRDS(fm_result, file.path(fm_dir, paste0(basename(prefix), ".multicontext_bvsr.rds")), compress="xz")
