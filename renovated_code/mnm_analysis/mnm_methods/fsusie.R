#!/usr/bin/env Rscript
# fsusie.R — Functional SuSiE fine-mapping for epigenomic QTL
# Called directly by route3 SoS notebook task blocks.
#
# Usage:
#   Rscript fsusie.R \
#     --genotype geno.bed \
#     --phenotype pheno1.bed.gz,pheno2.bed.gz \
#     --covariate cov1.gz,cov2.gz \
#     --region chr7:139293693-145380632 \
#     --window chr7:139293693-145380632 \
#     --region-name Mic \
#     --conditions ROSMAP_Mic_snATACQTL \
#     --output-prefix /path/to/output/Mic.chr7_139293693_145380632 \
#     --cwd /path/to/output

suppressPackageStartupMessages(library(optparse))

option_list <- list(
  # inputs
  make_option("--genotype",   type="character"),
  make_option("--phenotype",  type="character", help="Comma-separated phenotype file paths"),
  make_option("--covariate",  type="character", help="Comma-separated covariate file paths"),
  make_option("--region",     type="character"),
  make_option("--window",     type="character"),
  make_option("--region-name",type="character"),
  make_option("--conditions", type="character", default=""),
  # variant/sample filters
  make_option("--maf",      type="double",  default=0.0025),
  make_option("--mac",      type="integer", default=0),
  make_option("--imiss",    type="double",  default=1.0),
  make_option("--indel",    action="store_true", default=FALSE),
  make_option("--keep-samples",  type="character", default="."),
  make_option("--keep-variants", type="character", default="."),
  # fsusie model params
  make_option("--prior",          type="character", default="mixture_normal"),
  make_option("--max-snp-em",     type="integer",   default=100),
  make_option("--max-scale",      type="integer",   default=10),
  make_option("--min-purity",     type="double",    default=0.5),
  make_option("--epigenetics-mark-threshold", type="integer", default=16),
  make_option("--susie-top-pc",   type="integer",   default=10),
  make_option("--post-processing",type="character", default="TI"),
  make_option("--small-sample-correction", action="store_true", default=FALSE),
  # fine-mapping summary params
  make_option("--pip-cutoff",   type="double",    default=0.025),
  make_option("--coverage",     type="character", default="0.95,0.7,0.5"),
  make_option("--init-l",       type="integer",   default=1),
  make_option("--max-l",        type="integer",   default=10),
  # TWAS
  make_option("--skip-twas-weights", action="store_true", default=FALSE),
  make_option("--max-cv-variants",   type="integer", default=5000),
  make_option("--twas-cv-folds",     type="integer", default=5),
  make_option("--twas-cv-threads",   type="integer", default=1),
  # misc
  make_option("--save-data",    action="store_true", default=FALSE),
  make_option("--output-prefix",type="character"),
  make_option("--cwd",          type="character", default="output")
)

opt <- parse_args(OptionParser(option_list=option_list))

library(pecotmr)
library(tidyverse)

phenotype_files <- strsplit(opt$phenotype, ",")[[1]]
covariate_files <- strsplit(opt$covariate, ",")[[1]]
conditions      <- if (nchar(opt$conditions) > 0) strsplit(opt$conditions, ",")[[1]] else character(0)
coverage        <- as.numeric(strsplit(opt$coverage, ",")[[1]])
mark_thresh     <- opt[["epigenetics-mark-threshold"]]
prefix          <- opt[["output-prefix"]]
region_coord_str <- gsub(":", "_", gsub("-", "_", opt$region))

keep_samples <- if (file.exists(opt[["keep-samples"]])) {
  unlist(strsplit(readLines(opt[["keep-samples"]]), "\\s+"))
} else NULL
keep_variants <- if (file.exists(opt[["keep-variants"]])) opt[["keep-variants"]] else NULL

tryCatch({
  fdat <- load_regional_functional_data(
    genotype          = opt$genotype,
    phenotype         = phenotype_files,
    covariate         = covariate_files,
    region            = opt$region,
    association_window = opt$window,
    conditions        = conditions,
    maf_cutoff        = opt$maf,
    mac_cutoff        = opt$mac,
    imiss_cutoff      = opt$imiss,
    keep_indel        = opt$indel,
    keep_samples      = keep_samples,
    keep_variants     = keep_variants,
    tabix_header      = TRUE,
    phenotype_header  = 4,
    region_name_col   = 4,
    scale_residuals   = FALSE
  )
}, NoSNPsError = function(e) {
  message("Error: ", paste(e$message, paste0(opt[["region-name"]], "@", opt$window)))
  result <- list(); result[[opt$region]] <- e$message
  saveRDS(result, paste0(prefix, ".rds"), compress="xz")
  quit(save="no")
})

# Filter regions with too few epigenomic marks
filter_fdat_except_specific_names <- function(fdat, n) {
  indices_to_keep <- sapply(fdat$Y_coordinates, function(x) nrow(x) >= n)
  fdat_filtered   <- map(fdat[!names(fdat) %in% c("dropped_sample", "X", "chrom")],
                         ~.x[indices_to_keep])
  c(fdat_filtered, fdat[names(fdat) %in% c("dropped_sample", "X", "chrom")])
}

fdat <- filter_fdat_except_specific_names(fdat, n=mark_thresh)

if (length(fdat$Y_coordinates) == 0) {
  e_msg <- paste0("None of the studies have >= ", mark_thresh, " epigenetic marks; region skipped")
  message(e_msg)
  result <- list(); result[[opt$region]] <- e_msg
  saveRDS(result, paste0(prefix, ".rds"), compress="xz")
  quit(save="no")
}

if (opt[["save-data"]]) {
  saveRDS(list(opt$region = fdat),
          paste0(prefix, ".", mark_thresh, "_marks.dataset.rds"), compress="xz")
}

fitted <- setNames(replicate(length(fdat$residual_Y), list(), simplify=FALSE),
                   names(fdat$residual_Y))

for (r in seq_along(fitted)) {
  st <- proc.time()
  fitted[[r]] <- list()
  message(paste("Y matrix:", nrow(fdat$residual_Y[[r]]), "rows x", ncol(fdat$residual_Y[[r]]), "cols"))

  if (opt[["susie-top-pc"]] > 0 || !opt[["skip-twas-weights"]]) {
    top_pc_data <- prcomp(fdat$residual_Y[[r]], center=TRUE, scale.=TRUE)$x
    k <- min(opt[["susie-top-pc"]], ncol(top_pc_data))
    if (k > 0) top_pc_data <- top_pc_data[, 1:k, drop=FALSE]

    fitted[[r]]$susie_on_top_pc <- list()
    for (i in seq_len(ncol(top_pc_data))) {
      s <- susie_wrapper(fdat$residual_X[[r]], top_pc_data[, i],
                         init_L=opt[["init-l"]], max_L=opt[["max-l"]],
                         refine=TRUE, coverage=coverage[1])
      fitted[[r]]$susie_on_top_pc[[i]] <- susie_post_processor(
        s, fdat$residual_X[[r]], top_pc_data[, i],
        fdat$residual_X_scalar[[r]], 1, fdat$maf[[r]],
        secondary_coverage = coverage[-1],
        signal_cutoff      = opt[["pip-cutoff"]],
        other_quantities   = list(dropped_samples=list(
          X=fdat$dropped_sample$dropped_samples_X[[r]],
          y=fdat$dropped_sample$dropped_samples_Y[[r]],
          covar=fdat$dropped_sample$dropped_samples_covar[[r]]))
      )
    }

    if (!opt[["skip-twas-weights"]]) {
      twas_out <- twas_weights_pipeline(
        fdat$residual_X[[r]], top_pc_data[, 1],
        susie_fit  = fitted[[r]]$susie_on_top_pc[[1]]$susie_result_trimmed,
        cv_folds   = opt[["twas-cv-folds"]],
        max_cv_variants = opt[["max-cv-variants"]],
        cv_threads = opt[["twas-cv-threads"]]
      )
      fitted[[r]] <- c(fitted[[r]], twas_out)
      fitted[[r]]$twas_weights <- lapply(fitted[[r]]$twas_weights,
                                         function(x) { rownames(x) <- NULL; x })
    }
  }

  fsusie_args <- list(
    X              = fdat$residual_X[[r]],
    Y              = fdat$residual_Y[[r]],
    pos            = fdat$Y_coordinates[[r]]$start,
    L              = opt[["max-l"]],
    prior          = opt$prior,
    max_SNP_EM     = opt[["max-snp-em"]],
    max_scale      = opt[["max-scale"]],
    min_purity     = opt[["min-purity"]],
    cov_lev        = coverage[1],
    post_processing = opt[["post-processing"]]
  )
  if (opt[["small-sample-correction"]]) fsusie_args$cor_small <- TRUE
  fitted[[r]]$fsusie_result <- do.call(fsusie_wrapper, fsusie_args)

  fitted[[r]]$Y_coordinates <- fdat$Y_coordinates[[r]]
  names(fitted[[r]]$fsusie_result$pip) <- colnames(fdat$residual_X[[r]])

  fitted[[r]]$fsusie_summary <- susie_post_processor(
    fitted[[r]]$fsusie_result,
    fdat$residual_X[[r]], NULL,
    fdat$residual_X_scalar[[r]], 1, fdat$maf[[r]],
    secondary_coverage = coverage[-1],
    signal_cutoff      = opt[["pip-cutoff"]],
    other_quantities   = list(dropped_samples=list(
      X=fdat$dropped_sample$dropped_samples_X[[r]],
      y=fdat$dropped_sample$dropped_samples_Y[[r]],
      covar=fdat$dropped_sample$dropped_samples_covar[[r]]))
  )

  fitted[[r]]$total_time_elapsed <- proc.time() - st
  fitted[[r]]$region_info <- list(
    region_coord = parse_region(opt$region),
    grange       = parse_region(opt$window),
    region_name  = opt[["region-name"]]
  )
  fdat$residual_X[[r]] <- NA
  fdat$residual_Y[[r]] <- NA
}

result <- list()
result[[opt$region]] <- fitted
saveRDS(result, paste0(prefix, ".rds"), compress="xz")
