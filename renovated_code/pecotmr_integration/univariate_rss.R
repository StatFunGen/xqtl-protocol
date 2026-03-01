#!/usr/bin/env Rscript
# univariate_rss.R — Summary-statistic-based SuSiE RSS fine-mapping
# Called directly by route3 SoS notebook task blocks.
#
# Usage:
#   Rscript univariate_rss.R \
#     --ld-meta-data ld_meta_file.tsv \
#     --studies study1,study2 \
#     --sumstat-paths sumstats1.tsv.gz,sumstats2.tsv.gz \
#     --column-file-paths col1.tsv,col2.tsv \
#     --n-samples 10000,12000 \
#     --n-cases . ,. \
#     --n-controls .,. \
#     --region chr10:0-6480000 \
#     --skip-regions "" \
#     --output-prefix /path/to/output/noQC.chr10_0_6480000.univariate \
#     --output /path/to/output/noQC.chr10_0_6480000.univariate.rds

suppressPackageStartupMessages(library(optparse))

option_list <- list(
  # LD and summary stats inputs
  make_option("--ld-meta-data",       type="character"),
  make_option("--studies",            type="character", help="Comma-separated study names"),
  make_option("--sumstat-paths",      type="character", help="Comma-separated sumstat file paths"),
  make_option("--column-file-paths",  type="character", default="",
              help="Comma-separated column mapping file paths (use '.' for none)"),
  make_option("--n-samples",          type="character", default="",
              help="Comma-separated per-study sample sizes"),
  make_option("--n-cases",            type="character", default="",
              help="Comma-separated per-study case counts"),
  make_option("--n-controls",         type="character", default="",
              help="Comma-separated per-study control counts"),
  # region
  make_option("--region",             type="character",
              help="Region string chr:start-end"),
  make_option("--skip-regions",       type="character", default="",
              help="Comma-separated region IDs to skip"),
  make_option("--extract-region-name",type="character", default="NULL"),
  make_option("--region-name-col",    type="character", default="NULL"),
  # LD options
  make_option("--compute-ld-from-genotype", action="store_true", default=FALSE),
  make_option("--imiss",  type="double",  default=1.0),
  make_option("--maf",    type="double",  default=0.0025),
  # fine-mapping params
  make_option("--L",         type="integer", default=5),
  make_option("--max-l",     type="integer", default=10),
  make_option("--l-step",    type="integer", default=5),
  make_option("--pip-cutoff",type="double",  default=0.025),
  make_option("--skip-analysis-pip-cutoff", type="double", default=0.025),
  make_option("--coverage",  type="character", default="0.95,0.7,0.5"),
  make_option("--finemapping-method", type="character", default="single_effect"),
  # imputation params
  make_option("--impute",      action="store_true", default=FALSE),
  make_option("--rcond",       type="double", default=0.01),
  make_option("--lamb",        type="double", default=0.01),
  make_option("--r2-threshold",type="double", default=0.6),
  make_option("--minimum-ld",  type="integer",default=5),
  # QC
  make_option("--qc-method",   type="character", default=""),
  make_option("--comment-string", type="character", default="NULL"),
  make_option("--diagnostics", action="store_true", default=FALSE),
  # output
  make_option("--output-prefix", type="character"),
  make_option("--output",        type="character")
)

opt <- parse_args(OptionParser(option_list=option_list))

library(pecotmr)
library(dplyr)
library(data.table)

studies       <- strsplit(opt$studies,           ",")[[1]]
sumstat_paths <- strsplit(opt[["sumstat-paths"]], ",")[[1]]

col_paths <- if (nchar(opt[["column-file-paths"]]) > 0) {
  strsplit(opt[["column-file-paths"]], ",")[[1]]
} else rep("", length(studies))
col_paths[col_paths == "."] <- ""

parse_numeric_vec <- function(s, n) {
  if (nchar(s) == 0) return(rep(NA_real_, n))
  v <- strsplit(s, ",")[[1]]
  v[v == "."] <- NA_character_
  as.numeric(v)
}

n_samples  <- parse_numeric_vec(opt[["n-samples"]],  length(studies))
n_cases    <- parse_numeric_vec(opt[["n-cases"]],    length(studies))
n_controls <- parse_numeric_vec(opt[["n-controls"]], length(studies))

skip_region_vec <- if (nchar(opt[["skip-regions"]]) > 0) {
  strsplit(opt[["skip-regions"]], ",")[[1]]
} else character(0)

coverage <- as.numeric(strsplit(opt$coverage, ",")[[1]])

# Parse region string into chr/start/end
region_parts <- strsplit(sub("chr", "", opt$region), ":")[[1]]
region_chr   <- region_parts[1]
region_se    <- strsplit(region_parts[2], "-")[[1]]
region_start <- as.integer(region_se[1])
region_end   <- as.integer(region_se[2])
region_coord <- paste0("chr", region_chr, ":", region_start, "-", region_end)

extract_region_name <- if (opt[["extract-region-name"]] == "NULL") NULL else opt[["extract-region-name"]]
region_name_col     <- if (opt[["region-name-col"]] == "NULL") NULL else opt[["region-name-col"]]

# Load LD matrix
if (opt[["compute-ld-from-genotype"]]) {
  geno_path <- readr::read_delim(opt[["ld-meta-data"]], "\t") %>%
    filter(`#chr` == region_chr, start == region_start, end == region_end) %>%
    pull(path) %>%
    stringr::str_replace(".bed", "")
  LD_data <- pecotmr:::filter_X(
    load_genotype_region(geno_path, region_coord),
    missing_rate_thresh = opt$imiss, maf_thresh = opt$maf
  ) %>% cor
  correct_variants <- rownames(LD_data)[sapply(rownames(LD_data), function(x)
    sum(strsplit(x, "", fixed=TRUE)[[1]] == ":") == 3)]
  LD_data <- LD_data[correct_variants, correct_variants]
  LD_data <- list(combined_LD_matrix=LD_data, combined_LD_variants=rownames(LD_data))
} else {
  LD_data <- load_LD_matrix(opt[["ld-meta-data"]], region_coord)
}

res <- setNames(replicate(length(studies), list(), simplify=FALSE), studies)

for (r in seq_along(res)) {
  tryCatch({
    res[[r]] <- rss_analysis_pipeline(
      sumstat_path    = sumstat_paths[r],
      column_file_path = col_paths[r],
      LD_data         = LD_data,
      extract_region_name = extract_region_name,
      region_name_col = region_name_col,
      n_sample        = n_samples[r],
      n_case          = n_cases[r],
      n_control       = n_controls[r],
      skip_region     = skip_region_vec,
      qc_method       = if (nchar(opt[["qc-method"]]) == 0) NULL else opt[["qc-method"]],
      impute          = opt$impute,
      impute_opts     = list(
        rcond          = opt$rcond,
        R2_threshold   = opt[["r2-threshold"]],
        minimum_ld     = opt[["minimum-ld"]],
        lamb           = opt$lamb
      ),
      finemapping_method = if (nchar(opt[["finemapping-method"]]) == 0) NULL else opt[["finemapping-method"]],
      finemapping_opts   = list(
        init_L        = opt$L,
        max_L         = opt[["max-l"]],
        l_step        = opt[["l-step"]],
        coverage      = coverage,
        signal_cutoff = opt[["pip-cutoff"]]
      ),
      pip_cutoff_to_skip = opt[["skip-analysis-pip-cutoff"]],
      comment_string  = if (opt[["comment-string"]] == "NULL") NULL else opt[["comment-string"]],
      diagnostics     = opt$diagnostics
    )
    region_label <- paste0(opt[["output-prefix"]], ".", extract_region_name, studies[r], ".sumstats.tsv.gz")
    fwrite(res[[r]]$rss_data_analyzed, file=region_label,
           sep="\t", col.names=TRUE, row.names=FALSE, quote=FALSE, compress="gzip")
    if (is.null(res[[r]][[1]])) res[[r]] <- list()
  }, error=function(e) {
    res[[r]] <<- list()
    message("Error processing study ", studies[r], ": ", conditionMessage(e))
  })
}

region_key <- paste0("chr", gsub(":", "_", sub("chr", "", opt$region)))
full_result <- list()
full_result[[region_key]] <- res
saveRDS(full_result, file=opt$output)
