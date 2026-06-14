#!/usr/bin/env Rscript
# univariate_rss.R - Summary-statistic-based SuSiE RSS fine-mapping.
# Called directly by rss_analysis.ipynb task blocks.
#
# Usage:
#   Rscript univariate_rss.R \
#     --ld-meta-data ld_meta_file.tsv \
#     --studies study1,study2 \
#     --sumstat-paths sumstats1.tsv.gz,sumstats2.tsv.gz \
#     --column-file-paths col1.tsv,col2.tsv \
#     --n-samples 10000,12000 \
#     --n-cases .,. \
#     --n-controls .,. \
#     --region chr10_0_6480000 \
#     --skip-regions "" \
#     --output-prefix /path/to/output/noQC.chr10_0_6480000.univariate \
#     --output /path/to/output/noQC.chr10_0_6480000.univariate.rds

option_spec <- function(flag, type = "character", default = NULL,
                        action = NULL, help = NULL) {
  list(flag = flag, type = type, default = default, action = action, help = help)
}

option_specs <- list(
  option_spec("--ld-meta-data", type = "character"),
  option_spec("--ld-meta-data-paths", type = "character", default = "",
              help = "Comma-separated per-study LD metadata paths; defaults to --ld-meta-data for every study"),
  option_spec("--studies", type = "character", help = "Comma-separated study names"),
  option_spec("--sumstat-paths", type = "character", help = "Comma-separated sumstat file paths"),
  option_spec("--column-file-paths", type = "character", default = "",
              help = "Comma-separated column mapping file paths; use . for none"),
  option_spec("--n-samples", type = "character", default = "",
              help = "Comma-separated per-study sample sizes"),
  option_spec("--n-cases", type = "character", default = "",
              help = "Comma-separated per-study case counts"),
  option_spec("--n-controls", type = "character", default = "",
              help = "Comma-separated per-study control counts"),
  option_spec("--region", type = "character",
              help = "Region string chr:start-end or legacy chr_start_end"),
  option_spec("--skip-regions", type = "character", default = "",
              help = "Comma-separated region IDs to skip"),
  option_spec("--extract-region-name", type = "character", default = "NULL"),
  option_spec("--region-name-col", type = "character", default = "NULL"),
  option_spec("--compute-ld-from-genotype", action = "store_true", default = FALSE),
  option_spec("--imiss", type = "double", default = 1.0),
  option_spec("--maf", type = "double", default = 0.0025),
  option_spec("--ld-reference-size", type = "integer", default = 10000),
  option_spec("--ld-mismatch-correction", action = "store_true", default = FALSE),
  option_spec("--L", type = "integer", default = 15),
  option_spec("--L-greedy", type = "integer", default = 5),
  option_spec("--max-l", type = "integer", default = NA),
  option_spec("--l-step", type = "integer", default = NA),
  option_spec("--pip-cutoff", type = "double", default = 0.025),
  option_spec("--skip-analysis-pip-cutoff", type = "double", default = 0.025),
  option_spec("--coverage", type = "character", default = "0.95,0.7,0.5"),
  option_spec("--finemapping-method", type = "character", default = "single_effect"),
  option_spec("--impute", action = "store_true", default = FALSE),
  option_spec("--rcond", type = "double", default = 0.01),
  option_spec("--lamb", type = "double", default = 0.01),
  option_spec("--r2-threshold", type = "double", default = 0.6),
  option_spec("--minimum-ld", type = "integer", default = 5),
  option_spec("--qc-method", type = "character", default = ""),
  option_spec("--comment-string", type = "character", default = "NULL"),
  option_spec("--diagnostics", action = "store_true", default = FALSE),
  option_spec("--output-prefix", type = "character"),
  option_spec("--output", type = "character")
)

option_name <- function(spec) sub("^--", "", spec$flag)

cast_option_value <- function(value, type) {
  if (identical(type, "integer")) {
    return(as.integer(value))
  }
  if (identical(type, "double")) {
    return(as.numeric(value))
  }
  value
}

print_option_usage <- function(specs) {
  cat("Options:\n")
  for (spec in specs) {
    cat("  ", spec$flag, "\n", sep = "")
  }
}

parse_long_options <- function(specs) {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- setNames(lapply(specs, `[[`, "default"), vapply(specs, option_name, ""))
  flag_map <- setNames(specs, vapply(specs, `[[`, "", "flag"))
  i <- 1
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("--help", "-h")) {
      print_option_usage(specs)
      quit(status = 0)
    }
    if (!startsWith(arg, "--")) {
      stop("Unexpected positional argument: ", arg)
    }
    pieces <- strsplit(arg, "=", fixed = TRUE)[[1]]
    flag <- pieces[[1]]
    value <- if (length(pieces) > 1) paste(pieces[-1], collapse = "=") else NULL
    if (is.null(flag_map[[flag]])) {
      stop("Unknown option: ", flag)
    }
    spec <- flag_map[[flag]]
    name <- option_name(spec)
    if (identical(spec$action, "store_true")) {
      opt[[name]] <- TRUE
      i <- i + 1
      next
    }
    if (is.null(value)) {
      i <- i + 1
      if (i > length(args)) {
        stop("Option requires a value: ", flag)
      }
      value <- args[[i]]
    }
    opt[[name]] <- cast_option_value(value, spec$type)
    i <- i + 1
  }
  opt
}

parse_options <- function(specs) {
  if (requireNamespace("optparse", quietly = TRUE)) {
    option_list <- lapply(specs, function(spec) {
      optparse::make_option(
        spec$flag, type = spec$type, default = spec$default,
        action = spec$action, help = spec$help
      )
    })
    return(optparse::parse_args(optparse::OptionParser(option_list = option_list)))
  }
  parse_long_options(specs)
}

opt <- parse_options(option_specs)

suppressPackageStartupMessages({
  library(pecotmr)
  library(dplyr)
  library(data.table)
})

studies <- strsplit(opt$studies, ",")[[1]]
sumstat_paths <- strsplit(opt[["sumstat-paths"]], ",")[[1]]

col_paths <- if (nchar(opt[["column-file-paths"]]) > 0) {
  strsplit(opt[["column-file-paths"]], ",")[[1]]
} else {
  rep("", length(studies))
}
col_paths[col_paths == "."] <- ""

parse_numeric_vec <- function(s, n) {
  if (nchar(s) == 0) {
    return(rep(NA_real_, n))
  }
  v <- strsplit(s, ",")[[1]]
  v[v == "."] <- NA_character_
  as.numeric(v)
}

n_samples <- parse_numeric_vec(opt[["n-samples"]], length(studies))
n_cases <- parse_numeric_vec(opt[["n-cases"]], length(studies))
n_controls <- parse_numeric_vec(opt[["n-controls"]], length(studies))

parse_character_vec <- function(s, n, default) {
  if (nchar(s) == 0) {
    return(rep(default, n))
  }
  v <- strsplit(s, ",")[[1]]
  v[v == "."] <- default
  if (length(v) == 1) {
    return(rep(v, n))
  }
  if (length(v) != n) {
    stop("Expected ", n, " LD metadata paths, got ", length(v))
  }
  v
}

ld_meta_paths <- parse_character_vec(
  opt[["ld-meta-data-paths"]], length(studies), opt[["ld-meta-data"]])

if (!is.na(opt[["max-l"]])) {
  opt$L <- opt[["max-l"]]
}

skip_region_vec <- if (nchar(opt[["skip-regions"]]) > 0) {
  strsplit(opt[["skip-regions"]], ",")[[1]]
} else {
  character(0)
}

coverage <- as.numeric(strsplit(opt$coverage, ",")[[1]])

parse_region_arg <- function(region) {
  region_body <- sub("^chr", "", region)
  if (grepl(":", region_body, fixed = TRUE)) {
    region_parts <- strsplit(region_body, ":", fixed = TRUE)[[1]]
    region_se <- strsplit(region_parts[[2]], "-", fixed = TRUE)[[1]]
  } else {
    region_parts <- strsplit(region_body, "_", fixed = TRUE)[[1]]
    region_se <- region_parts[2:3]
  }
  if (length(region_parts) < 2 || length(region_se) != 2) {
    stop("Invalid --region format: ", region,
         ". Expected chr:start-end or legacy chr_start_end.")
  }
  region_chr <- region_parts[[1]]
  region_start <- as.integer(region_se[[1]])
  region_end <- as.integer(region_se[[2]])
  if (is.na(region_start) || is.na(region_end)) {
    stop("Invalid --region coordinates: ", region)
  }
  list(
    chr = region_chr,
    start = region_start,
    end = region_end,
    coord = paste0("chr", region_chr, ":", region_start, "-", region_end),
    key = paste0("chr", region_chr, "_", region_start, "_", region_end)
  )
}

region_info <- parse_region_arg(opt$region)
region_chr <- region_info$chr
region_start <- region_info$start
region_end <- region_info$end
region_coord <- region_info$coord

extract_region_name <- if (opt[["extract-region-name"]] == "NULL") {
  NULL
} else {
  opt[["extract-region-name"]]
}
region_name_col <- if (opt[["region-name-col"]] == "NULL") {
  NULL
} else {
  opt[["region-name-col"]]
}

load_ld_for_region <- function(ld_meta_path, region_coord) {
  if (opt[["compute-ld-from-genotype"]]) {
    geno_path <- readr::read_delim(ld_meta_path, "\t", show_col_types = FALSE) %>%
      filter(`#chr` == region_chr, start == region_start, end == region_end) %>%
      pull(path) %>%
      stringr::str_replace(".bed", "")
    LD_data <- pecotmr:::filter_X(
      load_genotype_region(geno_path, region_coord),
      missing_rate_thresh = opt$imiss,
      maf_thresh = opt$maf
    ) %>%
      cor()
    correct_variants <- rownames(LD_data)[sapply(rownames(LD_data), function(x) {
      sum(strsplit(x, "", fixed = TRUE)[[1]] == ":") == 3
    })]
    LD_data <- LD_data[correct_variants, correct_variants]
    return(list(combined_LD_matrix = LD_data,
                combined_LD_variants = rownames(LD_data)))
  }

  if (exists("load_study_LD", mode = "function")) {
    load_study_LD(ld_meta_path, region_coord)
  } else {
    load_LD_matrix(ld_meta_path, region_coord)
  }
}

res <- setNames(replicate(length(studies), list(), simplify = FALSE), studies)
ld_cache <- list()
study_errors <- character(0)

finemapping_l <- opt[["L-greedy"]]
finemapping_max_l <- opt$L
finemapping_l_step <- if (is.na(opt[["l-step"]])) {
  opt[["L-greedy"]]
} else {
  opt[["l-step"]]
}

for (r in seq_along(res)) {
  tryCatch({
    ld_key <- ld_meta_paths[r]
    if (is.null(ld_cache[[ld_key]])) {
      ld_cache[[ld_key]] <- load_ld_for_region(ld_key, region_coord)
    }
    LD_data <- ld_cache[[ld_key]]

    rss_args <- list(
      sumstat_path = sumstat_paths[r],
      column_file_path = col_paths[r],
      LD_data = LD_data,
      extract_region_name = extract_region_name,
      region_name_col = region_name_col,
      n_sample = n_samples[r],
      n_case = n_cases[r],
      n_control = n_controls[r],
      skip_region = if (length(skip_region_vec) == 0) NULL else skip_region_vec,
      qc_method = if (nchar(opt[["qc-method"]]) == 0) NULL else opt[["qc-method"]],
      impute = opt$impute,
      impute_opts = list(
        rcond = opt$rcond,
        R2_threshold = opt[["r2-threshold"]],
        minimum_ld = opt[["minimum-ld"]],
        lamb = opt$lamb
      ),
      finemapping_method = if (nchar(opt[["finemapping-method"]]) == 0) {
        NULL
      } else {
        opt[["finemapping-method"]]
      },
      finemapping_opts = list(
        L = finemapping_max_l,
        L_greedy = finemapping_l,
        init_L = finemapping_l,
        max_L = finemapping_max_l,
        l_step = finemapping_l_step,
        coverage = coverage,
        signal_cutoff = opt[["pip-cutoff"]],
        min_abs_corr = 0.8
      ),
      pip_cutoff_to_skip = opt[["skip-analysis-pip-cutoff"]],
      comment_string = if (opt[["comment-string"]] == "NULL") {
        NULL
      } else {
        opt[["comment-string"]]
      }
    )
    rss_formals <- names(formals(rss_analysis_pipeline))
    accepts_extra_args <- "..." %in% rss_formals
    if (accepts_extra_args || "diagnostics" %in% rss_formals) {
      rss_args$diagnostics <- opt$diagnostics
    }
    if (accepts_extra_args || "R_finite" %in% rss_formals) {
      rss_args$R_finite <- opt[["ld-reference-size"]]
    }
    if (accepts_extra_args || "R_mismatch" %in% rss_formals) {
      rss_args$R_mismatch <- if (opt[["ld-mismatch-correction"]]) "eb" else NULL
    }
    res[[r]] <- do.call(rss_analysis_pipeline, rss_args)
    region_label <- paste0(
      opt[["output-prefix"]], ".",
      if (is.null(extract_region_name)) "" else paste0(extract_region_name, "."),
      studies[r], ".sumstats.tsv.gz")
    fwrite(res[[r]]$rss_data_analyzed, file = region_label,
           sep = "\t", col.names = TRUE, row.names = FALSE,
           quote = FALSE, compress = "gzip")
    if (is.null(res[[r]][[1]])) {
      res[[r]] <- list()
    }
  }, error = function(e) {
    res[[r]] <<- list()
    study_errors[[studies[r]]] <<- conditionMessage(e)
    message("Error processing study ", studies[r], ": ", conditionMessage(e))
  })
}

if (length(study_errors) > 0) {
  stop(
    "RSS analysis failed for ", length(study_errors), " study/studies: ",
    paste(names(study_errors), study_errors, sep = "=", collapse = "; ")
  )
}

region_key <- region_info$key
full_result <- list()
full_result[[region_key]] <- res
saveRDS(full_result, file = opt$output)
