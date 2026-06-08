#!/usr/bin/env Rscript
# susie_twas.R — Univariate SuSiE fine-mapping and TWAS weights
# Called directly by modular_sos SoS notebook task blocks.
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

script_started_at <- Sys.time()
raw_command_args <- commandArgs(trailingOnly = FALSE)

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
  make_option("--L-greedy",     type="integer",   default=NA,
              help="Alias for --init-l used by newer xqtl-protocol notebooks"),
  make_option("--L",            type="integer",   default=NA,
              help="Alias for --max-l used by newer xqtl-protocol notebooks"),
  make_option("--estimate-residual-method", type="character", default="",
              help="Optional SuSiE estimate_residual_method passed through pecotmr"),
  make_option("--small-sample-correction", action="store_true", default=FALSE,
              help="Use the supported susieR small-sample residual-variance path for univariate SuSiE"),
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

if (!is.na(opt[["L-greedy"]])) {
  opt[["init-l"]] <- opt[["L-greedy"]]
}
if (!is.na(opt[["L"]])) {
  opt[["max-l"]] <- opt[["L"]]
}

library(pecotmr)

normalize_path_safe <- function(path) {
  if (is.null(path) || length(path) == 0) return(character(0))
  normalized <- vapply(path, function(entry) {
    if (is.na(entry) || entry == "") return("")
    normalizePath(entry, winslash = "/", mustWork = FALSE)
  }, character(1))
  normalized[nzchar(normalized)]
}

safe_system_capture <- function(command, args = character()) {
  tryCatch(
    paste(system2(command, args = args, stdout = TRUE, stderr = FALSE), collapse = "\n"),
    error = function(e) ""
  )
}

safe_package_version <- function(pkg) {
  tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
}

quote_command_args <- function(args) {
  paste(vapply(args, shQuote, character(1)), collapse = " ")
}

expand_input_paths <- function(path) {
  if (is.null(path) || length(path) == 0 || is.na(path) || path == "" || path == "." || !file.exists(path)) {
    return(character(0))
  }

  paths <- path
  prefix <- sub("\\.(bed|bim|fam|pgen|pvar|psam)$", "", path, ignore.case = TRUE)
  companions <- c(paste0(prefix, ".bed"), paste0(prefix, ".bim"), paste0(prefix, ".fam"))
  companions <- companions[file.exists(companions)]
  paths <- c(paths, companions)

  if (grepl("\\.(txt|tsv|tab)$", path, ignore.case = TRUE) && !grepl("\\.(bed|bim|fam)$", path, ignore.case = TRUE)) {
    manifest <- tryCatch(
      read.delim(path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE, comment.char = ""),
      error = function(e) NULL
    )
    if (!is.null(manifest) && ncol(manifest) >= 2) {
      manifest_paths <- manifest[[2]]
      manifest_paths <- manifest_paths[nzchar(manifest_paths) & !is.na(manifest_paths)]
      if (length(manifest_paths) > 0) {
        for (manifest_path in manifest_paths) {
          paths <- c(paths, expand_input_paths(manifest_path))
        }
      }
    }
  }

  unique(normalize_path_safe(paths))
}

build_input_records <- function(paths) {
  input_paths <- unique(unlist(lapply(paths, expand_input_paths), use.names = FALSE))
  lapply(input_paths, function(path) {
    info <- file.info(path)
    list(
      path = normalize_path_safe(path),
      md5 = unname(tools::md5sum(path)),
      size = unname(info$size)
    )
  })
}

format_provenance_lines <- function(provenance) {
  lines <- c(
    paste0("generated_at: ", provenance$generated_at),
    paste0("output_kind: ", provenance$output_kind),
    paste0("analysis_script: ", provenance$analysis_script),
    paste0("working_directory: ", provenance$working_directory),
    paste0("script_path: ", provenance$script_path),
    paste0("git_commit: ", provenance$environment$git_commit),
    paste0("R_version: ", provenance$environment$r_version),
    paste0("pecotmr_version: ", provenance$environment$packages$pecotmr),
    paste0("optparse_version: ", provenance$environment$packages$optparse),
    "parameters:"
  )
  lines <- c(lines, vapply(names(provenance$parameters), function(name) {
    value <- provenance$parameters[[name]]
    if (length(value) > 1) value <- paste(value, collapse = ",")
    paste0("  ", name, ": ", value)
  }, character(1)))
  lines <- c(lines, "inputs:")
  if (length(provenance$inputs) == 0) {
    lines <- c(lines, "  <none>")
  } else {
    lines <- c(lines, vapply(provenance$inputs, function(record) {
      paste0("  ", record$md5, "  ", record$path)
    }, character(1)))
  }
  lines
}

write_provenance_sidecars <- function(output_path, provenance) {
  stem <- sub("\\.rds$", "", output_path)
  writeLines(provenance$analysis_script, paste0(stem, ".analysis_script.txt"))
  input_lines <- c("md5\tpath", vapply(provenance$inputs, function(record) {
    paste(record$md5, record$path, sep = "\t")
  }, character(1)))
  writeLines(input_lines, paste0(stem, ".inputs.md5"))
  writeLines(format_provenance_lines(provenance), paste0(stem, ".provenance.txt"))
}

attach_provenance <- function(object, provenance) {
  attr(object, "provenance") <- provenance
  attr(object, "analysis_script") <- provenance$analysis_script
  if (is.list(object) && length(object) > 0) {
    object[] <- lapply(object, function(entry) {
      attr(entry, "provenance") <- provenance
      attr(entry, "analysis_script") <- provenance$analysis_script
      entry
    })
  }
  object
}

normalize_genotype_prefix <- function(path) {
  sub("\\.(bed|bim|fam|pgen|pvar|psam)$", "", path, ignore.case = TRUE)
}

phenotype_files <- strsplit(opt$phenotype, ",")[[1]]
covariate_files <- strsplit(opt$covariate, ",")[[1]]
conditions      <- if (nchar(opt$conditions) > 0) strsplit(opt$conditions, ",")[[1]] else character(0)
coverage        <- as.numeric(strsplit(opt$coverage, ",")[[1]])
genotype_prefix <- normalize_genotype_prefix(opt$genotype)
finemapping_extra_opts <- list(refine = TRUE)
if (nzchar(opt[["estimate-residual-method"]])) {
  finemapping_extra_opts$estimate_residual_method <- opt[["estimate-residual-method"]]
}
if (opt[["small-sample-correction"]]) {
  if (is.null(finemapping_extra_opts$estimate_residual_method)) {
    finemapping_extra_opts$estimate_residual_method <- "Servin_Stephens"
  }
  finemapping_extra_opts$estimate_prior_method <- "EM"
  finemapping_extra_opts$convergence_method <- "pip"
  finemapping_extra_opts$refine <- FALSE
  message(
    "Small-sample correction requested for univariate SuSiE; ",
    "using estimate_residual_method='",
    finemapping_extra_opts$estimate_residual_method,
    "' with refine=FALSE. The legacy cor_small argument is only accepted by fSuSiE, not susieR::susie."
  )
}

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
keep_samples <- if (file.exists(opt[["keep-samples"]]) && !dir.exists(opt[["keep-samples"]])) {
  message(paste("Loading keep_samples from", opt[["keep-samples"]]))
  unlist(strsplit(readLines(opt[["keep-samples"]]), "\\s+"))
} else NULL

keep_variants <- if (file.exists(opt[["keep-variants"]]) && !dir.exists(opt[["keep-variants"]])) opt[["keep-variants"]] else NULL

ld_ref <- if (file.exists(opt[["ld-reference-meta-file"]]) && !dir.exists(opt[["ld-reference-meta-file"]])) opt[["ld-reference-meta-file"]] else NULL

# Infer phenotype header width from the BED itself. Modular SoS expression BEDs can
# include a strand column after ID, which must not be parsed as a sample.
infer_phenotype_header <- function(path, is_coordinate_region) {
  if (!is_coordinate_region) {
    return(1L)
  }
  con <- if (grepl("\\.gz$", path)) gzfile(path, "rt") else file(path, "rt")
  on.exit(close(con), add = TRUE)
  header <- readLines(con, n = 1L, warn = FALSE)
  fields <- strsplit(header, "\t", fixed = TRUE)[[1]]
  if (length(fields) >= 5L && fields[5] == "strand") {
    return(5L)
  }
  4L
}

region_end  <- as.integer(sub(".*-", "", opt$region))
is_coordinate_region <- !is.na(region_end) && region_end > 0
pheno_header <- infer_phenotype_header(phenotype_files[[1]], is_coordinate_region)
region_name_col <- if (is_coordinate_region) 4L else 1L

region_str <- if (!opt[["trans-analysis"]]) opt$region else NULL

write_empty_region_outputs <- function(message_text) {
  message("Error: ", paste(message_text, paste0(opt[["region-name"]], "@", opt$window)))
  region_name_sym <- opt[["region-name"]]
  result <- list()
  result[[region_name_sym]] <- message_text

  if (opt[["skip-fine-mapping"]] && opt[["skip-twas-weights"]]) {
    data_dir <- file.path(opt$cwd, "data")
    dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(result, file.path(data_dir, paste0(basename(opt[["output-prefix"]]), ".univariate_data.rds")), compress = "xz")
  }
  if (!opt[["skip-fine-mapping"]]) {
    fm_dir <- file.path(opt$cwd, "fine_mapping")
    dir.create(fm_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(result, file.path(fm_dir, paste0(basename(opt[["output-prefix"]]), ".univariate_bvsr.rds")), compress = "xz")
  }
  if (!opt[["skip-twas-weights"]]) {
    tw_dir <- file.path(opt$cwd, "twas_weights")
    dir.create(tw_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(result, file.path(tw_dir, paste0(basename(opt[["output-prefix"]]), ".univariate_twas_weights.rds")), compress = "xz")
  }
  quit(save = "no", status = 0)
}

tryCatch({
  fdat <- load_regional_univariate_data(
    genotype          = genotype_prefix,
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
    region_name_col   = region_name_col,
    scale_residuals   = FALSE
  )
}, NoSNPsError = function(e) {
  write_empty_region_outputs(e$message)
}, error = function(e) {
  msg <- conditionMessage(e)
  if (grepl("subscript out of bounds", msg, fixed = TRUE)) {
    write_empty_region_outputs(paste("No analyzable variants after genotype QC:", msg))
  }
  stop(e)
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

# Preserve the old notebook contract: region_name contains the primary region
# plus the extracted region names, even when that duplicates the same ID.
region_name_vec <- if (length(extract_region_name) > 0) {
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
        finemapping_extra_opts = finemapping_extra_opts,
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
        finemapping_extra_opts = finemapping_extra_opts,
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

build_provenance <- function(output_kind, output_path) {
  script_path_arg <- raw_command_args[grep("^--file=", raw_command_args)]
  script_path <- if (length(script_path_arg) > 0) sub("^--file=", "", script_path_arg[[1]]) else NA_character_
  input_records <- build_input_records(c(
    opt$genotype,
    phenotype_files,
    covariate_files,
    opt[["keep-samples"]],
    opt[["keep-variants"]],
    opt[["ld-reference-meta-file"]]
  ))
  list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    output_kind = output_kind,
    output_path = normalize_path_safe(output_path),
    analysis_script = quote_command_args(raw_command_args),
    working_directory = normalize_path_safe(getwd()),
    script_path = normalize_path_safe(script_path),
    parameters = lapply(opt, function(value) {
      if (length(value) == 0) return("")
      if (is.logical(value) || is.numeric(value) || is.character(value)) return(value)
      as.character(value)
    }),
    inputs = input_records,
    environment = list(
      r_version = R.version.string,
      packages = list(
        pecotmr = safe_package_version("pecotmr"),
        optparse = safe_package_version("optparse")
      ),
      env_vars = list(
        HOME = Sys.getenv("HOME", unset = ""),
        PIXI_HOME = Sys.getenv("PIXI_HOME", unset = ""),
        TMPDIR = Sys.getenv("TMPDIR", unset = "")
      ),
      git_commit = safe_system_capture("git", c("rev-parse", "HEAD"))
    ),
    timing = list(
      started_at = format(script_started_at, "%Y-%m-%dT%H:%M:%S%z"),
      finished_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
    )
  )
}

if (!opt[["skip-fine-mapping"]]) {
  fm_dir <- file.path(opt$cwd, "fine_mapping")
  dir.create(fm_dir, recursive=TRUE, showWarnings=FALSE)
  output_path <- file.path(fm_dir, paste0(basename(prefix), ".univariate_bvsr.rds"))
  provenance <- build_provenance("fine_mapping", output_path)
  result <- list()
  result[[region_name_sym]] <- attach_provenance(finemapping_output, provenance)
  attr(result, "provenance") <- provenance
  attr(result, "analysis_script") <- provenance$analysis_script
  saveRDS(result, output_path, compress="xz")
  write_provenance_sidecars(output_path, provenance)
}

if (!opt[["skip-twas-weights"]]) {
  tw_dir <- file.path(opt$cwd, "twas_weights")
  dir.create(tw_dir, recursive=TRUE, showWarnings=FALSE)
  output_path <- file.path(tw_dir, paste0(basename(prefix), ".univariate_twas_weights.rds"))
  provenance <- build_provenance("twas_weights", output_path)
  result <- list()
  result[[region_name_sym]] <- attach_provenance(twas_output, provenance)
  attr(result, "provenance") <- provenance
  attr(result, "analysis_script") <- provenance$analysis_script
  saveRDS(result, output_path, compress="xz")
  write_provenance_sidecars(output_path, provenance)
}
