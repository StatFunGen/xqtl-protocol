#!/usr/bin/env Rscript
# gems_pipeline.R — GEMS (Generalized Expression Modifier Score) worker for the
# ems_training / ems_prediction notebooks. Port of gems_pipeline.py.
#
# Steps:
#   train    Train a CatBoost classifier of expression-modifier variants
#            (feature engineering + gene-constraint/MAF merge + imputation +
#            class-balanced weights + feature-weighted CatBoost + 5-fold CV).
#   predict  Not yet implemented (kept as a stub, mirroring the Python).
#
# Config-driven (data_config.yaml + model_config.yaml). The per-feature column
# groups are read from a JSON conversion of the upstream columns_dict pickle
# (jsonlite), so no Python is needed at runtime.
#
# NOTE: R catboost != Python CatBoost, so the model/predictions are NOT byte-
# reproducible against the Python MWE; validation is structural (feature set,
# AP/AUC range, importance ordering).

suppressPackageStartupMessages({
  library(argparser)
  library(yaml)
  library(jsonlite)
  library(readxl)
  library(arrow)
  library(vroom)
  library(catboost)
})

# ---- metric helpers (avoid a sklearn-equivalent dependency) ----------------

auc_score <- function(y, p) {                    # Mann-Whitney U form of ROC AUC
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

average_precision <- function(y, p) {            # AP = sum((R_n - R_{n-1}) * P_n)
  o <- order(p, decreasing = TRUE)
  y <- y[o]
  tp <- cumsum(y == 1); fp <- cumsum(y == 0)
  npos <- sum(y == 1); if (npos == 0) return(NA_real_)
  precision <- tp / (tp + fp); recall <- tp / npos
  drec <- diff(c(0, recall))
  sum(precision * drec)
}

make_variant_features <- function(df) {
  parts <- do.call(rbind, strsplit(df$variant_id, ":", fixed = TRUE))
  ref <- parts[, 3]; alt <- parts[, 4]
  df$length_diff <- nchar(ref) - nchar(alt)
  df$is_SNP <- as.integer(df$length_diff == 0)
  df$is_indel <- as.integer(df$length_diff != 0)
  df$is_insertion <- as.integer(df$length_diff > 0)
  df$is_deletion <- as.integer(df$length_diff < 0)
  df
}

# stratified k-fold row indices (per-class contiguous split, shuffled)
stratified_folds <- function(y, k, seed) {
  set.seed(seed)
  fold <- integer(length(y))
  for (cls in unique(y)) {
    idx <- sample(which(y == cls))
    fold[idx] <- rep(seq_len(k), length.out = length(idx))
  }
  fold
}

# ---- train -----------------------------------------------------------------

do_train <- function(argv) {
  cohort <- argv$cohort
  chromosome <- argv$chromosome
  data_config <- yaml.load_file(argv$data_config)
  model_config <- yaml.load_file(argv$model_config)

  set.seed(model_config$system$random_seeds$numpy_seed)
  NPR_tr <- model_config$experiment$sampling_parameters$npr_train

  chromosome_clean <- gsub("chr", "", chromosome)
  chromosome_out <- paste0("chr", chromosome_clean)
  train_chromosomes <- chromosome_out
  available <- c("1", "2", "3", "5")
  test_cand <- available[available != chromosome_clean]
  if (!length(test_cand)) stop("No different chromosome available for testing")
  test_chromosomes <- paste0("chr", test_cand[1])
  cat(sprintf("Train chrom: %s  Test chrom: %s\n", train_chromosomes, test_chromosomes))

  # -- gene constraint (xlsx) -> gene_id / gene_lof, log2 --------------------
  cc <- data_config$feature_data$gene_constraint
  m <- cc$column_mapping
  gene_lof_df <- as.data.frame(read_excel(cc$file_path, sheet = cc$xlsx_sheet))
  gene_lof_df <- gene_lof_df[, c(m$source_gene_id, m$source_value)]
  names(gene_lof_df) <- c(m$target_gene_id, m$target_value)
  gene_lof_df[[m$target_value]] <- log2(gene_lof_df[[m$target_value]])
  gene_col <- m$target_gene_id; gene_lof_col <- m$target_value

  # -- population MAF (tsv) --------------------------------------------------
  pg <- data_config$feature_data$population_genetics
  pgm <- pg$column_mapping
  maf_file <- gsub("\\{chromosome\\}", chromosome_clean, pg$file_pattern)
  maf_df <- as.data.frame(vroom(maf_file, delim = "\t", show_col_types = FALSE))
  maf_df <- maf_df[, c(pgm$variant_id, pgm$target_value)]
  vid_col <- pgm$variant_id; maf_col <- pgm$target_value

  # -- training / test parquet ----------------------------------------------
  data_dir <- gsub("\\{cohort\\}", cohort, data_config$training_data$base_dir)
  write_dir <- gsub("\\{cohort\\}", cohort, data_config$output$base_dir)
  dir.create(write_dir, recursive = TRUE, showWarnings = FALSE)
  pred_dir <- file.path(write_dir, data_config$output$predictions_dir)
  dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)

  ct <- model_config$experiment$classification_thresholds
  load_split <- function(chrom, dir_pattern, thr) {
    d <- gsub("\\{npr_tr\\}", NPR_tr, gsub("\\{npr_te\\}", model_config$experiment$sampling_parameters$npr_test,
         gsub("\\{pos_threshold\\}", thr$positive_class_threshold,
         gsub("\\{neg_threshold\\}", thr$negative_class_threshold, dir_pattern))))
    fp <- gsub("\\{chromosome\\}", chrom, gsub("\\{cohort\\}", cohort, data_config$training_data$file_pattern))
    path <- file.path(data_dir, d, fp)
    if (!file.exists(path)) stop(sprintf("data file not found: %s", path))
    as.data.frame(read_parquet(path))
  }
  train_df <- load_split(train_chromosomes, data_config$training_data$train_dir_pattern, ct$train)
  test_df  <- load_split(test_chromosomes,  data_config$training_data$test_dir_pattern,  ct$test)

  prep <- function(df) {
    df <- make_variant_features(df)
    df <- merge(df, gene_lof_df, by = gene_col, all.x = TRUE)
    df <- merge(df, maf_df, by = vid_col, all.x = TRUE)
    df
  }
  train_df <- prep(train_df); test_df <- prep(test_df)

  # imputation medians from TRAIN only (no leakage)
  gl_med <- median(train_df[[gene_lof_col]], na.rm = TRUE)
  maf_med <- median(train_df[[maf_col]], na.rm = TRUE)
  for (df_name in c("train_df", "test_df")) {
    df <- get(df_name)
    df[[gene_lof_col]][is.na(df[[gene_lof_col]])] <- gl_med
    df[[maf_col]][is.na(df[[maf_col]])] <- maf_med
    assign(df_name, df)
  }

  # class-balanced weights
  balance <- function(df) {
    c0 <- sum(df$label == 0); c1 <- sum(df$label == 1); tot <- c0 + c1
    w0 <- if (c0 > 0) tot / (2 * c0) else 1; w1 <- if (c1 > 0) tot / (2 * c1) else 1
    df$weight <- ifelse(df$label == 0, w0, w1); df
  }
  train_df <- balance(train_df); test_df <- balance(test_df)

  # feature matrices: drop metadata (+ gene_id), inf/NA -> 0
  meta <- data_config$training_data$metadata_columns
  build_X <- function(df) {
    X <- df[, !names(df) %in% c(meta, gene_col), drop = FALSE]
    num <- vapply(X, is.numeric, logical(1))
    X[num] <- lapply(X[num], function(x) { x[is.infinite(x)] <- 0; x[is.na(x)] <- 0; x })
    X
  }
  X_train <- build_X(train_df); X_test <- build_X(test_df)
  Y_train <- train_df$label; Y_test <- test_df$label

  # subset columns from the column-group dict + variant features, minus removals
  fd <- data_config$feature_data
  column_dict <- fromJSON(fd$distance_features$columns_dict_file, simplifyVector = FALSE)
  subset_keys <- c(unlist(fd$distance_features$subset_keys),
                   unlist(fd$regulatory_features$subset_keys),
                   unlist(fd$deep_learning_features$subset_keys))
  subset_cols <- unlist(lapply(subset_keys, function(k) unlist(column_dict[[k]])))
  subset_cols <- subset_cols[subset_cols %in% names(X_train)]
  subset_cols <- c(subset_cols, unlist(fd$variant_features$generated_columns))
  cols_abs <- intersect(unlist(fd$deep_learning_features$transformations$absolute_value), names(X_train))
  to_remove <- unlist(fd$distance_features$columns_to_remove)

  subset_cols <- setdiff(unique(subset_cols), to_remove)
  X_train_s <- X_train[, subset_cols, drop = FALSE]
  X_test_s  <- X_test[, subset_cols, drop = FALSE]
  for (col in intersect(cols_abs, names(X_train_s))) {
    X_train_s[[col]] <- abs(X_train_s[[col]]); X_test_s[[col]] <- abs(X_test_s[[col]])
  }

  # feature weights: high-priority patterns among the abs columns get high weight
  fw_cfg <- model_config$feature_weighting
  default_w <- fw_cfg$default_weight
  high_w <- fw_cfg$high_priority_patterns$weight
  patterns <- unlist(fw_cfg$high_priority_patterns$feature_patterns)
  feat_w <- setNames(rep(default_w, ncol(X_train_s)), names(X_train_s))
  for (col in intersect(cols_abs, names(feat_w))) {
    if (any(vapply(patterns, function(pt) grepl(pt, col, fixed = TRUE), logical(1)))) feat_w[col] <- high_w
  }

  # -- CatBoost (feature-weighted) ------------------------------------------
  params <- model_config$algorithm$parameter_sets$standard
  params$verbose <- NULL
  params$logging_level <- "Silent"
  params$feature_weights <- unname(feat_w[names(X_train_s)])
  train_pool <- catboost.load_pool(data = X_train_s, label = Y_train, weight = train_df$weight)
  model <- catboost.train(train_pool, params = params)
  test_pool <- catboost.load_pool(data = X_test_s, label = Y_test)
  preds <- catboost.predict(model, test_pool, prediction_type = "Probability")

  ap <- average_precision(Y_test, preds); auc <- auc_score(Y_test, preds)
  cat(sprintf("Test set: AP=%.4f  AUC=%.4f\n", ap, auc))

  catboost.save_model(model, file.path(write_dir,
    sprintf("model_standard_subset_weighted_chr_%s_NPR_%s.cbm", chromosome_out, NPR_tr)))

  imp <- catboost.get_feature_importance(model, pool = train_pool)
  feat_df <- data.frame(feature = rownames(imp), importance = imp[, 1], row.names = NULL)
  feat_df <- feat_df[order(-feat_df$importance), ]
  write.csv(feat_df, file.path(write_dir,
    sprintf("features_importance_model5_chr_%s_NPR_%s.csv", chromosome_out, NPR_tr)), row.names = FALSE)

  # -- 5-fold stratified CV on training data --------------------------------
  cv_folds <- 5
  folds <- stratified_folds(Y_train, cv_folds, model_config$system$random_seeds$numpy_seed)
  cv_ap <- c(); cv_auc <- c()
  for (f in seq_len(cv_folds)) {
    tr <- folds != f; va <- folds == f
    fp <- params
    fp_pool <- catboost.load_pool(data = X_train_s[tr, , drop = FALSE], label = Y_train[tr],
                                  weight = train_df$weight[tr])
    fm <- catboost.train(fp_pool, params = fp)
    vp <- catboost.predict(fm, catboost.load_pool(data = X_train_s[va, , drop = FALSE],
                                                  label = Y_train[va]), prediction_type = "Probability")
    cv_ap <- c(cv_ap, average_precision(Y_train[va], vp))
    cv_auc <- c(cv_auc, auc_score(Y_train[va], vp))
  }
  cat(sprintf("%d-fold CV: AP=%.4f±%.4f  AUC=%.4f±%.4f\n",
              cv_folds, mean(cv_ap), sd(cv_ap), mean(cv_auc), sd(cv_auc)))

  # summary (JSON — modern format, replaces the Python pickle)
  summary_list <- list(CatBoost = list(standard_subset_weighted = list(
    AP_test = ap, AUC_test = auc, params = params,
    cross_validation = list(cv_folds = cv_folds, cv_ap = cv_ap, cv_auc = cv_auc,
                            cv_ap_mean = mean(cv_ap), cv_auc_mean = mean(cv_auc))),
    test_num_positive_labels = sum(Y_test == 1), test_num_negative_labels = sum(Y_test == 0),
    train_standard_num_positive_labels = sum(Y_train == 1),
    train_standard_num_negative_labels = sum(Y_train == 0)))
  write_json(summary_list, file.path(write_dir,
    sprintf("model_5_summary_chr_%s_NPR_%s.json", chromosome_out, NPR_tr)),
    auto_unbox = TRUE, digits = 8)

  # predictions TSV
  test_out <- test_df
  test_out$standard_subset_weighted_pred_prob <- preds
  test_out$standard_subset_weighted_pred_label <- as.integer(catboost.predict(
    model, test_pool, prediction_type = "Class"))
  test_out$actual_label <- Y_test
  vroom_write(test_out, file.path(pred_dir, sprintf("predictions_weighted_model_chr%s.tsv", chromosome)),
              delim = "\t")
  cat("Training complete.\n")
}

do_predict <- function(argv) {
  cat(strrep("=", 80), "\n", sep = "")
  cat("GEMS PREDICTION MODE\n")
  cat(strrep("=", 80), "\n", sep = "")
  cat("\nThis functionality is not yet implemented.\n")
  cat("\nPlanned features:\n")
  cat("  - Load a trained GEMS model from the specified path\n")
  cat("  - Apply the model to new genomic data\n")
  cat("  - Generate expression modifier scores for variants\n")
  cat("  - Export predictions in standard formats\n")
  cat("\nFor now, please use the 'train' subcommand to train models.\n")
  cat(strrep("=", 80), "\n", sep = "")
}

# ---- CLI -------------------------------------------------------------------
p <- arg_parser("GEMS pipeline (train / predict)")
p <- add_argument(p, "--step", help = "train | predict")
p <- add_argument(p, "--cohort", help = "cohort / cell type", default = "protocol_example")
p <- add_argument(p, "--chromosome", help = "chromosome to train on", default = "2")
p <- add_argument(p, "--data-config", help = "data configuration YAML")
p <- add_argument(p, "--model-config", help = "model configuration YAML")
p <- add_argument(p, "--model-path", help = "predict: trained model file")
argv <- parse_args(p)

if (identical(argv$step, "train")) {
  do_train(argv)
} else if (identical(argv$step, "predict")) {
  do_predict(argv)
} else {
  stop("--step must be 'train' or 'predict'")
}
