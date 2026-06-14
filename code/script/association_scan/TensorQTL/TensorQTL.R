#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(purrr)
  library(tidyr)
  library(readr)
  library(dplyr)
  library(qvalue)
})

fallback_qvalues <- function(pvalues) {
  tryCatch({
    qvalue(pvalues, pi0 = 1, lfdr.out = FALSE)$qvalues
  }, error = function(e) {
    message("qvalue(pi0 = 1) failed; using BH")
    p.adjust(pvalues, method = "BH")
  })
}

compute_qvalues <- function(pvalues) {
  output <- rep(NA_real_, length(pvalues))
  keep <- !is.na(pvalues)
  if (sum(keep) == 0) {
    return(output)
  }
  if (sum(keep) < 2) {
    output[keep] <- pvalues[keep]
    return(output)
  }

  clean_pvalues <- pvalues[keep]
  tryCatch({
    output[keep] <- qvalue(clean_pvalues)$qvalues
    output
  }, error = function(e) {
    message("Too few p-values to calculate qvalue; using qvalue(pi0 = 1)")
    output[keep] <- fallback_qvalues(clean_pvalues)
    output
  })
}

resolve_interaction_name <- function(interaction, pairs_df) {
  if (!is.null(interaction) && nzchar(interaction)) {
    if (file.exists(interaction)) {
      interaction_df <- read_delim(interaction, delim = "\t", show_col_types = FALSE)
      if (ncol(interaction_df) < 2) {
        stop(sprintf("Interaction file has no value columns: %s", interaction), call. = FALSE)
      }
      return(names(interaction_df)[2])
    }
    return(interaction)
  }

  interaction_cols <- grep("^pvalue_.+_interaction$", names(pairs_df), value = TRUE)
  if (length(interaction_cols) == 0) {
    return("")
  }
  sub("^pvalue_(.+)_interaction$", "\\1", interaction_cols[1])
}

apply_nominal_qvalues <- function(file_path, interaction = "") {
  pairs_df <- read_delim(file_path, delim = "\t", show_col_types = FALSE)
  interaction_name <- resolve_interaction_name(interaction, pairs_df)

  if (nzchar(interaction_name)) {
    interaction_pvalue_col <- sprintf("pvalue_%s_interaction", interaction_name)
    if (!interaction_pvalue_col %in% names(pairs_df)) {
      stop(sprintf("Missing interaction p-value column: %s", interaction_pvalue_col), call. = FALSE)
    }
    pairs_df <- pairs_df %>%
      group_by(molecular_trait_id) %>%
      mutate(
        qvalue_main = compute_qvalues(pvalue),
        qvalue_interaction = compute_qvalues(.data[[interaction_pvalue_col]])
      )
  } else {
    pairs_df <- pairs_df %>%
      group_by(molecular_trait_id) %>%
      mutate(qvalue = compute_qvalues(pvalue))
  }

  pairs_df %>% ungroup() %>% write_delim(file_path, "\t")
}

apply_trans_qvalues <- function(file_path) {
  if (!file.exists(file_path) || file.info(file_path)$size == 0) {
    return(invisible(NULL))
  }
  first_line <- readLines(file_path, n = 1, warn = FALSE)
  if (length(first_line) == 0 || !grepl("\t", first_line)) {
    return(invisible(NULL))
  }

  pairs_df <- read_delim(file_path, delim = "\t", show_col_types = FALSE)
  if (!all(c("molecular_trait_id", "pvalue") %in% names(pairs_df))) {
    stop("Trans qvalue input must contain molecular_trait_id and pvalue columns", call. = FALSE)
  }
  pairs_df <- pairs_df %>%
    group_by(molecular_trait_id) %>%
    mutate(qvalue = compute_qvalues(pvalue)) %>%
    ungroup()
  pairs_df %>% write_delim(file_path, "\t")
}

run_regional_postprocess <- function(out_tsv, out_summary, input_files) {
  empirical_pd <- map_dfr(input_files, ~read_delim(.x, "\t", show_col_types = FALSE))

  empirical_pd["q_beta"] <- tryCatch(
    qvalue(empirical_pd$p_beta)$qvalues,
    error = function(e) {
      print("Too few pvalue to calculate qvalue; using qvalue(pi0 = 1)")
      fallback_qvalues(empirical_pd$p_beta)
    }
  )
  empirical_pd["q_perm"] <- tryCatch(
    qvalue(empirical_pd$p_perm)$qvalues,
    error = function(e) {
      print("Too few pvalue to calculate qvalue; using qvalue(pi0 = 1)")
      fallback_qvalues(empirical_pd$p_perm)
    }
  )
  empirical_pd["fdr_beta"] <- p.adjust(empirical_pd$p_beta, "fdr")
  empirical_pd["fdr_perm"] <- p.adjust(empirical_pd$p_perm, "fdr")

  if (!all(is.na(empirical_pd$p_beta))) {
    lb <- empirical_pd %>% filter(q_beta <= 0.05) %>% pull(p_beta) %>% sort()
    ub <- empirical_pd %>% filter(q_beta > 0.05) %>% pull(p_beta) %>% sort()
    if (length(lb) > 0) {
      lb_val <- tail(lb, 1)
      threshold <- if (length(ub) > 0) (lb_val + head(ub, 1)) / 2 else lb_val
      message(sprintf("min p-value threshold @ FDR 0.05: %g", threshold))
      empirical_pd <- empirical_pd %>%
        mutate(p_nominal_threshold = qbeta(threshold, beta_shape1, beta_shape2))
    }
  }

  summary <- tibble(
    "fdr_perm_0.05" = sum(empirical_pd["fdr_perm"] < 0.05),
    "fdr_beta_0.05" = sum(empirical_pd["fdr_beta"] < 0.05),
    "q_perm_0.05" = sum(empirical_pd["q_perm"] < 0.05),
    "q_beta_0.05" = sum(empirical_pd["q_beta"] < 0.05),
    "fdr_perm_0.01" = sum(empirical_pd["fdr_perm"] < 0.01),
    "fdr_beta_0.01" = sum(empirical_pd["fdr_beta"] < 0.01),
    "q_perm_0.01" = sum(empirical_pd["q_perm"] < 0.01),
    "q_beta_0.01" = sum(empirical_pd["q_beta"] < 0.01)
  )

  empirical_pd %>% write_delim(out_tsv, "\t")
  summary %>% write_delim(out_summary, "\t")
}

args <- commandArgs(TRUE)
if (length(args) < 1) {
  stop("Usage: TensorQTL.R <nominal_qvalues|trans_qvalues|regional_postprocess> [args...]", call. = FALSE)
}

mode <- args[1]
if (mode == "nominal_qvalues") {
  if (length(args) < 2 || length(args) > 3) {
    stop("Usage: TensorQTL.R nominal_qvalues <pairs.tsv> [interaction]", call. = FALSE)
  }
  interaction <- if (length(args) == 3) args[3] else ""
  apply_nominal_qvalues(args[2], interaction)
} else if (mode == "trans_qvalues") {
  if (length(args) != 2) {
    stop("Usage: TensorQTL.R trans_qvalues <pairs.tsv>", call. = FALSE)
  }
  apply_trans_qvalues(args[2])
} else if (mode == "regional_postprocess") {
  if (length(args) < 4) {
    stop("Usage: TensorQTL.R regional_postprocess <out.tsv.gz> <summary.txt> <regional.tsv.gz> [...]", call. = FALSE)
  }
  run_regional_postprocess(args[2], args[3], args[-c(1, 2, 3)])
} else {
  stop(sprintf("Unknown TensorQTL.R mode: %s", mode), call. = FALSE)
}
