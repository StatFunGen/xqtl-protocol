#!/usr/bin/env Rscript
# ============================================================
# GWAS_QC.R
# Mirrors: pipeline/GWAS_QC.ipynb R-specific steps
#
# Steps (selected via --step):
#   king_2  — kinship-based sample removal (maximise unrelated set)
#
# Usage:
#   Rscript GWAS_QC.R --step king_2 \
#       --input  name.kin0 \
#       --output name.related_id \
#       --kinship 0.0625
# ============================================================

suppressPackageStartupMessages(library(optparse))

opt_list <- list(
  make_option("--step",    type = "character", default = NULL,
              help = "Step to run: king_2"),
  make_option("--input",   type = "character", default = NULL,
              help = "KING relatedness output file (.kin0)"),
  make_option("--output",  type = "character", default = NULL,
              help = "Output file for related sample IDs"),
  make_option("--kinship", type = "double",    default = 0.0625,
              help = "Kinship coefficient threshold for relatedness (default: 0.0625 = 3rd degree)")
)

opt <- parse_args(OptionParser(option_list = opt_list))
if (is.null(opt$step))   stop("--step is required")
if (is.null(opt$input))  stop("--input is required")
if (is.null(opt$output)) stop("--output is required")

# ── filter_relatedness ───────────────────────────────────────────────────────
# By Rui Dong and Derek Lamb, Columbia Neurology
# This function enhances the standard plinkQC::relatednessFilter approach:
#
# 1. Graph-based preprocessing:
#    - For large relatedness networks, breaks down dense clusters of related
#      individuals by removing highly connected nodes first
#    - Prevents memory issues when processing large components (>20 individuals)
#    - Reduces input size for the more intensive plinkQC filtering
#
# 2. Additional checks:
#    - Verifies all related pairs are handled
#    - Runs up to 20 iterations to ensure complete filtering
#    - Catches edge cases missed in a single pass

filter_relatedness <- function(
  relatedness,
  relatednessTh         = 0.0625,
  analysis_type         = "maximize_unrelated",
  output_prefix         = "output_prefix",
  otherCriterion        = NULL,
  otherCriterionTh      = NULL,
  otherCriterionThDirection = "ge",
  relatednessIID1       = "IID1",
  relatednessIID2       = "IID2",
  relatednessFID1       = NULL,
  relatednessFID2       = NULL,
  relatednessRelatedness = "PI_HAT",
  pheno_file            = NULL,
  pheno_col             = "pheno",
  otherCriterionIID     = "IID",
  otherCriterionMeasure = NULL,
  verbose               = FALSE
) {
  suppressMessages({
    library(tidyverse)
    library(igraph)
    library(plinkQC)
    library(data.table)
  })

  if (is.null(relatedness)) stop("Relatedness file path is required")

  if (analysis_type == "maximize_cases" && is.null(pheno_file))
    stop("Must provide phenotype file when analysis_type is 'maximize_cases'")

  full_kin0 <- as.data.frame(relatedness)

  # ── Graph-based preprocessing: reduce large components ─────────────────────
  working_graph <- full_kin0 |>
    filter(!!sym(relatednessRelatedness) >= relatednessTh) |>
    select(!!sym(relatednessIID1), !!sym(relatednessIID2)) |>
    graph_from_data_frame(directed = FALSE)

  working_comp <- components(working_graph)

  graph_size_th   <- 20
  reduce_fraction <- 0.05
  high_related_indiv <- c()

  while (max(working_comp$csize > graph_size_th)) {
    if (verbose)
      print(paste0("Largest component has ", max(working_comp$csize),
                   " individuals. Removing top ", scales::percent(reduce_fraction), "."))

    large_comp_ids <- which(working_comp$csize > graph_size_th)
    nodes_to_remove <- c()

    for (comp_id in large_comp_ids) {
      comp_nodes  <- V(working_graph)[working_comp$membership == comp_id]
      comp_degrees <- degree(working_graph, v = comp_nodes)
      num_to_remove <- ceiling(length(comp_nodes) * reduce_fraction)
      high_degree_nodes <- names(sort(comp_degrees, decreasing = TRUE))[1:num_to_remove]
      nodes_to_remove <- c(nodes_to_remove, high_degree_nodes)
    }

    high_related_indiv <- c(high_related_indiv, nodes_to_remove)
    working_graph <- delete_vertices(working_graph, nodes_to_remove)
    working_comp  <- components(working_graph)
  }

  kin0 <- full_kin0 |>
    filter(!(!!sym(relatednessIID1) %in% high_related_indiv) &
           !(!!sym(relatednessIID2) %in% high_related_indiv))

  # ── Main filtering ─────────────────────────────────────────────────────────
  if (analysis_type == "maximize_unrelated") {
    rel <- plinkQC::relatednessFilter(
      relatedness               = kin0,
      otherCriterion            = otherCriterion,
      relatednessTh             = relatednessTh,
      relatednessIID1           = relatednessIID1,
      relatednessIID2           = relatednessIID2,
      otherCriterionTh          = otherCriterionTh,
      otherCriterionThDirection = otherCriterionThDirection,
      relatednessFID1           = relatednessFID1,
      relatednessFID2           = relatednessFID2,
      relatednessRelatedness    = relatednessRelatedness,
      otherCriterionIID         = otherCriterionIID,
      otherCriterionMeasure     = otherCriterionMeasure,
      verbose                   = verbose
    )$failIDs
    all_exclude <- rel$IID

  } else if (analysis_type == "maximize_cases") {
    related_individuals <- unique(c(kin0 |> pull(relatednessIID1),
                                    kin0 |> pull(relatednessIID2)))

    df_pheno <- fread(pheno_file) |>
      drop_na(!!sym(pheno_col)) |>
      filter(IID %in% related_individuals)

    related_pheno_individuals <- df_pheno |> pull(IID)
    related_cases    <- df_pheno |> filter(!!sym(pheno_col) == 1) |> pull(IID)
    related_controls <- df_pheno |> filter(!!sym(pheno_col) == 0) |> pull(IID)

    kin0 <- kin0 |>
      filter(!!sym(relatednessIID1) %in% related_pheno_individuals &
             !!sym(relatednessIID2) %in% related_pheno_individuals)

    df_case_kin <- kin0 |>
      filter(!!sym(relatednessIID1) %in% related_cases &
             !!sym(relatednessIID2) %in% related_cases)

    rel_cases <- plinkQC::relatednessFilter(
      relatedness               = df_case_kin,
      otherCriterion            = otherCriterion,
      relatednessTh             = relatednessTh,
      relatednessIID1           = relatednessIID1,
      relatednessIID2           = relatednessIID2,
      otherCriterionTh          = otherCriterionTh,
      otherCriterionThDirection = otherCriterionThDirection,
      relatednessFID1           = relatednessFID1,
      relatednessFID2           = relatednessFID2,
      relatednessRelatedness    = relatednessRelatedness,
      otherCriterionIID         = otherCriterionIID,
      otherCriterionMeasure     = otherCriterionMeasure,
      verbose                   = verbose
    )$failIDs

    related_cases_keep       <- setdiff(related_cases, rel_cases$IID)
    related_controls_exclude <- c()

    for (i in seq_len(nrow(kin0))) {
      iid1 <- kin0 |> pull(!!sym(relatednessIID1)) |> nth(i)
      iid2 <- kin0 |> pull(!!sym(relatednessIID2)) |> nth(i)
      if (iid1 %in% related_cases_keep & iid2 %in% related_controls)
        related_controls_exclude <- c(related_controls_exclude, iid2)
      else if (iid2 %in% related_cases_keep & iid1 %in% related_controls)
        related_controls_exclude <- c(related_controls_exclude, iid1)
    }

    related_controls_keep <- setdiff(related_controls, related_controls_exclude)

    df_control_kin <- kin0 |>
      filter(!!sym(relatednessIID1) %in% related_controls_keep &
             !!sym(relatednessIID2) %in% related_controls_keep)

    rel_controls <- plinkQC::relatednessFilter(
      relatedness               = df_control_kin,
      otherCriterion            = otherCriterion,
      relatednessTh             = relatednessTh,
      relatednessIID1           = relatednessIID1,
      relatednessIID2           = relatednessIID2,
      otherCriterionTh          = otherCriterionTh,
      otherCriterionThDirection = otherCriterionThDirection,
      relatednessFID1           = relatednessFID1,
      relatednessFID2           = relatednessFID2,
      relatednessRelatedness    = relatednessRelatedness,
      otherCriterionIID         = otherCriterionIID,
      otherCriterionMeasure     = otherCriterionMeasure,
      verbose                   = verbose
    )$failIDs

    all_exclude <- c(rel_cases$IID, related_controls_exclude, rel_controls$IID)

  } else {
    rel <- kin0 |> dplyr::filter(!!sym(relatednessRelatedness) >= relatednessTh)
    all_exclude <- unique(c(rel |> pull(relatednessIID1), rel |> pull(relatednessIID2)))
  }

  # ── Iterative check ────────────────────────────────────────────────────────
  df_related <- kin0 |>
    filter(!(!!sym(relatednessIID1) %in% all_exclude) &
           !(!!sym(relatednessIID2) %in% all_exclude)) |>
    filter(!!sym(relatednessRelatedness) > relatednessTh)

  if (nrow(df_related) > 0) {
    if (verbose)
      print(paste0("Warning: ", nrow(df_related), " related pairs remain after first pass."))

    iter <- 0
    while (nrow(df_related) > 0 & iter < 20) {
      additional_exclude <- plinkQC::relatednessFilter(
        relatedness               = df_related,
        otherCriterion            = otherCriterion,
        relatednessTh             = relatednessTh,
        relatednessIID1           = relatednessIID1,
        relatednessIID2           = relatednessIID2,
        otherCriterionTh          = otherCriterionTh,
        otherCriterionThDirection = otherCriterionThDirection,
        relatednessFID1           = relatednessFID1,
        relatednessFID2           = relatednessFID2,
        relatednessRelatedness    = relatednessRelatedness,
        otherCriterionIID         = otherCriterionIID,
        otherCriterionMeasure     = otherCriterionMeasure,
        verbose                   = verbose
      )$failIDs

      all_exclude <- c(all_exclude, additional_exclude$IID)
      df_related  <- kin0 |>
        filter(!(!!sym(relatednessIID1) %in% all_exclude) &
               !(!!sym(relatednessIID2) %in% all_exclude)) |>
        filter(!!sym(relatednessRelatedness) > relatednessTh)
      iter <- iter + 1
    }

    if (verbose) {
      if (nrow(df_related) == 0)
        print(paste0("All related subjects removed after ", iter, " iterations."))
      else
        stop(paste0("After 20 plinkQC passes, ", nrow(df_related), " related pairs remain."))
    }
  }

  all_exclude <- c(all_exclude, high_related_indiv)
  dat <- data.frame(IID = all_exclude, FID = as.character(all_exclude))

  output_related <- paste0(output_prefix, "_kinship_cutoff_", relatednessTh, ".related_id")
  write.table(dat, output_related, quote = FALSE, row.names = FALSE, col.names = FALSE)

  if (verbose) {
    cat("\n[INFO]", nrow(dat), "excluded individuals (kinship threshold =", relatednessTh, ")\n")
    cat("[OUTPUT]", normalizePath(output_related), "\n")
  }

  return(dat)
}


# ── Step: king_2 ─────────────────────────────────────────────────────────────
if (opt$step == "king_2") {
  suppressMessages({
    library(tidyverse)
    library(data.table)
  })

  kin0 <- read.table(opt$input, header = FALSE, stringsAsFactors = FALSE)
  colnames(kin0) <- c("FID1", "ID1", "FID2", "ID2", "NSNP", "HETHET", "IBS0", "KINSHIP")

  result <- filter_relatedness(
    relatedness            = kin0,
    relatednessTh          = opt$kinship,
    analysis_type          = "maximize_unrelated",
    relatednessIID1        = "ID1",
    relatednessIID2        = "ID2",
    relatednessFID1        = "FID1",
    relatednessFID2        = "FID2",
    relatednessRelatedness = "KINSHIP",
    verbose                = FALSE
  )

  tmp1 <- kin0[, 1:2]
  tmp2 <- kin0[, 3:4]
  colnames(tmp1) <- colnames(tmp2) <- c("FID", "ID")
  lookup <- dplyr::distinct(rbind(tmp1, tmp2))
  dat    <- lookup[which(lookup[, 2] %in% result$IID), ]

  cat("There are", nrow(dat), "related individuals using a kinship threshold of",
      opt$kinship, "\n")
  write.table(dat, opt$output, quote = FALSE, row.names = FALSE, col.names = FALSE)

} else {
  stop(paste0("Unknown step: '", opt$step, "'. Valid steps: king_2"))
}
