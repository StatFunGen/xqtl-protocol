#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option("--step", type = "character", default = NULL),
  make_option("--phenoFile", type = "character", default = ""),
  make_option("--ensembl-version", type = "integer", default = NA_integer_),
  make_option("--output-bed", type = "character", default = ""),
  make_option("--output-region-list", type = "character", default = "")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$step) || !nzchar(opt$step)) {
  stop("--step is required")
}

run_command <- function(command, args = character()) {
  status <- system2(command, args = args)
  if (!identical(status, 0L)) {
    stop(sprintf("Command failed: %s %s", command, paste(args, collapse = " ")))
  }
}

annotate_coord_biomart <- function(opt) {
  suppressPackageStartupMessages({
    library(biomaRt)
    library(dplyr)
    library(readr)
  })

  required <- c("phenoFile", "output-bed", "output-region-list")
  missing_required <- required[vapply(required, function(name) !nzchar(opt[[name]]), logical(1))]
  if (length(missing_required) > 0) {
    stop(sprintf(
      "Missing required options for annotate_coord_biomart: %s",
      paste(missing_required, collapse = ", ")
    ))
  }
  if (is.na(opt[["ensembl-version"]])) {
    stop("--ensembl-version is required for annotate_coord_biomart")
  }

  dir.create(dirname(opt[["output-bed"]]), recursive = TRUE, showWarnings = FALSE)
  biomartCacheClear()

  gene_exp <- read_delim(opt[["phenoFile"]], delim = "\t", show_col_types = FALSE)
  if ("#chr" %in% colnames(gene_exp)) {
    gene_exp <- gene_exp[, 4:ncol(gene_exp)]
  }

  id_col <- if ("gene_ID" %in% colnames(gene_exp)) {
    "gene_ID"
  } else if ("gene_id" %in% colnames(gene_exp)) {
    "gene_id"
  } else {
    colnames(gene_exp)[1]
  }

  ensembl <- useEnsembl(
    biomart = "ensembl",
    dataset = "hsapiens_gene_ensembl",
    version = opt[["ensembl-version"]]
  )

  ensembl_df <- getBM(
    attributes = c("ensembl_gene_id", "chromosome_name", "start_position", "end_position"),
    mart = ensembl
  )

  my_genes_ann <- ensembl_df %>%
    filter(.data$ensembl_gene_id %in% gene_exp[[id_col]], .data$chromosome_name %in% as.character(1:23)) %>%
    rename(
      "#chr" = "chromosome_name",
      start = "start_position",
      end = "end_position",
      gene_ID = "ensembl_gene_id"
    ) %>%
    filter(.data$gene_ID != "NA")

  my_genes_ann %>%
    select(`#chr`, start, end, gene_ID) %>%
    write_delim(file = opt[["output-region-list"]], delim = "\t")

  colnames(gene_exp)[colnames(gene_exp) == id_col] <- "gene_ID"
  my_gene_bed <- my_genes_ann %>%
    mutate(end = start + 1) %>%
    inner_join(gene_exp, by = "gene_ID") %>%
    arrange(`#chr`, start)

  plain_output <- sub("\\.gz$", "", opt[["output-bed"]])
  write_tsv(my_gene_bed, file = plain_output, na = "NA")
  run_command("bgzip", c("-f", plain_output))
  run_command("tabix", c("-f", "-p", "bed", opt[["output-bed"]]))
}

if (opt$step == "annotate_coord_biomart") {
  annotate_coord_biomart(opt)
} else {
  stop(sprintf("Unknown step: %s", opt$step))
}
