#!/usr/bin/env Rscript
# ============================================================
# gsea.R
# Standalone CLI worker for gsea.ipynb [pathway_analysis].
#
# Per gene group, run KEGG (enrichKEGG) and GO BP/CC/MF (enrichGO) enrichment
# via clusterProfiler, standardize the columns, and save a combined RDS.
#
# Conventions: read with vroom; enrichment via clusterProfiler + org.Hs.eg.db.
# CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
})

parser <- arg_parser("Pathway enrichment (KEGG + GO) per gene group")
parser <- add_argument(parser, "--genes-file", type = "character",
                       help = "TSV with columns: group, gene_id (ENSEMBL)")
parser <- add_argument(parser, "--organism", type = "character", default = "hsa",
                       help = "KEGG organism code")
parser <- add_argument(parser, "--pvalue-cutoff", type = "numeric", default = 1,
                       help = "Enrichment p-value cutoff")
parser <- add_argument(parser, "--output", type = "character",
                       help = "Output combined-results RDS")
argv <- parse_args(parser)
if (is.na(argv$genes_file) || is.na(argv$output))
  stop("--genes-file and --output are required")

suppressPackageStartupMessages({
  library(org.Hs.eg.db)
  library(AnnotationDbi)
  library(clusterProfiler)
})

organism      <- argv$organism
pvalue_cutoff <- argv$pvalue_cutoff
gene_data <- vroom::vroom(argv$genes_file, delim = "\t",
                          col_types = cols(.default = "c"), show_col_types = FALSE)

# ENSEMBL -> ENTREZID
convert_ids <- function(gene_list) {
  entrez <- AnnotationDbi::mapIds(org.Hs.eg.db, keys = gene_list,
                                  column = "ENTREZID", keytype = "ENSEMBL")
  na.omit(unique(entrez))
}

standardize_columns <- function(result_df, analysis_type, ont_category, group_name) {
  if (is.null(result_df) || nrow(result_df) == 0) return(data.frame())
  result_df$group         <- group_name
  result_df$analysis_type <- analysis_type
  result_df$ont_category  <- ont_category
  if (analysis_type == "KEGG") {
    result_df$category    <- NA
    result_df$subcategory <- NA
  } else {
    result_df$category    <- "Gene Ontology"
    result_df$subcategory <- switch(ont_category,
      BP = "Biological Process", CC = "Cellular Component",
      MF = "Molecular Function", NA)
  }
  result_df
}

perform_kegg <- function(group_genes, group_name) {
  entrez <- convert_ids(group_genes)
  if (length(entrez) == 0) return(data.frame())
  enriched <- tryCatch(
    enrichKEGG(gene = entrez, organism = organism, pvalueCutoff = pvalue_cutoff),
    error = function(e) NULL)
  if (is.null(enriched)) return(data.frame())
  standardize_columns(as.data.frame(enriched), "KEGG", "PATHWAY", group_name)
}

perform_go <- function(group_genes, group_name, ont_type) {
  entrez <- convert_ids(group_genes)
  if (length(entrez) == 0) return(data.frame())
  enriched <- tryCatch(
    enrichGO(gene = entrez, OrgDb = org.Hs.eg.db, ont = ont_type,
             pvalueCutoff = pvalue_cutoff, readable = TRUE),
    error = function(e) NULL)
  if (is.null(enriched)) return(data.frame())
  standardize_columns(as.data.frame(enriched), "GO", ont_type, group_name)
}

all_results <- list()
for (group_name in unique(gene_data$group)) {
  group_genes <- gene_data$gene_id[gene_data$group == group_name]
  kegg <- perform_kegg(group_genes, group_name)
  if (nrow(kegg) > 0) all_results[[paste0(group_name, "_KEGG")]] <- kegg
  for (ont_type in c("BP", "CC", "MF")) {
    go <- perform_go(group_genes, group_name, ont_type)
    if (nrow(go) > 0) all_results[[paste0(group_name, "_GO_", ont_type)]] <- go
  }
}

if (length(all_results) > 0) {
  combined_results <- do.call(rbind, all_results)
  rownames(combined_results) <- NULL
} else {
  combined_results <- data.frame(
    category = character(), subcategory = character(), ID = character(),
    Description = character(), GeneRatio = character(), BgRatio = character(),
    RichFactor = numeric(), FoldEnrichment = numeric(), zScore = numeric(),
    pvalue = numeric(), p.adjust = numeric(), qvalue = numeric(),
    geneID = character(), Count = integer(), group = character(),
    analysis_type = character(), ont_category = character(),
    stringsAsFactors = FALSE)
}

dir.create(dirname(argv$output), showWarnings = FALSE, recursive = TRUE)
saveRDS(combined_results, argv$output, compress = "xz")
message(sprintf("Written: %s (%d rows)", argv$output, nrow(combined_results)))
