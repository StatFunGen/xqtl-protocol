#!/usr/bin/env Rscript
# intact.R
#
# TWAS x COLOC integration via the Bioconductor INTACT package. Thin CLI
# wrapper for the [intact] step of intact.ipynb: it joins PTWAS output
# (per-gene TWAS z-scores) with fastenloc output (per-gene colocalization
# probabilities) on gene name, runs INTACT's empirical-Bayes integration
# (TWAS z -> Bayes factor, GLCP -> linear prior -> posterior) plus its FDR
# control, and writes the merged table.
#
# INTACT is the analysis engine (INTACT::intact / INTACT::fdr_rst); this
# wrapper only handles file I/O and the gene join, so it stays thin.
#
# Inputs:
#   --ptwas-file <tsv>      PTWAS output; TSV with (at least) columns GENE,
#                           STAT (TWAS z-score) and SUBCLASS. Duplicate
#                           (GENE, SUBCLASS) rows are collapsed.
#   --fastenloc-file <txt>  fastenloc output; whitespace table with columns
#                           Gene and GLCP (gene-level colocalization prob).
#   --alpha <num>           FDR significance threshold. Default 0.05.
#   --output <RDS>          Output RDS: the merged PTWAS + fastenloc table
#                           augmented with intact_pip (posterior) and fdr_sig
#                           (INTACT FDR flag), sorted by descending intact_pip.
#
# Genes are matched by an inner join on gene name (fastenloc "-" -> "."
# normalized); genes present in only one input are dropped.

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
  library(readr)
  library(stringr)
  library(INTACT)
})

parser <- arg_parser("INTACT integration of PTWAS TWAS z-scores and fastenloc colocalization probabilities")
parser <- add_argument(parser, "--ptwas-file",
                       help = "PTWAS output TSV (columns GENE, STAT, SUBCLASS)",
                       type = "character")
parser <- add_argument(parser, "--fastenloc-file",
                       help = "fastenloc output whitespace table (columns Gene, GLCP)",
                       type = "character")
parser <- add_argument(parser, "--alpha",
                       help = "FDR significance threshold",
                       type = "numeric", default = 0.05)
parser <- add_argument(parser, "--output",
                       help = "Output RDS path",
                       type = "character")
argv <- parse_args(parser)

run_intact <- function(ptwas, fastenloc, alpha) {
  # join the columns we want
  sub_ptwas <- ptwas %>% select(gene = GENE, zscore = STAT)
  # match the gene names
  sub_fastenloc <- fastenloc %>%
    select(gene = Gene, GLCP) %>%
    mutate(gene = str_replace_all(gene, "-", "\\."))
  ptwas_fastenloc <- sub_ptwas %>% inner_join(sub_fastenloc, by = "gene")

  res <- intact(GLCP_vec = ptwas_fastenloc$GLCP,
                z_vec = ptwas_fastenloc$zscore,
                prior_fun = linear)
  pip_fdr <- fdr_rst(res, alpha)

  # combine results, sorted by descending posterior
  cbind(ptwas_fastenloc, intact_pip = res, fdr_sig = pip_fdr[["sig"]]) %>%
    arrange(-intact_pip)
}

# PTWAS: tab-delimited; collapse duplicate (GENE, SUBCLASS) rows.
ptwas <- vroom(argv$ptwas_file, delim = "\t", show_col_types = FALSE) %>%
  distinct(GENE, SUBCLASS, .keep_all = TRUE)
# fastenloc: ragged-whitespace table (read_table handles the variable spacing);
# guard against out-of-range GLCP.
fastenloc <- read_table(argv$fastenloc_file, show_col_types = FALSE) %>%
  filter(GLCP <= 1)

res <- run_intact(ptwas, fastenloc, alpha = argv$alpha)
saveRDS(res, argv$output)
