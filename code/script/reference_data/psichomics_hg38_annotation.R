#!/usr/bin/env Rscript
# ============================================================
# psichomics_hg38_annotation.R
# Standalone CLI worker for reference_data_preparation.ipynb [psi_hg38_annotation].
#
# Reconcile the gene-symbol column of the psichomics hg38 splicing annotation
# (loadAnnotation, AnnotationHub AH63657) to Ensembl gene IDs, drawing on four
# sources in priority order: the collapsed gene-model GTF (name<->id), the HGNC
# database (approved / alias / previous symbols + chromosome), and the Ensembl
# substrings embedded in the VAST-TOOLS and SUPPA event IDs. `Gene` is rewritten
# to the resolved Ensembl ID and unresolved rows are dropped; the modified
# annotation list is saved as RDS.
#
# Faithful port of the original notebook R (same tidyr/dplyr/purrr pipeline). The
# only change is the HARD RULE swap data.table::fread -> vroom for the HGNC table.
#
# NOTE: not exercised by the test suite — loadAnnotation("AH63657") requires an
# AnnotationHub network download and psichomics, and the HGNC database file
# (--hgnc-db, the notebook's undocumented `hgrc_db`) is not part of the MWE.
# ============================================================

suppressPackageStartupMessages(library(argparser))

parser <- arg_parser("psichomics hg38 annotation: reconcile gene symbols to Ensembl IDs")
parser <- add_argument(parser, "--hg-gtf", type = "character", help = "collapsed gene-model GTF (name<->id source)")
parser <- add_argument(parser, "--hgnc-db", type = "character", help = "HGNC database TSV (approved/alias/previous symbols)")
parser <- add_argument(parser, "--output", type = "character", help = "output RDS")
parser <- add_argument(parser, "--annotation-id", type = "character", default = "AH63657",
                       help = "psichomics AnnotationHub id (see listSplicingAnnotations())")
argv <- parse_args(parser)
if (is.na(argv$hg_gtf) || is.na(argv$hgnc_db) || is.na(argv$output))
  stop("--hg-gtf, --hgnc-db and --output are required")

suppressPackageStartupMessages({
  library(psichomics)
  library(purrr)
  library(tidyr)
  library(vroom)
  library(dplyr)
})

# load psichomics default annotation (hg38 = AH63657 from listSplicingAnnotations())
annotation <- loadAnnotation(argv$annotation_id)

# reduce the dimension of the annotation (Gene is a list-column)
annotation <- map(annotation, ~ .x %>% tidyr::unnest(cols = `Gene`))

# create empty event-id columns where a given event type lacks them (for uniform mapping)
annotation[["Tandem UTR"]][["SUPPA.Event.ID"]] <- NA
annotation[["Tandem UTR"]][["VAST-TOOLS.Event.ID"]] <- NA
annotation[["Alternative first exon"]][["VAST-TOOLS.Event.ID"]] <- NA
annotation[["Alternative last exon"]][["VAST-TOOLS.Event.ID"]] <- NA
annotation[["Mutually exclusive exon"]][["VAST-TOOLS.Event.ID"]] <- NA

# extract Ensembl ID substrings from the SUPPA / VAST-TOOLS event IDs
annotation <- map(annotation, ~ .x %>%
  mutate(ENSG.SUPPA = substr(`SUPPA.Event.ID`, 1, 15)) %>%
  mutate(ENSG.VAST  = substr(`VAST-TOOLS.Event.ID`, 1, 15)))

# GTF gene_name <-> gene_id map
gtf_sample <- read.table(argv$hg_gtf, header = FALSE, sep = "\t")
gtf_sample <- separate(gtf_sample, V9, sep = ";",  into = c("gene_id", "transcript_id", "exon_number", "gene_name"))
gtf_sample <- separate(gtf_sample, gene_id,   sep = " ",  into = c("gene_id", "gene_id_val"))
gtf_sample <- separate(gtf_sample, gene_name, sep = "e ", into = c("gene_name", "gene_name_val"))
gtf_name_id_match <- gtf_sample[, c("gene_id_val", "gene_name_val")]
gtf_name_id_match <- gtf_name_id_match[!duplicated(gtf_name_id_match), ]

annotation <- map(annotation, ~ .x %>%
  mutate(`ENSG.GTF` = gtf_name_id_match$gene_id_val[match(`Gene`, gtf_name_id_match$gene_name_val)]))

# HGNC database (vroom replaces data.table::fread; quote="" disables quoting, as fread quote="")
hgnc_db <- vroom::vroom(argv$hgnc_db, delim = "\t", quote = "", show_col_types = FALSE)

# combine the two Ensembl-ID columns, preferring `Ensembl ID(supplied by Ensembl)`
hgnc_db <- hgnc_db %>%
  mutate(ENSG.ID = ifelse(`Ensembl ID(supplied by Ensembl)` == "",
                          `Ensembl gene ID`, `Ensembl ID(supplied by Ensembl)`))

# one-to-one references for approved / previous / alias symbols
hgnc_name_id_match    <- hgnc_db[, c("Approved symbol", "ENSG.ID")]
hgnc_name_prev_check  <- hgnc_db[, c("Previous symbols", "Chromosome", "ENSG.ID")]
hgnc_name_alias_check <- hgnc_db[, c("Alias symbols", "Chromosome", "ENSG.ID")]

hgnc_name_prev_check  <- hgnc_name_prev_check[hgnc_name_prev_check$ENSG.ID != "", ]
hgnc_name_alias_check <- hgnc_name_alias_check[hgnc_name_alias_check$ENSG.ID != "", ]
hgnc_name_prev_check  <- hgnc_name_prev_check[hgnc_name_prev_check$"Previous symbols" != "", ]
hgnc_name_alias_check <- hgnc_name_alias_check[hgnc_name_alias_check$"Alias symbols" != "", ]

# explode comma/space-separated symbol lists to one symbol per row
hgnc_name_prev_check  <- separate_rows(hgnc_name_prev_check, "Previous symbols", convert = FALSE)
hgnc_name_alias_check <- separate_rows(hgnc_name_alias_check, "Alias symbols", convert = FALSE)

# reduce chromosome band (e.g. "1p36.3") to the chromosome number for cross-db matching
add_chr <- function(df) {
  df <- separate(df, "Chromosome", sep = "p", into = "Chrp", remove = FALSE)
  df <- separate(df, "Chromosome", sep = "q", into = "Chrq", remove = FALSE)
  df %>% mutate(Chr = ifelse(nchar(Chrp) <= 2, Chrp, Chrq))
}
hgnc_name_prev_check  <- add_chr(hgnc_name_prev_check)
hgnc_name_alias_check <- add_chr(hgnc_name_alias_check)

# match approved symbol -> Ensembl ID
annotation <- map(annotation, ~ .x %>%
  mutate(`ENSG.HGNC` = hgnc_name_id_match$`ENSG.ID`[match(`Gene`, hgnc_name_id_match$"Approved symbol")]))

# drop hypothetical genes
annotation <- map(annotation, ~ .x %>% subset(`Gene` != "Hypothetical"))

# fill remaining NAs from alias, then previous symbols (with chromosome)
annotation <- map(annotation, ~ .x %>%
  mutate(ENSG.HGNC = ifelse(is.na(`ENSG.HGNC`) | `ENSG.HGNC` == "",
                            hgnc_name_alias_check$`ENSG.ID`[match(`Gene`, hgnc_name_alias_check$"Alias symbols") & match(`Chromosome`, hgnc_name_alias_check$Chr)],
                            `ENSG.HGNC`)) %>%
  mutate(ENSG.HGNC = ifelse(is.na(`ENSG.HGNC`) | `ENSG.HGNC` == "",
                            hgnc_name_prev_check$`ENSG.ID`[match(`Gene`, hgnc_name_prev_check$"Previous symbols") & match(`Chromosome`, hgnc_name_prev_check$Chr)],
                            `ENSG.HGNC`)))

# final Ensembl ID: GTF -> HGNC -> VAST -> SUPPA; keep only ENSG*; else LOC* gene names
annotation <- map(annotation, ~ .x %>%
  mutate(`ENSG.ID` = `ENSG.GTF`) %>%
  mutate(`ENSG.ID` = ifelse(is.na(`ENSG.ID`), `ENSG.HGNC`, `ENSG.ID`)) %>%
  mutate(`ENSG.ID` = ifelse(is.na(`ENSG.ID`), `ENSG.VAST`, `ENSG.ID`)) %>%
  mutate(`ENSG.ID` = ifelse(is.na(`ENSG.ID`), `ENSG.SUPPA`, `ENSG.ID`)) %>%
  mutate(`ENSG.ID` = ifelse(substr(`ENSG.ID`, 1, 4) == "ENSG", `ENSG.ID`, NA)) %>%
  mutate(`ENSG.ID` = ifelse(is.na(`ENSG.ID`) & substr(`Gene`, 1, 3) == "LOC", `Gene`, `ENSG.ID`)))

# rewrite Gene to the Ensembl ID and drop unresolved rows
annotation <- map(annotation, ~ .x %>% mutate(`Gene` = `ENSG.ID`) %>% drop_na(`Gene`))

saveRDS(annotation, file = argv$output)
message(sprintf("Written: %s", argv$output))
