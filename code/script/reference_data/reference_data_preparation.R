#!/usr/bin/env Rscript
# ============================================================
# reference_data_preparation.R
# Standalone CLI worker for reference_data_preparation.ipynb steps.
#
# Steps (selected via --step):
#   hg_reference_1 — drop ALT / HLA / decoy contigs from a genome FASTA (Biostrings)
#   faidx          — index a FASTA (Rsamtools; replaces `samtools faidx`)
#   hg_gtf_1       — match GTF chr naming to the FASTA + transcript_biotype->transcript_type
#   ercc_gtf       — expand the ERCC exon GTF into gene/transcript/exon records
#   suppa_annot    — psichomics annotation RDS from a directory of SUPPA .ioe files
#   tad_annotate   — annotate regions with the TAD they fall inside + genotype-file list
#
# Conventions: read with vroom, sequences via Biostrings, faidx via Rsamtools.
# CLI flags match the SoS notebook parameter names.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
})

parser <- arg_parser("reference_data_preparation worker (see --step)")
parser <- add_argument(parser, "--step", type = "character", help = "hg_reference_1 | faidx | hg_gtf_1 | ercc_gtf")
parser <- add_argument(parser, "--input", type = "character", default = "", help = "input file")
parser <- add_argument(parser, "--output", type = "character", default = "", help = "output file")
parser <- add_argument(parser, "--hg-reference", type = "character", default = "",
                       help = "[hg_gtf_1] genome FASTA (to detect chr naming)")
parser <- add_argument(parser, "--hg-gtf", type = "character", default = "", help = "[hg_gtf_1] GTF to reformat")
parser <- add_argument(parser, "--ioe-dir", type = "character", default = "",
                       help = "[suppa_annot] directory of SUPPA .ioe files")
parser <- add_argument(parser, "--genome", type = "character", default = "hg38",
                       help = "[suppa_annot] genome tag for parseSuppaAnnotation")
parser <- add_argument(parser, "--tad-list", type = "character", default = "",
                       help = "[tad_annotate] TAD table (#chr,start,end,tad_index)")
parser <- add_argument(parser, "--region-list", type = "character", default = "",
                       help = "[tad_annotate] region table (#chr,start,end,ID,...)")
parser <- add_argument(parser, "--tad-genotype", type = "character", default = "",
                       help = "[tad_annotate] TAD genotype-file list (keyed by #id == tad_index)")
argv <- parse_args(parser)
if (is.na(argv$step)) stop("--step is required")

# ---------------------------------------------------------------------------
# hg_reference_1: exclude _alt / HLA* / _decoy contigs
# ---------------------------------------------------------------------------
hg_reference_1 <- function(argv) {
  suppressPackageStartupMessages(library(Biostrings))
  fa  <- readDNAStringSet(argv$input)
  ids <- sub(" .*", "", names(fa))                      # first token of each header
  keep <- !(grepl("_alt$", ids) | grepl("^HLA", ids) | grepl("_decoy$", ids))
  writeXStringSet(fa[keep], argv$output)
  message(sprintf("Kept %d of %d contigs -> %s", sum(keep), length(fa), argv$output))
}

# ---------------------------------------------------------------------------
# faidx: samtools faidx -> Rsamtools::indexFa (writes <input>.fai)
# ---------------------------------------------------------------------------
faidx <- function(argv) {
  suppressPackageStartupMessages(library(Rsamtools))
  Rsamtools::indexFa(argv$input)
  message(sprintf("Indexed: %s.fai", argv$input))
}

# ---------------------------------------------------------------------------
# hg_gtf_1: make GTF chr naming match the FASTA; transcript_biotype -> transcript_type
# ---------------------------------------------------------------------------
hg_gtf_1 <- function(argv) {
  suppressPackageStartupMessages({ library(stringr); library(dplyr) })
  fasta_first <- readLines(argv$hg_reference, n = 1)
  gtf <- vroom::vroom(argv$hg_gtf, delim = "\t", col_names = FALSE, comment = "#",
                      col_types = cols(.default = "c"))
  names(gtf) <- paste0("X", seq_len(ncol(gtf)))
  if (!str_detect(fasta_first, ">chr")) {
    gtf <- gtf %>% mutate(X1 = str_remove_all(X1, "chr"))
  } else if (!any(str_detect(gtf$X1[1], "chr"))) {
    gtf <- gtf %>% mutate(X1 = paste0("chr", X1))
  }
  if (any(str_detect(gtf$X9, "transcript_biotype"))) {
    gtf <- gtf %>% mutate(X9 = str_replace_all(X9, "transcript_biotype", "transcript_type"))
  }
  vroom::vroom_write(gtf, argv$output, delim = "\t", col_names = FALSE, quote = "none", escape = "none")
  message(sprintf("Written: %s", argv$output))
}

# ---------------------------------------------------------------------------
# ercc_gtf: one ERCC exon row -> gene + transcript + exon rows (GTEx convention)
# ---------------------------------------------------------------------------
ercc_gtf <- function(argv) {
  g <- vroom::vroom(argv$input, delim = "\t", col_names = FALSE, comment = "#",
                    col_types = cols(.default = "c"))
  names(g) <- paste0("X", seq_len(ncol(g)))
  g$X1 <- gsub("-", "_", g$X1)                          # no '-' in contig names (RNA-SeQC/GATK)
  gid  <- sub('.*gene_id "?([^";]+)"?.*', "\\1", g$X9)
  full <- sprintf(paste0('gene_id "%s"; transcript_id "%s"; gene_type "ercc_control"; ',
                         'gene_status "KNOWN"; gene_name "%s"; transcript_type "ercc_control"; ',
                         'transcript_status "KNOWN"; transcript_name "%s"; level 2;'),
                  gid, gid, gid, gid)
  exon <- sprintf('gene_id "%s"; transcript_id "%s";', gid, gid)
  n <- nrow(g)
  out <- data.frame(
    X1 = rep(g$X1, each = 3), X2 = rep(g$X2, each = 3),
    X3 = rep(c("gene", "transcript", "exon"), n),
    X4 = rep(g$X4, each = 3), X5 = rep(g$X5, each = 3), X6 = rep(g$X6, each = 3),
    X7 = rep(g$X7, each = 3), X8 = rep(g$X8, each = 3),
    X9 = as.vector(rbind(full, full, exon)),
    stringsAsFactors = FALSE)
  vroom::vroom_write(out, argv$output, delim = "\t", col_names = FALSE, quote = "none", escape = "none")
  message(sprintf("Written: %s (%d rows)", argv$output, nrow(out)))
}

# ---------------------------------------------------------------------------
# suppa_annot: psichomics annotation object from SUPPA .ioe files ([SUPPA_annotation_2])
# ---------------------------------------------------------------------------
suppa_annot <- function(argv) {
  suppressPackageStartupMessages(library(psichomics))
  suppa <- parseSuppaAnnotation(argv$ioe_dir, genome = argv$genome)
  annot <- prepareAnnotationFromEvents(suppa)
  saveRDS(annot, argv$output)
  message(sprintf("Written: %s", argv$output))
}

# ---------------------------------------------------------------------------
# tad_annotate: regions inside each TAD + genotype-file list ([Tad_annotateion])
#   Faithful port of the pandas step: merge on '#chr' with pandas _x/_y suffixes
#   (TAD coords = _x, region coords = _y), keep regions strictly inside a TAD,
#   drop the region coords. The genotype-file list reproduces the original's
#   positional iloc[:, [4, 6]] on the (region <-> genotype) inner merge — for the
#   standard region schema those columns are the region ID and its file path.
#   (The original pandas wrote a meaningless row-index column; we omit it.)
# ---------------------------------------------------------------------------
tad_annotate <- function(argv) {
  suppressPackageStartupMessages(library(dplyr))
  tad    <- vroom::vroom(argv$tad_list,    delim = "\t", show_col_types = FALSE)
  region <- vroom::vroom(argv$region_list, delim = "\t", show_col_types = FALSE)
  annot <- inner_join(tad, region, by = "#chr", suffix = c("_x", "_y")) %>%
    filter(start_y > start_x & end_y < end_x) %>%
    select(-start_y, -end_y)
  vroom::vroom_write(annot, argv$output, delim = "\t", quote = "none", escape = "none")

  geno <- vroom::vroom(argv$tad_genotype, delim = "\t", show_col_types = FALSE)
  m2 <- inner_join(annot, geno, by = join_by(tad_index == `#id`), keep = TRUE)
  out2 <- m2[, c(5, 7)]                                  # pandas iloc[:, [4, 6]] (0-indexed)
  names(out2) <- c("ID", "dir")
  vroom::vroom_write(out2, paste0(argv$output, ".genotype_files_list.tsv"),
                     delim = "\t", quote = "none", escape = "none")
  message(sprintf("Written: %s (+ .genotype_files_list.tsv)", argv$output))
}

# ---------------------------------------------------------------------------
switch(argv$step,
  hg_reference_1 = hg_reference_1(argv),
  faidx          = faidx(argv),
  hg_gtf_1       = hg_gtf_1(argv),
  ercc_gtf       = ercc_gtf(argv),
  suppa_annot    = suppa_annot(argv),
  tad_annotate   = tad_annotate(argv),
  stop(sprintf("Unknown step: '%s'", argv$step))
)
