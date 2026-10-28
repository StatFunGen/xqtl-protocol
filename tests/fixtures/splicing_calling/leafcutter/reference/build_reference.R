#!/usr/bin/env Rscript
# ============================================================
# build_reference.R
# Build the chr22 GTF + FASTA pair that test_cluster_and_classify feeds to
# leafcutter2_cluster. LeafCutter2 classifies only junctions inside an annotated
# gene with CDS / start_codon / stop_codon records, and translates those genes from
# the FASTA, so the pair must cover the loci the junction fixtures cluster into.
#
# At --minclureads 3 the SAMPLE_00{1,2}.junc.gz fixtures keep three clusters:
#   chr22:12314919-12351352   intergenic
#   chr22:16451635-16770384   intergenic
#   chr22:28168701-28214161   inside TTC28 (ENSG00000100154, protein coding)
#
# Outputs (in --outdir):
#   Homo_sapiens.GRCh38.103.chr22.leafcutter2.gtf  every Ensembl 103 record of the
#       genes overlapping those clusters, Ensembl naming. Run hg_gtf_1 on it to get
#       the committed *.reformatted.gtf (see README.md).
#   chr22.leafcutter2.fa.gz (+ .fai, .gzi)  full-length chr22, N everywhere except
#       the kept genes and the clusters (each padded by PAD bp), bgzipped.
# ============================================================

suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(dplyr)
  library(Biostrings)
  library(Rsamtools)
})

parser <- arg_parser("Build the LeafCutter2 chr22 reference fixture")
parser <- add_argument(parser, "--ensembl-gtf", type = "character",
                       help = "Ensembl release 103 Homo_sapiens.GRCh38.103.chr.gtf.gz")
parser <- add_argument(parser, "--fasta", type = "character",
                       help = "protocol FASTA (GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta, indexed)")
parser <- add_argument(parser, "--outdir", type = "character", default = ".", help = "output directory")
argv <- parse_args(parser)

CLUSTERS <- data.frame(start = c(12314919L, 16451635L, 28168701L),
                       end   = c(12351352L, 16770384L, 28214161L))
PAD <- 1000L

gtf <- vroom::vroom(argv$ensembl_gtf, delim = "\t", col_names = FALSE, comment = "#",
                    col_types = cols(.default = "c"), progress = FALSE) %>%
  filter(X1 == "22") %>%
  mutate(gene_id = sub('.*gene_id "([^"]+)".*', "\\1", X9))
genes <- gtf %>%
  filter(X3 == "gene") %>%
  mutate(start = as.integer(X4), end = as.integer(X5))
hit <- vapply(seq_len(nrow(genes)), function(i)
  any(genes$start[[i]] <= CLUSTERS$end & genes$end[[i]] >= CLUSTERS$start), TRUE)
genes <- genes[hit, ]
message("genes kept: ", paste(genes$gene_id, collapse = ", "))
out_gtf <- gtf %>% filter(gene_id %in% genes$gene_id) %>% select(-gene_id)
vroom::vroom_write(out_gtf, file.path(argv$outdir, "Homo_sapiens.GRCh38.103.chr22.leafcutter2.gtf"),
                   delim = "\t", col_names = FALSE, quote = "none", escape = "none")

chr22 <- scanFa(FaFile(argv$fasta), param = GRanges("chr22", IRanges(1L, seqlengths(FaFile(argv$fasta))[["chr22"]])))[[1]]
keep <- reduce(IRanges(c(genes$start, CLUSTERS$start) - PAD, c(genes$end, CLUSTERS$end) + PAD))
masked <- DNAString(strrep("N", length(chr22)))
for (i in seq_along(keep)) subseq(masked, start(keep)[[i]], end(keep)[[i]]) <- subseq(chr22, start(keep)[[i]], end(keep)[[i]])
message("unmasked bp: ", sum(width(keep)), " of ", length(chr22))
fa <- file.path(argv$outdir, "chr22.leafcutter2.fa")
writeXStringSet(DNAStringSet(list(chr22 = masked)), fa, width = 60L)
invisible(bgzip(fa, paste0(fa, ".gz"), overwrite = TRUE))
unlink(fa)
invisible(indexFa(paste0(fa, ".gz")))
