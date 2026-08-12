suppressPackageStartupMessages({
  library(argparser)
  library(vroom)
  library(GenomicRanges)
  library(rtracklayer)
})

p <- arg_parser("Build the H3K9ac domain set from MACS2 peaks and flag ENCODE-blacklist overlaps")
p <- add_argument(p, "--peaks",     help = "MACS2 peaks .xls (e.g. positivePool_peaks.xls)")
p <- add_argument(p, "--blacklist", help = "ENCODE blacklist BED (.bed / .bed.gz)")
p <- add_argument(p, "--output",    help = "Output H3K9acDomains.csv")
argv <- parse_args(p)

# MACS2 .xls has leading '#' comment lines; vroom keeps the real column names.
peaks <- vroom(argv$peaks, comment = "#", show_col_types = FALSE)
domains <- GRanges(
  seqnames = peaks$chr,
  ranges   = IRanges(start = peaks$start, end = peaks$end),
  log10p         = peaks[["-log10(pvalue)"]],
  log10q         = peaks[["-log10(qvalue)"]],
  foldEnrichment = peaks$fold_enrichment,
  pileup         = peaks$pileup,
  name           = peaks$name)
domains <- sortSeqlevels(domains)
domains <- sort(domains)
domains$name <- paste("peak_", seq_along(domains), sep = "")  # rename peaks by sorted order

blacklist <- import(con = argv$blacklist, format = "bed")
seqlevelsStyle(blacklist) <- "Ensembl"
domains$blacklist <- overlapsAny(domains, blacklist)

domains <- as.data.frame(domains)
colnames(domains)[1] <- "chr"
vroom_write(domains, argv$output, delim = ",", quote = "none")
