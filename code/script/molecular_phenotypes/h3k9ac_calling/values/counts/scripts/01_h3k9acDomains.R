library(GenomicRanges)
library(rtracklayer)

domains <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/positivePool/positivePool_peaks.xls",
                      header=TRUE, stringsAsFactors=FALSE)
domains <- GRanges(IRanges(start=domains$start, end=domains$end),
                   seqnames=domains$chr,
                   log10p=domains$X.log10.pvalue.,
                   log10q=domains$X.log10.qvalue.,
                   foldEnrichment=domains$fold_enrichment,
                   pileup=domains$pileup,
                   name=domains$name)
domains <- sortSeqlevels(domains)
domains <- sort(domains)
domains$name <- paste("peak_", 1:length(domains), sep="") # rename peaks based on proper order

blacklist <- import(con="/mnt/mfs/ctcn/resources/encodeBlacklists/hg38-blacklist.v2.clean.bed.gz", format="bed")
seqlevelsStyle(blacklist) <- "Ensembl"

domains$blacklist <- overlapsAny(domains, blacklist)

domains <- as.data.frame(domains)
colnames(domains)[1] <- "chr"

write.csv(domains, file="H3K9acDomains.csv", quote=FALSE, row.names=FALSE)