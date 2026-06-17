# Read in the raw count data with domain information and quality control
# metrics as SummarizedExperiment object.
#
# Hans
# July 14, 2022

library(SummarizedExperiment)

readH3k9acCounts <- function() {
  
  counts <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/H3K9acCounts.txt", check.names=FALSE)
  domains <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/H3K9acDomains.csv",
                      header=TRUE, stringsAsFactors=FALSE)
  domains <- GRanges(IRanges(start=domains$start, end=domains$end),
                     seqnames=domains$chr,
                     strand=domains$strand,
                     log10p=domains$log10p,
                     log10q=domains$log10q,
                     foldEnrichment=domains$foldEnrichment,
                     pileup=domains$pileup,
                     blacklist=domains$blacklist,
                     name=domains$name)
  qualityMetrics <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/qualityMetrics.csv",
                               header=TRUE, row.names="Sample")
  qualityMetrics$ESTIMATED_LIBRARY_SIZE <- NULL # does not exist for single-end reads
  batches <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/experimentDesign/sampleSheet.csv",
                      header=TRUE, row.names="SampleID", stringsAsFactors=FALSE)
  
  stopifnot(all(names(domains) == rownames(counts)))
  stopifnot(all(colnames(counts) %in% rownames(qualityMetrics)))
  qualityMetrics <- qualityMetrics[colnames(counts), ]
  stopifnot(all(colnames(counts) %in% rownames(batches)))
  batches <- batches[colnames(counts), ]
  
  stopifnot(all(rownames(qualityMetrics) == rownames(batches)))
  qualityMetrics$Batch <- factor(batches$Batch)
  
  h3k9ac <- SummarizedExperiment(assays=list(counts=counts), rowRanges=domains,
                                 colData=qualityMetrics)
  
  return(h3k9ac)
}