# Read in the residuals data with domain information and quality control
# metrics as SummarizedExperiment object.
#
# Hans
# July 19, 2022

library(SummarizedExperiment)

readH3k9acResiduals <- function(noBatch=FALSE) {
  
  if (noBatch) {
    resids <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/residuals/H3K9acResidualsNB.txt", check.names=FALSE)
  } else {
    resids <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/residuals/H3K9acResiduals.txt", check.names=FALSE)    
  }
  resids <- as.matrix(resids)
  
  domainsDf <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/domains/domains.csv",
                        header=TRUE, stringsAsFactors=FALSE, check.names=FALSE)
  domains <- GRanges(IRanges(start=domainsDf$start, end=domainsDf$end),
                     seqnames=domainsDf$chr,
                     strand=domainsDf$strand)
  mcols(domains) <- domainsDf[, -(1:5)]
  names(domains) <- domains$name
  stopifnot(all(rownames(resids) %in% names(domains)))
  domains <- domains[rownames(resids)]
  
  qualityMetrics <- read.table("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/qualityMetrics.csv",
                               header=TRUE, row.names="Sample")
  qualityMetrics$ESTIMATED_LIBRARY_SIZE <- NULL # does not exist for single-end reads
  batches <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/experimentDesign/sampleSheet.csv",
                      header=TRUE, row.names="SampleID", stringsAsFactors=FALSE)
  
  stopifnot(all(names(domains) == rownames(resids)))
  stopifnot(all(colnames(resids) %in% rownames(qualityMetrics)))
  qualityMetrics <- qualityMetrics[colnames(resids), ]
  stopifnot(all(colnames(resids) %in% rownames(batches)))
  batches <- batches[colnames(resids), ]
  
  stopifnot(all(rownames(qualityMetrics) == rownames(batches)))
  qualityMetrics$Batch <- factor(batches$Batch)
  
  h3k9ac <- SummarizedExperiment(assays=list(h3k9ac=resids), rowRanges=domains,
                                 colData=qualityMetrics)
  
  return(h3k9ac)
}