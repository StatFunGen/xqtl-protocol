# Read the residuals data (with domain information and quality-control metrics)
# as a SummarizedExperiment. Sourced by the entry-point scripts, which pass the paths.
#
# Hans
# July 19, 2022

suppressPackageStartupMessages({
  library(vroom)
  library(GenomicRanges)
  library(SummarizedExperiment)
})

readH3k9acResiduals <- function(residFile, domainsCsv, qcFile, sampleSheet) {

  # Residuals are an R-matrix-format text file (domain rownames + sample header);
  # read.table handles that layout natively (vroom has no row-name concept).
  resids <- as.matrix(read.table(residFile, check.names = FALSE))

  domainsDf <- as.data.frame(vroom(domainsCsv, show_col_types = FALSE))
  domains <- GRanges(
    seqnames = domainsDf$chr,
    ranges   = IRanges(start = domainsDf$start, end = domainsDf$end),
    strand   = domainsDf$strand)
  mcols(domains) <- domainsDf[, -(1:5)]
  names(domains) <- domains$name
  stopifnot(all(rownames(resids) %in% names(domains)))
  domains <- domains[rownames(resids)]

  qualityMetrics <- as.data.frame(vroom(qcFile, delim = " ", show_col_types = FALSE))
  rownames(qualityMetrics) <- qualityMetrics$Sample
  qualityMetrics$Sample <- NULL
  qualityMetrics$ESTIMATED_LIBRARY_SIZE <- NULL

  batches <- as.data.frame(vroom(sampleSheet, show_col_types = FALSE))
  rownames(batches) <- batches$SampleID

  stopifnot(all(colnames(resids) %in% rownames(qualityMetrics)))
  qualityMetrics <- qualityMetrics[colnames(resids), ]
  stopifnot(all(colnames(resids) %in% rownames(batches)))
  batches <- batches[colnames(resids), ]

  stopifnot(all(rownames(qualityMetrics) == rownames(batches)))
  qualityMetrics$Batch <- factor(batches$Batch)

  h3k9ac <- SummarizedExperiment(assays = list(h3k9ac = resids), rowRanges = domains,
                                 colData = DataFrame(qualityMetrics, row.names = colnames(resids)))
  return(h3k9ac)
}
