# Read the raw count data (with domain information and quality-control metrics)
# as a SummarizedExperiment. Sourced by the entry-point scripts, which pass the paths.
#
# Hans
# July 14, 2022

suppressPackageStartupMessages({
  library(vroom)
  library(SummarizedExperiment)
})

readH3k9acCounts <- function(countsRds, qcFile, sampleSheet) {

  # counts assay + domain rowRanges (built by 03_generateMatrix.R)
  se <- readRDS(countsRds)

  qualityMetrics <- as.data.frame(vroom(qcFile, delim = " ", show_col_types = FALSE))
  rownames(qualityMetrics) <- qualityMetrics$Sample
  qualityMetrics$Sample <- NULL
  qualityMetrics$ESTIMATED_LIBRARY_SIZE <- NULL  # does not exist for single-end reads

  batches <- as.data.frame(vroom(sampleSheet, show_col_types = FALSE))
  rownames(batches) <- batches$SampleID

  stopifnot(all(colnames(se) %in% rownames(qualityMetrics)))
  qualityMetrics <- qualityMetrics[colnames(se), ]
  stopifnot(all(colnames(se) %in% rownames(batches)))
  batches <- batches[colnames(se), ]

  stopifnot(all(rownames(qualityMetrics) == rownames(batches)))
  qualityMetrics$Batch <- factor(batches$Batch)

  colData(se) <- DataFrame(qualityMetrics, row.names = colnames(se))

  return(se)
}
