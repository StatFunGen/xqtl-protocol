# Reads in the count data and calculates the residuals after regressing out all
# the technical variables identified in selectCovariates. Offsets based on a
# "median" sample are added back to the residuals to retain peak-height information.
#
# Hans Klein
# 19 July 2022

suppressPackageStartupMessages({
  library(argparser)
  library(limma)
  library(edgeR)
})

p <- arg_parser("Regress technical covariates out of H3K9ac counts (voom) and return residuals + median offset")
p <- add_argument(p, "--counts-rds",      help = "H3K9acCounts.rds")
p <- add_argument(p, "--qc",              help = "qualityMetrics.csv")
p <- add_argument(p, "--sample-sheet",    help = "sampleSheet.csv")
p <- add_argument(p, "--lib",             help = "Path to readH3k9acCounts.R")
p <- add_argument(p, "--phenotype-rdata", help = "ROSMAP basic phenotype .Rdata (contains 'rosmap')")
p <- add_argument(p, "--old-phenotype",   help = "Older ROSMAP basic phenotype table (for samples dropped from newer files)")
p <- add_argument(p, "--output-resids",    help = "Output residuals table (Batch model)")
p <- add_argument(p, "--output-resids-nb", help = "Output residuals table (no-Batch model)")
argv <- parse_args(p)

source(argv$lib)

# Read counts, remove blacklist peaks, and attach technical variables.
readData <- function() {

  h3k9ac <- readH3k9acCounts(argv$counts_rds, argv$qc, argv$sample_sheet)
  h3k9ac <- h3k9ac[!rowData(h3k9ac)$blacklist, ]

  # Add phenotypes
  load(argv$phenotype_rdata)  # 'rosmap'
  misPheno <- colnames(h3k9ac)[!colnames(h3k9ac) %in% rownames(rosmap)]
  rosmap <- rosmap[colnames(h3k9ac), c("projid", "study", "pmi")]
  colData(h3k9ac) <- cbind(colData(h3k9ac), rosmap)

  # Add phenotypes for samples dropped from newer phenotype files
  oldPheno <- read.table(argv$old_phenotype, sep = "\t", header = TRUE,
                         stringsAsFactors = FALSE, colClasses = c(projid = "character"))
  oldPheno <- oldPheno[oldPheno$projid == misPheno, c("projid", "study", "pmi")]
  colData(h3k9ac[, misPheno])$projid <- oldPheno$projid
  colData(h3k9ac[, misPheno])$study <- oldPheno$study
  colData(h3k9ac[, misPheno])$pmi <- oldPheno$pmi

  # Impute missing PMI with the median
  h3k9ac$pmi[is.na(h3k9ac$pmi)] <- median(h3k9ac$pmi, na.rm = TRUE)

  # Transform technical variables as needed for adjustment
  h3k9ac$LogUniqueFrags   <- log(h3k9ac$UniqueFrags)
  h3k9ac$LOG_AT_DROPOUT   <- log(h3k9ac$AT_DROPOUT)
  h3k9ac$LOG_GC_DROPOUT   <- log(h3k9ac$GC_DROPOUT)
  h3k9ac$LogPercMt        <- log(h3k9ac$PercMt)
  h3k9ac$LogTotalNumPeaks <- log(h3k9ac$TotalNumPeaks)

  return(h3k9ac)
}

# Calculate the offset (H3K9ac levels of a median sample) for a given model fit.
predictOffset <- function(fit) {
  usedFactors <- c("Batch", "study")
  usedContinuous <- c("pmi", "LOG_AT_DROPOUT", "UniqueFragsFRiP", "NRF",
                      "LogPercMt", "CCFragSize", "LogUniqueFrags",
                      "LogTotalNumPeaks", "LOG_GC_DROPOUT",
                      "MedianWidth", "PBC2")
  facInd <- unlist(lapply(as.list(usedFactors), function(f) { return(grep(paste("^", f, sep = ""), colnames(fit$design))) }))
  contInd <- unlist(lapply(as.list(usedContinuous), function(f) { return(grep(paste("^", f, sep = ""), colnames(fit$design))) }))
  stopifnot(!any(duplicated(c(1, facInd, contInd))))
  stopifnot(all(c(1, facInd, contInd) %in% 1:ncol(fit$design)))
  stopifnot(1:ncol(fit$design) %in% c(1, facInd, contInd))

  D <- fit$design
  D[, facInd] <- 0
  medContVals <- apply(D[, contInd], 2, median)
  for (i in 1:length(medContVals)) {
    D[, names(medContVals)[i]] <- medContVals[i]
  }

  stopifnot(all(colnames(coefficients(fit)) == colnames(D)))
  offsets <- apply(coefficients(fit), 1, function(c) {
    return(D %*% c)
  })
  offsets <- t(offsets)
  colnames(offsets) <- rownames(fit$design)

  return(offsets)
}

# Run pipeline and return residuals and offset matrices.
runVoom <- function(model) {

  h3k9ac <- readData()

  # Convert to DGEList and apply TMM normalization
  dge <- DGEList(counts = assay(h3k9ac), samples = colData(h3k9ac))
  dge <- calcNormFactors(dge, method = "TMM")

  # Fit model
  design <- model.matrix(model, data = dge$samples)
  stopifnot(is.fullrank(design))
  v <- voom(dge, design, plot = FALSE)
  fit <- lmFit(v, v$design)
  fit <- eBayes(fit)

  # Offset and residuals
  offset <- predictOffset(fit)
  resids <- residuals(fit, y = v)
  stopifnot(all(rownames(offset) == rownames(resids)) &
            all(colnames(offset) == colnames(resids)))

  return(list(offset = offset, resids = resids))
}

# Model with Batch (12 variables + Batch)
model <- ~ pmi + LOG_AT_DROPOUT + UniqueFragsFRiP + NRF + LogPercMt +
  CCFragSize + LogUniqueFrags + LogTotalNumPeaks + Batch + LOG_GC_DROPOUT +
  study + MedianWidth + PBC2
values <- runVoom(model)
write.table(values$offset + values$resids, file = argv$output_resids, quote = FALSE)

# Model without Batch (same 12 variables)
modelNB <- ~ pmi + LOG_AT_DROPOUT + UniqueFragsFRiP + NRF + LogPercMt +
  CCFragSize + LogUniqueFrags + LogTotalNumPeaks + LOG_GC_DROPOUT +
  MedianWidth + PBC2 + study
valuesNB <- runVoom(modelNB)
write.table(valuesNB$offset + valuesNB$resids, file = argv$output_resids_nb, quote = FALSE)
