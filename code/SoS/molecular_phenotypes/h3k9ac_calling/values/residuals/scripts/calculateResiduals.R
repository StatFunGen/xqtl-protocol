# Reads in the count data and calculates the residuals after regression out
# all the technical variables identified in the folder selectCovariates.
# Offsets based on an "median" samples are added back to the residuals to
# retain information about peak height.
#
# Hans Klein
# 19 July 2022

library(limma)
library(edgeR)
source("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/scripts/readH3k9acCounts.R")


#############################################################################
# Read in count data, remove peaks from blacklist, and add tech. variables. #
#############################################################################
readData <- function() {
  
  # Read counts and peaks
  h3k9ac <- readH3k9acCounts()
  h3k9ac <- h3k9ac[!rowData(h3k9ac)$blacklist, ]
  
  # Add phenotypes
  load("/mnt/mfs/ctcn/datasets/rosmap/phenotypes/2020Dec10/Rdata/rosmap_basic.Rdata")
  misPheno <- colnames(h3k9ac)[!colnames(h3k9ac) %in% rownames(rosmap)] # "90214403" (has been removed after Jan 2018)
  rosmap <- rosmap[colnames(h3k9ac), c("projid", "study", "pmi")]
  colData(h3k9ac) <- cbind(colData(h3k9ac), rosmap)
  
  # Add phenotypes of 90214403, which has been removed in newer phenotype files
  oldPheno <- read.table("/mnt/mfs/ctcn/datasets/rosmap/phenotypes/2018Jan26/dataset_157_basic.ccw.unix.txt",
                         sep="\t", header=TRUE, stringsAsFactors=FALSE, colClasses=c(projid="character"))
  oldPheno <- oldPheno[oldPheno$projid == misPheno, c("projid", "study", "pmi")]
  colData(h3k9ac[, misPheno])$projid <- oldPheno$projid
  colData(h3k9ac[, misPheno])$study <- oldPheno$study
  colData(h3k9ac[, misPheno])$pmi <- oldPheno$pmi
  
  # One sample (21411459) has missing PMI - impute median PMI
  h3k9ac$pmi[is.na(h3k9ac$pmi)] <- median(h3k9ac$pmi, na.rm=TRUE)
  
  # Transform technical variables as needed for adjustment
  h3k9ac$LogUniqueFrags <- log(h3k9ac$UniqueFrags)
  h3k9ac$LOG_AT_DROPOUT <- log(h3k9ac$AT_DROPOUT)
  h3k9ac$LOG_GC_DROPOUT <- log(h3k9ac$GC_DROPOUT)
  h3k9ac$LogPercMt <- log(h3k9ac$PercMt)
  h3k9ac$LogTotalNumPeaks <- log(h3k9ac$TotalNumPeaks)
  
  return(h3k9ac)
}


###############################################################################
# Calculate the offset (= H3K9ac levels of a median sample) for a given model #
###############################################################################
predictOffset <- function (fit) {
  usedFactors <- c("Batch", "study")
  usedContinuous <- c("pmi", "LOG_AT_DROPOUT", "UniqueFragsFRiP", "NRF",
                      "LogPercMt", "CCFragSize", "LogUniqueFrags",
                      "LogTotalNumPeaks", "LOG_GC_DROPOUT",
                      "MedianWidth", "PBC2")
  facInd <- unlist(lapply(as.list(usedFactors), function (f) {return(grep(paste("^", f, sep=""), colnames(fit$design)))}))
  contInd <- unlist(lapply(as.list(usedContinuous), function (f) {return(grep(paste("^", f, sep=""), colnames(fit$design)))}))
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
  offsets <- apply(coefficients(fit), 1, function (c) {
    return(D %*% c)
  })
  offsets <- t(offsets)
  colnames(offsets) <- rownames(fit$design)

  return(offsets)
}



#########################################################
# Run pipeline and return residuals and offset matrices #
#########################################################
runVoom <- function(model) {

  # Get data
  h3k9ac <- readData()
  
  # Convert to dge list and apply TMM normalization
  dge <- DGEList(counts=assay(h3k9ac), samples=colData(h3k9ac))
  dge <- calcNormFactors(dge, method="TMM")
  
  # Fit model
  design <- model.matrix(model, data=dge$samples)
  stopifnot(is.fullrank(design))
  v <- voom(dge, design, plot=FALSE)
  fit <- lmFit(v, v$design)
  fit <- eBayes(fit)
  
  # Get offset and residuals
  offset <- predictOffset(fit)
  resids <- residuals(fit, y=v)
  stopifnot(all(rownames(offset) == rownames(resids)) &
            all(colnames(offset) == colnames(resids)))
  
  return(list(offset=offset, resids=resids))
}



# Model with Batch (see selectCovariates/RosmapH3k9acCovariates.html)
# 12 variables + Batch (with 62 factor levels)
model <- ~ pmi + LOG_AT_DROPOUT + UniqueFragsFRiP + NRF + LogPercMt +
  CCFragSize + LogUniqueFrags + LogTotalNumPeaks + Batch + LOG_GC_DROPOUT +
  study + MedianWidth + PBC2
values <- runVoom(model)
write.table(values$offset + values$resids,
            file="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/residuals/H3K9acResiduals.txt", quote=FALSE)


# Model without Batch (see selectCovariates/RosmapH3k9acCovariates_noBatch.html)
# 12 variables (same 12 variables as with batch were selected)
modelNB <- ~ pmi + LOG_AT_DROPOUT + UniqueFragsFRiP + NRF + LogPercMt +
  CCFragSize + LogUniqueFrags + LogTotalNumPeaks + LOG_GC_DROPOUT +
  MedianWidth + PBC2 + study
valuesNB <- runVoom(modelNB)
write.table(valuesNB$offset + valuesNB$resids,
            file="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/residuals/H3K9acResidualsNB.txt", quote=FALSE)
