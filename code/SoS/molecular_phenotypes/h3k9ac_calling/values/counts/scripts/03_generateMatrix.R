samples <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/experimentDesign/sampleSheet.csv",
                    header=TRUE, stringsAsFactors=FALSE)
samples <- samples[samples$Quality == "Pass", ]

domains <- read.csv("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/H3K9acDomains.csv",
                    header=TRUE, stringsAsFactors=FALSE)

countMat <- matrix(as.integer(NA), nrow=nrow(domains), ncol=nrow(samples))
rownames(countMat) <- domains$name
colnames(countMat) <- samples$SampleID
for (s in colnames(countMat)) {
  load(paste("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/tmp/", s, ".RData", sep=""))
  stopifnot(all(names(counts) == rownames(countMat)))
  countMat[, s] <- counts
}

write.table(countMat, file="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/H3K9acCounts.txt", quote=FALSE)