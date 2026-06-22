expSheet <- read.csv("orig/ChIP-Seq data summary 140801.csv", sep="\t",
                     stringsAsFactors=FALSE)

# 66754397 (remove, failed QC and bam file is missing, was sequenced twice)
expSheet <- expSheet[expSheet$ProjID != "66754397", ]
# 11464261 (remove once, duplicate processed in 4-plex and 8-plex batch, unsure which bam we have but sample failed anyway)
expSheet <- expSheet[!(expSheet$ProjID == "11464261" & expSheet$Pool == "8-Plex"), ]

natNeuro <- read.csv("../../synapseRelease1/experimentalData/QualityControl.csv", sep="\t",
                     stringsAsFactors=FALSE, colClasses=c("SampleID"="character"))
stopifnot(all(natNeuro$SampleID %in% expSheet$ProjID))

# Add QC filter from the Nature Neuroscience paper
ind <- match(natNeuro$SampleID, expSheet$ProjID)
stopifnot(all(expSheet$ChIP.Batch[ind] == natNeuro$Batch))
expSheet$Quality <- "Fail"
expSheet$Quality[ind] <- "Pass"

# Match name of controls to bam files
expSheet$ProjID[expSheet$ProjID == "Positive Control"] <- c("PC-Pool-1_8plex", "PC-Pool-2_8plex")
expSheet$ProjID[expSheet$ProjID == "Negative Control"] <- c("NC-Pool-1_8plex", "NC-Pool-2_8plex")

sampleSheet <- expSheet[, c(1,5, 19)]
colnames(sampleSheet) <- c("SampleID", "Batch", "Quality")

# Add merged controls
# sampleSheet <- rbind(sampleSheet, 
#                      data.frame(SampleID=c("PC-Pool", "NC-Pool"), Batch=c(NA, NA), Quality=c("Fail", "Fail"), stringsAsFactors=FALSE))

# Check that all bam files exist
stopifnot(all(file.exists(paste("../source/", sampleSheet$SampleID, ".bam", sep=""))))


write.table(sampleSheet, file="sampleSheet.csv", row.names=FALSE, sep=",", quote=FALSE)
