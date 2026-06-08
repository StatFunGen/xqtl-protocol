source("/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/scripts/lib/readQualityMetrics.R")

qualityMetrics <- readChIPseqQC(sourcePicard="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned", sourceMacs2="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks", paired=FALSE)
write.table(qualityMetrics, file="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/qualityMetrics.csv",
            col.names=TRUE, row.names=FALSE, quote=FALSE)
chromatinMetrics <- aggregateChromatinStateFrequencies(peaksDir="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks")
if (!is.na(chromatinMetrics)[1]) {
  write.table(chromatinMetrics, file="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/chromatinMetrics.csv",
              col.names=TRUE, row.names=FALSE, quote=FALSE)
}
