#!/bin/bash

#$-l h_vmem=32G,h_rt=12:00:00
#$-t 1-727
#$-tc 64
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/logs/'$JOB_NAME'.o'$JOB_ID'.'$TASK_ID'
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/logs/'$JOB_NAME'.e'$JOB_ID'.'$TASK_ID'
#$-N chipQc_n727

module load R/4.0

sample=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $1}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/scripts/peakFiles_n727.csv`
bed=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $2}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/scripts/peakFiles_n727.csv`
peaks=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $3}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/scripts/peakFiles_n727.csv`
blacklist=/mnt/mfs/ctcn/resources/encodeBlacklists/hg38-blacklist.v2.clean.bed.gz
epigenome=/mnt/mfs/ctcn/resources/chromatinStates/coreMarks_15/hg38/E073_15_coreMarks_hg38lift_dense.bed.gz

mkdir -p /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}

Rscript --vanilla -e "source(\"/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/scripts/lib/encodeMetrics.R\"); \
  metrics <- encodeMetrics(sample=\"${sample}\", bed=\"${bed}\", peaks=\"${peaks}\", blacklist=\"${blacklist}\"); \
  write.csv(metrics, quote=FALSE, row.names=FALSE, file=\"/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}/${sample}_metrics.csv\"); \
  crossCor <- crossCorrelation(bed=\"${bed}\", blacklist=\"${blacklist}\", rmDup=TRUE); \
  write.csv(crossCor, quote=FALSE, row.names=FALSE, file=\"/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}/${sample}_crossCor.csv\"); \
  epigenome <- strsplit(\"${epigenome}\", \",\")[[1]]; \
  for (i in 1:length(epigenome)) { \
    chromStates <- chromatinStateOverlap(sample=\"${sample}\", bed=\"${bed}\", chromState=epigenome[i], blacklist=\"${blacklist}\", rmDup=TRUE); \
    outFile <- paste(\"/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}/${sample}_\", chromStates\$Epigenome[1], \"_chromStates.csv\", sep=\"\"); \
    write.csv(chromStates, quote=FALSE, row.names=FALSE, file=outFile); \
  }"

mv logs/chipQc_n727.e${JOB_ID}.${SGE_TASK_ID} logs/chipQc_n727.e.${sample}
mv logs/chipQc_n727.o${JOB_ID}.${SGE_TASK_ID} logs/chipQc_n727.o.${sample}
