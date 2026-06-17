#!/bin/bash

#$-l h_vmem=32G,h_rt=24:00:00
#$-t 1-726
#$-tc 726
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/logs/'$JOB_NAME'.o'$JOB_ID'.'$TASK_ID'
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/logs/'$JOB_NAME'.e'$JOB_ID'.'$TASK_ID'
#$-N alignmentQc_n726

picard=/mnt/mfs/ctcn/tools/picard_2.23.6

readarray bamfiles < /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/scripts/alignedBamFiles_n726.txt
bamfull=${bamfiles[((${SGE_TASK_ID} - 1))]}
bamdir=$(dirname -- ${bamfull})
bamfile=$(basename -- ${bamfull})
sample=${bamfile%.*}

java -Xmx24G -jar ${picard}/picard.jar CollectMultipleMetrics \
  REFERENCE_SEQUENCE=/mnt/mfs/ctcn/resources/GRCh38/v1/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
  PROGRAM=CollectAlignmentSummaryMetrics \
  PROGRAM=QualityScoreDistribution \
  PROGRAM=MeanQualityByCycle \
  PROGRAM=CollectBaseDistributionByCycle \
  PROGRAM=CollectGcBiasMetrics \
  VALIDATION_STRINGENCY=SILENT \
  INPUT=${bamfull} \
  OUTPUT=${bamdir}/${sample}

java -Xmx24G -jar ${picard}/picard.jar CollectRnaSeqMetrics \
  REF_FLAT=/mnt/mfs/ctcn/resources/GRCh38/v1/Homo_sapiens.GRCh38.91.refFat \
  STRAND_SPECIFICITY=NONE \
  CHART_OUTPUT=${bamdir}/${sample}.rna_metrics.pdf \
  VALIDATION_STRINGENCY=SILENT \
  INPUT=${bamfull} \
  OUTPUT=${bamdir}/${sample}.rna_metrics

mv logs/alignmentQc_n726.e${JOB_ID}.${SGE_TASK_ID} logs/alignmentQc_n726.e.${sample}
mv logs/alignmentQc_n726.o${JOB_ID}.${SGE_TASK_ID} logs/alignmentQc_n726.o.${sample}
