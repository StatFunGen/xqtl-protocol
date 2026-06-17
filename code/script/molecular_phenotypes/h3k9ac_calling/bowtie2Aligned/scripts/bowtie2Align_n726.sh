#!/bin/bash

#$-l h_vmem=64G,h_rt=24:00:00
#$-t 1-726
#$-tc 64
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/logs/'$JOB_NAME'.o'$JOB_ID'.'$TASK_ID'
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/logs/'$JOB_NAME'.e'$JOB_ID'.'$TASK_ID'
#$-N bowtie2Align_n726

picard=/mnt/mfs/ctcn/tools/picard_2.23.6

fastqFile=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $1}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/scripts/fastqFiles_n726.txt`
sampleName=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $2}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/scripts/fastqFiles_n726.txt`
date=`date +%Y-%m-%dT%H:%M:%S:%z`

mkdir /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}

/mnt/mfs/ctcn/tools/bowtie2-2.3.5.1-linux-x86_64/bowtie2 \
  -q --phred33 \
  --local --very-sensitive-local \
  --threads 4 \
  --rg "ID:${sampleName}" \
  --rg "SM:${sampleName}" \
  --rg "CN:CTCN" \
  --rg "DT:${date}" \
  -x /mnt/mfs/ctcn/resources/GRCh38/v1/Bowtie2_2.3.5.1/GRCh38.91 \
  -U ${fastqFile} \
  -S /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.sam

java -Xmx48g -jar ${picard}/picard.jar SamFormatConverter \
  INPUT="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.sam" \
  OUTPUT="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}_raw.bam" \
  VALIDATION_STRINGENCY=SILENT

java -Xmx48g -jar ${picard}/picard.jar SortSam \
  INPUT="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}_raw.bam" \
  OUTPUT="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}_sorted.bam" \
  SORT_ORDER=coordinate \
  VALIDATION_STRINGENCY=SILENT

java -Xmx48g -jar ${picard}/picard.jar MarkDuplicates \
  INPUT="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}_sorted.bam" \
  OUTPUT="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.bam" \
  VALIDATION_STRINGENCY=SILENT \
  METRICS_FILE="/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.duplicate_metrics" \
  REMOVE_SEQUENCING_DUPLICATES=false \
  REMOVE_DUPLICATES=false \
  ASSUME_SORTED=false \
  OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 \
  CREATE_INDEX=true \
  CREATE_MD5_FILE=true

rm "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.sam"
rm "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}_raw.bam"
rm "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}_sorted.bam"

echo "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.bam" >> /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/scripts/alignedBamFiles_n726.txt

mv logs/bowtie2Align_n726.e${JOB_ID}.${SGE_TASK_ID} logs/bowtie2Align_n726.e.${sampleName}
mv logs/bowtie2Align_n726.o${JOB_ID}.${SGE_TASK_ID} logs/bowtie2Align_n726.o.${sampleName}

endBowtieLog=`grep -n -m 1 INFO logs/bowtie2Align_n726.e.${sampleName} | cut -d : -f 1`
endBowtieLog=$(( $endBowtieLog - 1 ))
head -n $endBowtieLog logs/bowtie2Align_n726.e.${sampleName} > "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/${sampleName}/${sampleName}.log"
