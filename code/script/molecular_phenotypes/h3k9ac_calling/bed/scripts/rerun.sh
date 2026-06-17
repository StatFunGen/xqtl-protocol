#!/bin/bash

#$-l h_vmem=64G,h_rt=24:00:00
#$-t 1-1
#$-tc 1
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/logs/'$JOB_NAME'.o'$JOB_ID'.'$TASK_ID'
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/logs/'$JOB_NAME'.e'$JOB_ID'.'$TASK_ID'
#$-N bamToBed_n728

wigToBigWig=/mnt/mfs/ctcn/tools/ucscTools/wigToBigWig
bamToBed=/mnt/mfs/cluster/bin/bin/bamToBed
module load Bedtools/2.30
module load SAMTOOLS/1.14
module load R/4.0

readarray bamfiles < /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/scripts/alignedBamFiles_n728.txt
bam=/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned/positivePool/positivePool.bam
sample=$(basename -- ${bam})
sample=${sample%.*}

mkdir -p /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}

# -q 2 (map quality > 2)
samtools view -b -q 2 ${bam} \
  | ${bamToBed} -i stdin > /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}.bed

# -q 2 (map quality > 2) -F 0x400 (rm duplicates)
samtools view -b -F 0x400 -q 2 ${bam} \
  | ${bamToBed} -i stdin > /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}_noDups.bed

Rscript -e "f <- read.table(\"${sample}/${sample}_noDups.bed\", sep=\"\\t\", header=FALSE); \
  l <- f[, 3] - f[, 2]; \
  o <- data.frame(Fragments=length(l), Bases=sum(as.numeric(l)), ScaleFactor=50000000 * 100 / sum(as.numeric(l))); \
  write.table(o, file=\"${sample}/${sample}_stats.txt\", sep=\"\\t\", quote=FALSE, row.names=FALSE)"

scaleFactor=$(awk 'NR == 2 {print $3}' ${sample}/${sample}_stats.txt)
cut -f 1,2 /mnt/mfs/ctcn/resources/GRCh38/v1/Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai > /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/genome.txt
bedtools genomecov -i /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}_noDups.bed -bg -scale ${scaleFactor} -trackline -g /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/genome.txt > /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}.bg
${wigToBigWig} /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}.bg /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/genome.txt /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}.bw

rm /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}_noDups.bed
rm /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/${sample}.bg
rm /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/${sample}/genome.txt


mv logs/bamToBed_n728.e${JOB_ID}.${SGE_TASK_ID} logs/bamToBed_n728.e.${sample}
mv logs/bamToBed_n728.o${JOB_ID}.${SGE_TASK_ID} logs/bamToBed_n728.o.${sample}
