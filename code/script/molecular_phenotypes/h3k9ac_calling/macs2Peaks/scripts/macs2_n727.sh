#!/bin/bash

#$-l h_vmem=12G,h_rt=6:00:00
#$-t 1-727
#$-tc 64
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/logs/'$JOB_NAME'.o'$JOB_ID'.'$TASK_ID'
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/logs/'$JOB_NAME'.e'$JOB_ID'.'$TASK_ID'
#$-N macs2_n727

module load CONDA/2019.07
conda activate MACS2
macs2 --version

chipBed=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $1}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/scripts/matchedBedFiles_n727.txt`
sample=$(basename -- "${chipBed}")
sample=${sample%.*}
mkdir /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}

controlBed=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $2}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bed/scripts/matchedBedFiles_n727.txt`

macs2 callpeak -t ${chipBed} \
  -c ${controlBed} \
  --format BED \
  --gsize hs \
  --keep-dup auto \
  --qvalue 0.05 \
  --broad \
  --broad-cutoff 0.1 \
  --name ${sample} \
  --outdir /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}

cat /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}/${sample}_peaks.xls | grep -v '#' | grep -v '^$' | grep -v 'chr' | cut -f 1-3,6 > /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}/${sample}_peaks.bed

echo "${sample},${chipBed},/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/${sample}/${sample}_peaks.bed" >> /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks/scripts/peakFiles_n727.csv

mv logs/macs2_n727.e${JOB_ID}.${SGE_TASK_ID} logs/macs2_n727.e.${sample}
mv logs/macs2_n727.o${JOB_ID}.${SGE_TASK_ID} logs/macs2_n727.o.${sample}
