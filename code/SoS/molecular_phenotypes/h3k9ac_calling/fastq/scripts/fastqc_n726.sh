#!/bin/bash

#$-l h_vmem=8G
#$-t 1-726
#$-tc 32
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq
#$-N fastqc_n726

source /mnt/mfs/cluster/bin/HgrcPathSetup.sh

fastqFile=`awk -F "," '{if (NR=='${SGE_TASK_ID}') print $1}' /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/scripts/fastqFiles_n726.txt`
outfilename=(`dirname ${fastqFile} .fastq.gz`)
outfilename=(`basename ${outfilename} `)

mkdir -p /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}
mkdir -p /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}/fastqc

fastqc \
  -o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}/fastqc \
  -d /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}/fastqc \
  --extract ${fastqFile}

mv fastqc_n726.e*.${SGE_TASK_ID} logs/fastqc_n726.e.${outfilename}
mv fastqc_n726.o*.${SGE_TASK_ID} logs/fastqc_n726.o.${outfilename}
