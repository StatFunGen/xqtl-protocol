#!/bin/bash

readarray -t samples < <(tail --lines +2 ../experimentDesign/sampleSheet.csv | cut -d, -f1)

for s in "${samples[@]}"
do
   echo "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/source/${s}.bam" >> /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/scripts/sourceBam_n726.txt
done
