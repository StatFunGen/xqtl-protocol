#!/bin/bash

#$-l h_vmem=32G,h_rt=24:00:00
#$-t 1-669
#$-tc 70
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/logs/'$JOB_NAME'.o'$JOB_ID'.'$TASK_ID'
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/logs/'$JOB_NAME'.e'$JOB_ID'.'$TASK_ID'
#$-N countOverlaps_n669

mkdir -p /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/logs
mkdir -p /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/values/counts/tmp

readarray -t samples < <(cat /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/experimentDesign/sampleSheet.csv | grep Pass | cut -d, -f 1)
sample=${samples[((${SGE_TASK_ID} - 1))]}
/mnt/mfs/cluster/bin/R-4.0.0/bin/Rscript --vanilla scripts/02_countFragments.R ${sample}

mv logs/countOverlaps_n669.e${JOB_ID}.${SGE_TASK_ID} logs/countOverlaps_n669.e.${sample}
mv logs/countOverlaps_n669.o${JOB_ID}.${SGE_TASK_ID} logs/countOverlaps_n669.o.${sample}
