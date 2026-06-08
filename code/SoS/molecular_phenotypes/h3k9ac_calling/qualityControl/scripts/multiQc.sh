#!/bin/bash
#$-l h_vmem=16G,h_rt=48:00:00
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl
#$-o /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/logs/multiQc.o
#$-e /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/logs/multiQc.e
#$-N multiQc

module load CONDA/2019.07
conda activate MultiQC

fastqDir=/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq
alnDir=/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/bowtie2Aligned
macs2Dir=/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/macs2Peaks

configFile=/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/qualityControl/scripts/multiqc_config.yaml

multiqc -f -i "H3K9ac DLPFC ChIP-seq" -o multiqc -c ${configFile} -v ${fastqDir} ${alnDir} ${macs2Dir}
