#!/bin/bash

#$-l h_vmem=16G
#$-t 1-726
#$-tc 32
#$-wd /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq
#$-N bamToFastq_n726

source /mnt/mfs/cluster/bin/HgrcPathSetup.sh

picard=/mnt/mfs/ctcn/tools/picard_2.20.3/picard.jar

readarray filenames < /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/scripts/sourceBam_n726.txt
inputfile=${filenames[((${SGE_TASK_ID}-1))]}
outfilename=(`basename ${inputfile} .bam`)
mkdir /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}

java -Xmx8G -jar ${picard} SamToFastq VALIDATION_STRINGENCY=LENIENT \
	INCLUDE_NON_PF_READS=true \
	INPUT=${inputfile} \
	VERBOSITY=DEBUG \
	CREATE_MD5_FILE=true \
	FASTQ=/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}/${outfilename}.fastq

gzip /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}/${outfilename}.fastq

echo "/mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/${outfilename}/${outfilename}.fastq.gz,${outfilename}" >> /mnt/mfs/ctcn/datasets/rosmap/h3k9ac/dlpfcTissue/batch1/fastq/scripts/fastqFiles_n726.txt

mv bamToFastq_n726.e*.${SGE_TASK_ID} logs/bamToFastq_n726.e.${outfilename}
mv bamToFastq_n726.o*.${SGE_TASK_ID} logs/bamToFastq_n726.o.${outfilename}
