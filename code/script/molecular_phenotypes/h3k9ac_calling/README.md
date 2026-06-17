# Histone acetylation peak calling pipeline 

## 0. Bam to fastq
If the file format for raw data is bam, the "bamtofastq.sh" in the fastq folder can convert it back to fastq so that QC and other processing can be done.

Otherwise, users can skip this step.

## 1. fastqc
Besides 0, The scripts in the fastq folder also qc the fastq files.

## 2.Alignment
The scripts in the bowtie2Aligned folder (re)align the fastq files realigned to bam files. 

## 3. Bam to bed
The scripts in the bed folder convert the output of 2 into bed format, ready for downstream analysis.

## 4. Bed to peak
The scripts in the macs2peaks folder process the output from 3

## 5.QC
The scripts in qualityCountrol folder qc the output from step 2-4

## 6.Extracting results
The scripts in the values/domain folder determined the range of each peak. And the scripts in the values/counts folder formatted the output into a matrix.
