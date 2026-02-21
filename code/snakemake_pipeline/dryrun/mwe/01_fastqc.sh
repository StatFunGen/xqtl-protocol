#!/usr/bin/env bash
# MWE: RNA_calling.ipynb :: fastqc
# Tests: FastQC quality control on raw FASTQ files
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/RNA_calling.ipynb fastqc \
    --cwd $T/output \
    --sample-list $T/AC_sample_fastq.list \
    --data-dir $T/fastq_AC \
    --numThreads 4
