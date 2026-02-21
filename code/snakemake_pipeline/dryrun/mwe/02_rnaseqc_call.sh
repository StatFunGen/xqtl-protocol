#!/usr/bin/env bash
# MWE: RNA_calling.ipynb :: rnaseqc_call
# Tests: STAR alignment + RNASeQC quantification pipeline
#
# DESIGN GAP NOTE: The Snakemake rnaseqc_call rule only passes --sample-list.
# However, sub-steps rnaseqc_call_1/2/3/4 also require --bam-list (the STAR
# output BAM manifest).  A dedicated STAR_align Snakemake rule is needed
# upstream to produce AC_bam_list.txt before rnaseqc_call can run fully.
# Until that rule exists, the sos call needs both --sample-list AND --bam-list.
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/RNA_calling.ipynb rnaseqc_call \
    --cwd $T/output \
    --sample-list $T/AC_sample_fastq.list \
    --data-dir $T/fastq_AC \
    --bam-list $T/AC_bam_list.txt \
    --gtf $T/ERCC.gtf \
    --reference-fasta $T/ref.fasta \
    --numThreads 4
