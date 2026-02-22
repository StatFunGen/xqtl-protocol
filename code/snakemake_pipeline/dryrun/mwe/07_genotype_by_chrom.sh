#!/usr/bin/env bash
# MWE: genotype_formatting.ipynb :: genotype_by_chrom
# Tests: split plink BED into per-chromosome files
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/genotype_formatting.ipynb genotype_by_chrom \
    --cwd $T/output \
    --genoFile $T/xqtl.plink_qc.bed \
    --name xqtl_protocol_data.plink_qc \
    --chrom chr1 chr2 chr22 \
    --numThreads 4 \
    -n
