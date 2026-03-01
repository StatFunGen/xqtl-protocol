#!/usr/bin/env bash
# MWE: phenotype_formatting.ipynb :: phenotype_by_chrom
# Tests: split BED phenotype file into per-chromosome files
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/phenotype_formatting.ipynb phenotype_by_chrom \
    --cwd $T/output \
    --phenoFile $T/AC.tmm_cpm_voom.expression.bed.gz \
    --name AC \
    --chrom chr1 chr2 chr22 \
    --numThreads 4 \
    -n
