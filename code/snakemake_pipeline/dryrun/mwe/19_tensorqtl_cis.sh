#!/usr/bin/env bash
# MWE: TensorQTL.ipynb :: cis
# Tests: cis-QTL mapping with TensorQTL
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/TensorQTL.ipynb cis \
    --cwd $T/output \
    --genotype-file $T/genotype_by_chrom_files.txt \
    --phenotype-file $T/phenotype_by_chrom_files.txt \
    --covariate-file $T/AC.Marchenko_PC.gz \
    --window 1000000 \
    --MAC 5 \
    --maf-threshold 0 \
    --numThreads 4
