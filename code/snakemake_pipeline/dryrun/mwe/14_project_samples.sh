#!/usr/bin/env bash
# MWE: PCA.ipynb :: project_samples
# Tests: project related samples onto unrelated PCA space
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/PCA.ipynb project_samples \
    --cwd $T/output \
    --genoFile $T/AC.related.plink_qc.extracted.bed \
    --pca-model $T/AC.pca.projected.rds \
    --maha-k 5 \
    --prob 0.997 \
    --numThreads 4 \
    -n
