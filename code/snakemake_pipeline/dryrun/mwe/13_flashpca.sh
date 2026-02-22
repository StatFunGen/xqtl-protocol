#!/usr/bin/env bash
# MWE: PCA.ipynb :: flashpca
# Tests: FlashPCA on LD-pruned unrelated genotype data
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/PCA.ipynb flashpca \
    --cwd $T/output \
    --genoFile $T/AC.unrelated.plink_qc.prune.bed \
    --k 20 \
    --maha-k 5 \
    --prob 0.997 \
    --numThreads 4 \
    -n
