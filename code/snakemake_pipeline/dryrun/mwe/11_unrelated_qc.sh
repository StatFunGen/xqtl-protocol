#!/usr/bin/env bash
# MWE: GWAS_QC.ipynb :: qc
# Tests: LD pruning + plink QC on unrelated samples only
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/GWAS_QC.ipynb qc \
    --cwd $T/output \
    --genoFile $T/AC.unrelated.bed \
    --mac-filter 5 \
    --window 200 \
    --shift 25 \
    --r2 0.1 \
    --numThreads 4 \
    -n
