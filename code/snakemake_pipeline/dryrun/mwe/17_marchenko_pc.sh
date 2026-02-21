#!/usr/bin/env bash
# MWE: covariate_hidden_factor.ipynb :: Marchenko_PC
# Tests: Marchenko-Pastur law-based PC selection for hidden factor correction
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/covariate_hidden_factor.ipynb Marchenko_PC \
    --cwd $T/output \
    --phenoFile $T/AC.tmm_cpm_voom.expression.bed.gz \
    --covFile $T/merged_cov.gz \
    --N 0 \
    --numThreads 4
