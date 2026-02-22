#!/usr/bin/env bash
# MWE: covariate_hidden_factor.ipynb :: PEER
# Tests: PEER factor estimation for hidden factor correction
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/covariate_hidden_factor.ipynb PEER \
    --cwd $T/output \
    --phenoFile $T/AC.tmm_cpm_voom.expression.bed.gz \
    --covFile $T/merged_cov.gz \
    --N 0 \
    --iteration 1000 \
    --convergence_mode fast \
    --numThreads 4 \
    -n
