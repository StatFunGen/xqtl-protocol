#!/usr/bin/env bash
# MWE: GWAS_QC.ipynb :: king
# Tests: KING kinship estimation and related/unrelated sample split
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/GWAS_QC.ipynb king \
    --cwd $T/output \
    --genoFile $T/xqtl.plink_qc.bed \
    --keep-samples $T/AC.sample_genotypes.txt \
    --name xqtl_protocol_data.plink_qc.AC \
    --numThreads 4
