#!/usr/bin/env bash
# MWE: GWAS_QC.ipynb :: qc_no_prune
# Tests: plink QC (MAF/missingness/HWE) without LD pruning
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/GWAS_QC.ipynb qc_no_prune \
    --cwd $T/output \
    --genoFile $T/xqtl.converted.bed \
    --name xqtl_protocol_data.plink_qc \
    --mac-filter 5 \
    --maf-filter 0 \
    --geno-filter 0.1 \
    --mind-filter 0.1 \
    --hwe-filter 1e-6 \
    --numThreads 4 \
    -n
