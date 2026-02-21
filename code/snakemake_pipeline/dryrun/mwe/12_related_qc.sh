#!/usr/bin/env bash
# MWE: GWAS_QC.ipynb :: qc_no_prune
# Tests: extract pruned SNP set for related samples (no LD pruning)
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/GWAS_QC.ipynb qc_no_prune \
    --cwd $T/output \
    --genoFile $T/AC.related.bed \
    --keep-variants $T/AC.unrelated.plink_qc.prune.in \
    --mac-filter 5 \
    --numThreads 4
