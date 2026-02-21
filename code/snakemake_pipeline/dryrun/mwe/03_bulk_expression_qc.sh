#!/usr/bin/env bash
# MWE: bulk_expression_QC.ipynb :: qc
# Tests: low-expression filter + RLE/D-statistic outlier removal
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/bulk_expression_QC.ipynb qc \
    --cwd $T/output \
    --tpm-gct $T/AC.rnaseqc.gene_tpm.gct.gz \
    --counts-gct $T/AC.rnaseqc.gene_reads.gct.gz \
    --low-expr-TPM 0.1 \
    --low-expr-TPM-percent 0.2 \
    --RLEFilterPercent 0.05 \
    --DSFilterPercent 0.05 \
    --numThreads 4
