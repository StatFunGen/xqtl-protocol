#!/usr/bin/env bash
# MWE: bulk_expression_normalization.ipynb :: normalize
# Tests: TMM-CPM-voom normalization to produce a BED.gz phenotype matrix
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/bulk_expression_normalization.ipynb normalize \
    --cwd $T/output \
    --counts-gct $T/AC.low_expr.count.gct.gz \
    --tpm-gct $T/AC.low_expr.tpm.gct.gz \
    --annotation-gtf $T/collapsed.gtf \
    --sample_participant_lookup $T/AC_sample_fastq.list \
    --count-threshold 6 \
    --sample-frac-threshold 0.2 \
    --normalization-method tmm_cpm_voom \
    --numThreads 4
