#!/usr/bin/env bash
# MWE: GWAS_QC.ipynb :: genotype_phenotype_sample_overlap
# Tests: identify overlapping samples between genotype and phenotype data
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run -n $PIPE/GWAS_QC.ipynb genotype_phenotype_sample_overlap \
    --cwd $T/output \
    --genoFile $T/xqtl.plink_qc.bed \
    --phenoFile $T/AC.tmm_cpm_voom.expression.bed.gz \
    --name xqtl_protocol_data.plink_qc.AC \
    --numThreads 4
