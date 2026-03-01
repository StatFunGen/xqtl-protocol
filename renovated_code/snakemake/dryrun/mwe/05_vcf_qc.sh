#!/usr/bin/env bash
# MWE: VCF_QC.ipynb :: qc
# Tests: bcftools left-norm, biallelic split, dbSNP annotation, variant-level QC
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/VCF_QC.ipynb qc \
    --cwd $T/output \
    --genoFile $T/genotype.vcf.gz \
    --dbsnp-variants $T/dbsnp.vcf.gz \
    --reference-genome $T/ref.fasta \
    --numThreads 4 \
    -n
