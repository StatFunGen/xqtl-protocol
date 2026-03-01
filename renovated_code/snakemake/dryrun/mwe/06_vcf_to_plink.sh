#!/usr/bin/env bash
# MWE: genotype_formatting.ipynb :: vcf_to_plink
# Tests: VCF → plink BED conversion
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/genotype_formatting.ipynb vcf_to_plink \
    --cwd $T/output \
    --genoFile $T/genotype.vcf.gz \
    --name xqtl_protocol_data.converted \
    --numThreads 4 \
    -n
