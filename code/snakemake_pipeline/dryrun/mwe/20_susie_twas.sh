#!/usr/bin/env bash
# MWE: mnm_regression.ipynb :: susie_twas
# Tests: SuSiE univariate fine-mapping + TWAS weights
#
# Note: mnm_regression phenoFile expects the actual per-chrom BED files,
# NOT the file-listing produced by phenotype_by_chrom.  We extract the
# BED paths from column 2 of the file listing.
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test

PHENO_BEDS=$(awk 'NR>1 {print $2}' $T/phenotype_by_chrom_files.txt | tr '\n' ' ')

sos run $PIPE/mnm_regression.ipynb susie_twas \
    --cwd $T/output \
    --name AC \
    --genoFile $T/genotype_by_chrom_files.txt \
    --phenoFile $PHENO_BEDS \
    --covFile $T/AC.Marchenko_PC.gz \
    --init-L 5 \
    --max-L 10 \
    --pip-cutoff 0.025 \
    --min_twas_maf 0.0025 \
    --numThreads 4 \
    -n
