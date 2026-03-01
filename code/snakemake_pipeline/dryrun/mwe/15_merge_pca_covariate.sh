#!/usr/bin/env bash
# MWE: covariate_formatting.ipynb :: merge_genotype_pc
# Tests: merge genotype PCs with phenotype covariates
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test
sos run $PIPE/covariate_formatting.ipynb merge_genotype_pc \
    --cwd $T/output \
    --pcaFile $T/AC.pca.projected.rds \
    --covFile $T/AC.covariates.cov.gz \
    --k 20 \
    --tol-cov 0.3 \
    --mean-impute \
    --numThreads 4 \
    -n
