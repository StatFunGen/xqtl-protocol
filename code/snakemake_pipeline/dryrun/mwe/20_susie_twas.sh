#!/usr/bin/env bash
# MWE: mnm_regression.ipynb :: susie_twas
# Tests: SuSiE univariate fine-mapping + TWAS weights
#
# DESIGN BUG NOTE: The smk susie_twas rule extracts per-chrom expression BED
# paths from phenotype_by_chrom_files.txt (column 2) and passes them as
# --phenoFile.  However, mnm_regression's get_analysis_regions step calls
# process_cis_files() which treats column 4 of each phenotype BED as a FILE
# PATH (pointing to per-region phenotype data), not an expression value.
# Passing standard expression BEDs (col4 = float expression value) causes:
#   TypeError: stat: path should be string, bytes, os.PathLike or integer, not float
#
# The smk rule must instead pass mnm-format BEDs where column 4 is a path to
# the actual per-region phenotype file.  These are created by a data-prep step
# that is currently missing from the Snakemake pipeline.
#
# This MWE uses correctly-formatted placeholder mnm BED files to demonstrate
# the expected interface.
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test

sos run $PIPE/mnm_regression.ipynb susie_twas \
    --cwd $T/output \
    --name AC \
    --genoFile $T/genotype_by_chrom_files.txt \
    --phenoFile $T/AC.chr1.mnm.bed.gz $T/AC.chr2.mnm.bed.gz $T/AC.chr22.mnm.bed.gz \
    --covFile $T/AC.Marchenko_PC.gz $T/AC.Marchenko_PC.gz $T/AC.Marchenko_PC.gz \
    --cis-window 500000 \
    --init-L 5 \
    --max-L 10 \
    --pip-cutoff 0.025 \
    --min_twas_maf 0.0025 \
    --numThreads 4 \
    -n
