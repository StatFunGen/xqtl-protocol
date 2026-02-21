#!/usr/bin/env bash
# MWE: rss_analysis.ipynb :: univariate_plot
#
# NOTE — SoS _input limitation:
#   This step takes its input via SoS pipeline chaining (_input = previous
#   step's output RDS file).  SoS does not expose a command-line flag to
#   provide _input for a standalone step, so it cannot be dry-run in
#   isolation.  In the Snakemake pipeline, `finemapping_plots` drives it
#   with a find loop over the susie_twas output directory:
#
#     find finemapping_dir -name "*.rds" | sort | while IFS= read -r rds; do
#         sos run [-n] rss_analysis.ipynb univariate_plot "$rds" --cwd ...
#     done
#
#   That shell syntax is correct for SoS; the limitation here is only in
#   standalone dry-run testing where we lack real upstream SuSiE output.
#
# What we can validate here: parameter acceptance (non-_input params only).
PIPE=/home/user/xqtl-protocol/pipeline
T=/tmp/xqtl_test

echo "=== Checking rss_analysis.ipynb univariate_plot (non-_input params) ==="
sos run -n $PIPE/rss_analysis.ipynb univariate_plot \
    --cwd $T/output \
    --numThreads 4 2>&1 || true
echo
echo "=== Snakemake shell block (for reference) ==="
echo "find {finemapping_dir} -name '*.rds' | sort | while IFS= read -r rds; do"
echo "    sos run [-n] rss_analysis.ipynb univariate_plot \"\$rds\" --cwd {outdir} ..."
echo "done"
