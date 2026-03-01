#!/usr/bin/env python3
# ============================================================
# RNA_calling.py
# Mirrors: pipeline/RNA_calling.ipynb Python-specific steps
#
# Steps (selected via --step):
#   rnaseqc_merge  — merge per-sample RNA-SeQC GCT outputs and
#                    aggregate sample-level RNA-SeQC metrics
#
# Usage:
#   python3 RNA_calling.py --step rnaseqc_merge \
#       --cwd output/ \
#       --name sample_list \
#       --input s1.tpm.gct.gz s1.gc.gct.gz s1.ec.gct.gz s1.metrics.tsv \
#               s2.tpm.gct.gz ...
# ============================================================

import argparse
import os
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step",   required=True,
                   help="Step to run: rnaseqc_merge")
    p.add_argument("--cwd",    required=True,
                   help="Working/output directory")
    p.add_argument("--name",   required=True,
                   help="Output prefix (bam_list:bn in SoS notation)")
    p.add_argument("--input",  nargs="+", required=True,
                   help="Per-sample GCT and metrics files, interleaved as "
                        "tpm, gc, ec, metrics for each sample")
    return p.parse_args()


# ── GCT helpers ──────────────────────────────────────────────────────────────

def make_gct(gct_path):
    """Read a single-sample GCT file and return a one-column DataFrame."""
    sample_name = ".".join(os.path.basename(gct_path).split(".")[:-4])
    pre_gct = (
        pd.read_csv(gct_path, sep="\t", skiprows=2, index_col="Name")
        .drop("Description", axis=1)
    )
    pre_gct.index.name = "gene_ID"
    pre_gct.columns = [sample_name]
    return pre_gct


def merge_gct(gct_path_list):
    """Merge per-sample GCT files into one multi-sample DataFrame."""
    gct = pd.DataFrame()
    for gct_path in gct_path_list:
        gct_col = make_gct(gct_path)
        gct = gct.merge(gct_col, right_index=True, left_index=True, how="outer")
    return gct


# ── Step: rnaseqc_merge ───────────────────────────────────────────────────────

def run_rnaseqc_merge(args):
    """
    Merge per-sample RNA-SeQC outputs (tpm / reads / exon GCTs + metrics TSV).

    Expects input files interleaved as:
        sample1.gene_tpm.gct.gz
        sample1.gene_readsCount.gct.gz
        sample1.exon_readsCount.gct.gz
        sample1.metrics.tsv
        sample2.gene_tpm.gct.gz
        ...
    """
    prefix     = os.path.join(args.cwd, args.name)
    input_list = args.input

    tpm_list     = input_list[0::4]
    gc_list      = input_list[1::4]
    ec_list      = input_list[2::4]
    metrics_list = input_list[3::4]

    # ── Merge GCT files ───────────────────────────────────────────────────────
    gct_outputs = [
        (tpm_list, f"{prefix}.rnaseqc.gene_tpm.gct.gz"),
        (gc_list,  f"{prefix}.rnaseqc.gene_readsCount.gct.gz"),
        (ec_list,  f"{prefix}.rnaseqc.exon_readsCount.gct.gz"),
    ]
    for gct_list, out_path in gct_outputs:
        merged = merge_gct(gct_list)
        merged.to_csv(out_path, sep="\t")

    # ── Aggregate RNA-SeQC metrics ────────────────────────────────────────────
    metrics_list_file = f"{prefix}.rnaseqc.metrics_output_list"
    with open(metrics_list_file, "w") as fh:
        fh.write("\n".join(metrics_list))

    # Read the file-list back: index = path, value = NaN (no sample-ID column)
    path_s = (
        pd.read_csv(metrics_list_file, sep="\t", index_col=0,
                    header=None, names=["sample_id", "metrics_path"])
        .squeeze("columns")
    )
    if path_s.isnull().all():
        # No sample-ID column — derive IDs from file names
        path_s = pd.Series(
            path_s.index,
            index=[
                os.path.split(i)[1].split(".metrics.tsv")[0]
                for i in path_s.index
            ],
        )

    # Detect RNA-SeQC version from first file
    df0 = pd.read_csv(path_s.iloc[0], sep="\t", header=None)
    if df0.shape[0] == 2:          # RNA-SeQC v1.1.9 format
        dfs = [pd.read_csv(i, sep="\t") for i in path_s]
    elif df0.shape[1] == 2:        # RNA-SeQC v2 format (metric, value)
        dfs = [
            pd.read_csv(i, sep="\t", header=None, index_col=0).T
            for i in path_s
        ]
    else:
        raise ValueError(f"Unrecognized RNA-SeQC metrics format (shape {df0.shape}).")

    metrics_df = pd.concat(dfs, axis=0)
    metrics_df.index = metrics_df["Sample"]
    metrics_df.to_csv(
        f"{prefix}.rnaseqc.metrics.tsv",
        sep="\t", index=False, float_format="%.8g",
    )


# ── Dispatch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.step == "rnaseqc_merge":
        run_rnaseqc_merge(args)
    else:
        sys.exit(f"Unknown step: '{args.step}'. Valid steps: rnaseqc_merge")
