#!/usr/bin/env python3
# ============================================================
# RNA_calling.py
# Mirrors selected Python-heavy steps from RNA_calling.ipynb.
#
# Steps:
#   rnaseqc_merge
#   star_align_3
#   rsem_call_2
# ============================================================

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["rnaseqc_merge", "star_align_3", "rsem_call_2"],
        help="Step to run",
    )
    parser.add_argument("--cwd", help="Working/output directory")
    parser.add_argument("--name", help="Output prefix name")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Interleaved input files for the selected step",
    )
    parser.add_argument("--output", help="Single output file path")
    parser.add_argument(
        "--sample-id",
        nargs="+",
        default=[],
        help="Sample IDs for STAR_align_3",
    )
    parser.add_argument(
        "--strand",
        nargs="+",
        default=[],
        help="Per-sample strand values for STAR_align_3",
    )
    return parser.parse_args()


def make_gct(gct_path):
    sample_name = ".".join(os.path.basename(gct_path).split(".")[:-4])
    pre_gct = (
        pd.read_csv(gct_path, sep="\t", skiprows=2, index_col="Name")
        .drop("Description", axis=1)
    )
    pre_gct.index.name = "gene_ID"
    pre_gct.columns = [sample_name]
    return pre_gct


def merge_gct(gct_path_list):
    gct = pd.DataFrame()
    for gct_path in gct_path_list:
        gct_col = make_gct(gct_path)
        if gct.empty:
            gct = gct_col
        else:
            gct = gct.merge(gct_col, right_index=True, left_index=True, how="outer")
    return gct


def run_rnaseqc_merge(args):
    if not args.cwd or not args.name or not args.input:
        raise SystemExit("rnaseqc_merge requires --cwd, --name, and --input")

    prefix = os.path.join(args.cwd, args.name)
    input_list = args.input

    tpm_list = input_list[0::4]
    gc_list = input_list[1::4]
    ec_list = input_list[2::4]
    metrics_list = input_list[3::4]

    gct_outputs = [
        (tpm_list, f"{prefix}.rnaseqc.gene_tpm.gct.gz"),
        (gc_list, f"{prefix}.rnaseqc.gene_readsCount.gct.gz"),
        (ec_list, f"{prefix}.rnaseqc.exon_readsCount.gct.gz"),
    ]
    for gct_list, out_path in gct_outputs:
        merged = merge_gct(gct_list)
        merged.to_csv(out_path, sep="\t")

    metrics_list_file = f"{prefix}.rnaseqc.metrics_output_list"
    with open(metrics_list_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(metrics_list))

    path_s = (
        pd.read_csv(
            metrics_list_file,
            sep="\t",
            index_col=0,
            header=None,
            names=["sample_id", "metrics_path"],
        )
        .squeeze("columns")
    )
    if path_s.isnull().all():
        path_s = pd.Series(
            path_s.index,
            index=[
                os.path.split(i)[1].split(".metrics.tsv")[0]
                for i in path_s.index
            ],
        )

    df0 = pd.read_csv(path_s.iloc[0], sep="\t", header=None)
    if df0.shape[0] == 2:
        dfs = [pd.read_csv(i, sep="\t") for i in path_s]
    elif df0.shape[1] == 2:
        dfs = [
            pd.read_csv(i, sep="\t", header=None, index_col=0).T
            for i in path_s
        ]
    else:
        raise ValueError(
            f"Unrecognized RNA-SeQC metrics format (shape {df0.shape})."
        )

    metrics_df = pd.concat(dfs, axis=0)
    metrics_df.index = metrics_df["Sample"]
    metrics_df.to_csv(
        f"{prefix}.rnaseqc.metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.8g",
    )


def strip_extensions(path_like, count):
    value = Path(path_like).name
    for _ in range(count):
        if "." not in value:
            break
        value = value.rsplit(".", 1)[0]
    return value


def run_star_align_3(args):
    if not args.output or not args.input:
        raise SystemExit("star_align_3 requires --output and --input")
    if not args.sample_id or not args.strand:
        raise SystemExit("star_align_3 requires --sample-id and --strand")
    if len(args.sample_id) != len(args.strand):
        raise SystemExit("--sample-id and --strand must have the same length")
    if len(args.input) % 6 != 0:
        raise SystemExit("star_align_3 expects inputs in groups of 6")

    coord_bam_list = [Path(x).name for x in args.input[2::6]]
    sorted_bigwig_list = [Path(x).name for x in args.input[4::6]]
    sj_list = [f"{strip_extensions(x, 5)}.SJ.out.tab" for x in coord_bam_list]
    trans_bam_list = [
        f"{strip_extensions(x, 4)}.toTranscriptome.out.bam"
        for x in coord_bam_list
    ]

    out = pd.DataFrame(
        {
            "sample_id": args.sample_id,
            "strand": args.strand,
            "coord_bam_list": coord_bam_list,
            "BW_list": sorted_bigwig_list,
            "SJ_list": sj_list,
            "trans_bam_list": trans_bam_list,
        }
    )
    out.to_csv(args.output, sep="\t", index=False)


def derive_sample_name(path_str):
    name = Path(path_str).name
    for suffix in [".rsem.isoforms.results", ".rsem.genes.results", ".rsem.cnt"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def merge_rsem_metric(input_files, metric_name, output_path):
    first = pd.read_csv(input_files[0], sep="\t")
    if metric_name not in first.columns:
        raise SystemExit(f"{metric_name} not found in {input_files[0]}")

    id_col = first.columns[0]
    extra_cols = [col for col in first.columns[:2] if col != metric_name]
    base = first[extra_cols].copy()

    merged = base
    for path in input_files:
        frame = pd.read_csv(path, sep="\t")
        sample_name = derive_sample_name(path)
        sample_frame = frame[[id_col, metric_name]].copy()
        sample_frame.columns = [id_col, sample_name]
        merged = merged.merge(sample_frame, on=id_col, how="outer")

    merged.to_csv(output_path, sep="\t", index=False, compression="gzip")


def read_rsem_cnt(source_paths):
    files = list(source_paths)
    if not files:
        raise SystemExit("No RSEM count files provided")

    metrics = []
    for path in files:
        frame = pd.read_csv(
            path,
            sep=" ",
            header=None,
            comment="#",
            nrows=3,
            engine="python",
        )
        metrics.append(
            {
                "Sample": derive_sample_name(path),
                "File": path,
                "TotalReads": frame.iloc[0, 3],
                "AlignedReads": frame.iloc[0, 1],
                "UniquelyAlignedReads": frame.iloc[1, 0],
            }
        )
    return pd.DataFrame(metrics)


def run_rsem_call_2(args):
    if not args.cwd or not args.name or not args.input:
        raise SystemExit("rsem_call_2 requires --cwd, --name, and --input")

    Path(args.cwd).mkdir(parents=True, exist_ok=True)
    prefix = Path(args.cwd) / args.name
    input_list = args.input

    isoform_files = input_list[0::3]
    gene_files = input_list[1::3]
    cnt_files = input_list[2::3]

    iso_list = Path(f"{prefix}.rsem.isoforms_output_list")
    gene_list = Path(f"{prefix}.rsem.genes_output_list")
    iso_list.write_text("\n".join(isoform_files), encoding="utf-8")
    gene_list.write_text("\n".join(gene_files), encoding="utf-8")

    merge_rsem_metric(
        isoform_files,
        "expected_count",
        f"{prefix}.rsem_transcripts_expected_count.txt.gz",
    )
    merge_rsem_metric(
        isoform_files,
        "TPM",
        f"{prefix}.rsem_transcripts_tpm.txt.gz",
    )
    merge_rsem_metric(
        isoform_files,
        "FPKM",
        f"{prefix}.rsem_transcripts_fpkm.txt.gz",
    )
    merge_rsem_metric(
        isoform_files,
        "IsoPct",
        f"{prefix}.rsem_transcripts_isopct.txt.gz",
    )
    merge_rsem_metric(
        gene_files,
        "expected_count",
        f"{prefix}.rsem_genes_expected_count.txt.gz",
    )
    merge_rsem_metric(
        gene_files,
        "TPM",
        f"{prefix}.rsem_genes_tpm.txt.gz",
    )
    merge_rsem_metric(
        gene_files,
        "FPKM",
        f"{prefix}.rsem_genes_fpkm.txt.gz",
    )

    metrics = read_rsem_cnt(cnt_files)
    metrics.to_csv(
        f"{prefix}.rsem.aggregated_quality.metrics.tsv",
        sep="\t",
        index=False,
    )


def main():
    args = parse_args()
    if args.step == "rnaseqc_merge":
        run_rnaseqc_merge(args)
    elif args.step == "star_align_3":
        run_star_align_3(args)
    elif args.step == "rsem_call_2":
        run_rsem_call_2(args)
    else:
        sys.exit(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()
