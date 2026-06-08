#!/usr/bin/env python3
"""
Script-backed helpers for modular_sos genotype_formatting notebook steps that are
not pure shell wrappers.
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def split_list_arg(value: str, sep: str = "::") -> list[str]:
    if not value:
        return []
    return [item for item in value.split(sep) if item]


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def human_readable_size(path: str | Path) -> str:
    size = os.path.getsize(path)
    units = ("B", "K", "M", "G", "T", "P")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def open_text(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def summarize_vcf_gz_files(files: list[str]) -> None:
    for file in files:
        output_rows = 0
        output_column = 0
        output_header_row = 0
        preview: list[str] = []

        with open_text(file) as handle:
            for line in handle:
                output_rows += 1
                line = line.rstrip("\n")
                if line.startswith("##"):
                    output_header_row += 1
                    continue
                if output_column == 0:
                    output_column = len(line.split())
                if len(preview) < 10:
                    preview.append("\t".join(line.split("\t")[:11]))

        print(
            "output_info: {file}\n"
            "output_size: {size}\n"
            "output_rows: {rows}\n"
            "output_column: {cols}\n"
            "output_header_row: {headers}\n"
            "output_preview:\n{preview}".format(
                file=file,
                size=human_readable_size(file),
                rows=output_rows,
                cols=output_column,
                headers=output_header_row,
                preview="\n".join(preview),
            )
        )


def summarize_file_sizes(files: list[str]) -> None:
    for file in files:
        print(f"output_info: {file} ")
        print(f"output_size: {human_readable_size(file)}")


def summarize_table_file(file: str | Path) -> None:
    file = str(file)
    rows = 0
    output_column = 0
    preview: list[str] = []

    with open_text(file) as handle:
        for line in handle:
            rows += 1
            line = line.rstrip("\n")
            if output_column == 0:
                output_column = len(line.split())
            if not line.startswith("##") and len(preview) < 10:
                preview.append("\t".join(line.split("\t")[:6]))

    print(
        "output_info: {file}\n"
        "output_size: {size}\n"
        "output_rows: {rows}\n"
        "output_column: {cols}\n"
        "output_preview:\n{preview}".format(
            file=file,
            size=human_readable_size(file),
            rows=rows,
            cols=output_column,
            preview="\n".join(preview),
        )
    )


def summarize_ld_prefix(prefix: Path) -> None:
    ld_path = Path(f"{prefix}.ld")
    rows = 0
    cols = 0
    with open(ld_path) as handle:
        for line in handle:
            rows += 1
            if cols == 0:
                cols = len(line.split())

    print("The npz file is a numpy compressed version of the .ld file described below")
    print(f"output_info: {ld_path} ")
    print(f"output_size: {human_readable_size(ld_path)}")
    print(f"output_column: {cols}")
    print(f"output_row: {rows}")


def ld_by_region_plink_1(args: argparse.Namespace) -> None:
    if not args.genoFile:
      raise ValueError("--genoFile is required")
    if not args.output:
      raise ValueError("--output is required")
    for name in ("region_chrom", "region_start", "region_end"):
      if getattr(args, name) in (None, ""):
        raise ValueError(f"--{name.replace('_', '-')} is required")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_prefix = output.with_suffix("").with_suffix("")

    geno_prefix = str(Path(args.genoFile).with_suffix(""))
    run_cmd([
        "plink",
        "--bfile", geno_prefix,
        "--out", str(out_prefix),
        "--chr", str(args.region_chrom),
        "--from-bp", str(args.region_start),
        "--to-bp", str(args.region_end),
        "--r", "square0",
        "--make-just-bim",
        "--threads", str(args.numThreads),
    ])

    ld_path = f"{out_prefix}.ld"
    bim_path = f"{out_prefix}.bim"
    np_ld = np.loadtxt(ld_path, delimiter="\t", dtype=f"float{args.float_type}")
    bim = pd.read_csv(bim_path, sep="\t", header=None)[1].to_numpy()
    np.savez_compressed(output, np_ld, bim, allow_pickle=True)
    summarize_ld_prefix(out_prefix)


def write_data_list(args: argparse.Namespace) -> None:
    if not args.output:
      raise ValueError("--output is required")
    files = split_list_arg(args.data_files)
    if not files:
      raise ValueError("--data-files is required")

    n = len(args.ext.split(".")) + 1
    non_empty_files: list[str] = []
    non_empty_ids: list[str] = []

    for file in files:
        file_id = file.split(".")[-n]
        try:
            if os.path.getsize(file) > 0:
                non_empty_files.append(file)
                non_empty_ids.append(file_id)
            else:
                print(f"Empty file found: {file}", file=sys.stderr)
        except OSError as e:
            print(f"Error accessing file {file}: {e}", file=sys.stderr)

    if not non_empty_files:
        raise ValueError("No non-empty files found. Exiting.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "#id": non_empty_ids,
        "#path": non_empty_files,
    }).to_csv(output, index=False, sep="\t")
    summarize_table_file(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)
    parser.add_argument("--genoFile", default="")
    parser.add_argument("--region-chrom", dest="region_chrom", default="")
    parser.add_argument("--region-start", dest="region_start", default="")
    parser.add_argument("--region-end", dest="region_end", default="")
    parser.add_argument("--float-type", dest="float_type", type=int, default=16)
    parser.add_argument("--output", default="")
    parser.add_argument("--data-files", default="")
    parser.add_argument("--ext", default="")
    parser.add_argument("--numThreads", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.step == "ld_by_region_plink_1":
        ld_by_region_plink_1(args)
    elif args.step == "write_data_list":
        write_data_list(args)
    elif args.step == "vcf_gz_summary":
        files = split_list_arg(args.data_files)
        if not files and args.output:
            files = [args.output]
        if not files:
            raise ValueError("--data-files or --output is required")
        summarize_vcf_gz_files(files)
    elif args.step == "file_size_summary":
        files = split_list_arg(args.data_files)
        if not files and args.output:
            files = [args.output]
        if not files:
            raise ValueError("--data-files or --output is required")
        summarize_file_sizes(files)
    else:
        raise ValueError(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()
