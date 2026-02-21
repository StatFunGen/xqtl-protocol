#!/usr/bin/env python3
"""
phenotype_formatting.py
Mirrors: code/data_preprocessing/phenotype/phenotype_formatting.ipynb

Steps (selected via --step):
  phenotype_by_chrom   — split BED.gz into per-chromosome files via tabix
  phenotype_by_region  — extract phenotypes by a region list via tabix

Flags are kept identical to the SoS notebook parameter names.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, print it, raise on failure."""
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def ensure_tabix_index(bed_gz: str) -> None:
    """Create a tabix .tbi index if one does not exist."""
    tbi = bed_gz + ".tbi"
    if not os.path.exists(tbi):
        run(f"tabix -p bed {bed_gz}")


def get_name(phenoFile: str, name: str) -> str:
    """Return output name prefix: use --name if provided, else strip .bed(.gz)."""
    if name:
        return name
    base = os.path.basename(phenoFile)
    for suffix in [".bed.gz", ".bed"]:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


# ---------------------------------------------------------------------------
# Step: phenotype_by_chrom
# ---------------------------------------------------------------------------

def phenotype_by_chrom(args) -> None:
    """
    Split a phenotype BED.gz file into one file per chromosome.

    Outputs:
      {cwd}/{name}.{chrom}.bed.gz           — per-chromosome BED
      {cwd}/{name}.{chrom}.bed.gz.tbi       — tabix index
      {cwd}/{name}.phenotype_by_chrom_files.txt — manifest of all chrom files
    """
    os.makedirs(args.cwd, exist_ok=True)
    ensure_tabix_index(args.phenoFile)

    name = get_name(args.phenoFile, args.name)
    manifest_lines = []

    for chrom in args.chrom:
        out_bed = os.path.join(args.cwd, f"{name}.{chrom}.bed.gz")
        run(f"tabix -h {args.phenoFile} {chrom} | bgzip -@ {args.numThreads} > {out_bed}")
        run(f"tabix -p bed {out_bed}")
        manifest_lines.append(out_bed)
        print(f"  Written: {out_bed}", flush=True)

    manifest = os.path.join(args.cwd, f"{name}.phenotype_by_chrom_files.txt")
    with open(manifest, "w") as fh:
        fh.write("\n".join(manifest_lines) + "\n")
    print(f"Manifest: {manifest}", flush=True)


# ---------------------------------------------------------------------------
# Step: phenotype_by_region
# ---------------------------------------------------------------------------

def phenotype_by_region(args) -> None:
    """
    Extract phenotypes for a set of named regions from a BED.gz file.

    region_list is a TSV with columns: chr, start, end, name
    Outputs one BED.gz per region and a manifest file.
    """
    os.makedirs(args.cwd, exist_ok=True)
    ensure_tabix_index(args.phenoFile)

    name = get_name(args.phenoFile, args.name)
    manifest_lines = []

    with open(args.region_list) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom, start, end, region_name = parts[0], parts[1], parts[2], parts[3]
            coord = f"{chrom}:{start}-{end}"
            out_bed = os.path.join(args.cwd, f"{name}.{region_name}.bed.gz")
            run(f"tabix -h {args.phenoFile} {coord} | bgzip -@ {args.numThreads} > {out_bed}")
            run(f"tabix -p bed {out_bed}")
            manifest_lines.append(out_bed)
            print(f"  Written: {out_bed}", flush=True)

    manifest = os.path.join(args.cwd, f"{name}.phenotype_by_region_files.txt")
    with open(manifest, "w") as fh:
        fh.write("\n".join(manifest_lines) + "\n")
    print(f"Manifest: {manifest}", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phenotype formatting utilities (mirrors phenotype_formatting.ipynb)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--step", required=True,
                   choices=["phenotype_by_chrom", "phenotype_by_region"],
                   help="Which step to run")
    p.add_argument("--phenoFile", required=True, metavar="PATH",
                   help="Input phenotype BED.gz file (tabix-indexed)")
    p.add_argument("--cwd", default="output", metavar="DIR",
                   help="Output directory")
    p.add_argument("--name", default="", metavar="STR",
                   help="Output name prefix (default: derived from phenoFile)")
    p.add_argument("--numThreads", type=int, default=20, metavar="N",
                   help="Number of threads for bgzip compression")

    # phenotype_by_chrom
    p.add_argument("--chrom", nargs="+", metavar="CHR",
                   help="[phenotype_by_chrom] Chromosomes to extract (e.g. chr1 chr2 ...)")

    # phenotype_by_region
    p.add_argument("--region-list", metavar="PATH",
                   help="[phenotype_by_region] TSV with chr/start/end/name columns")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.step == "phenotype_by_chrom":
        if not args.chrom:
            parser.error("--chrom is required for step phenotype_by_chrom")
        phenotype_by_chrom(args)

    elif args.step == "phenotype_by_region":
        if not args.region_list:
            parser.error("--region-list is required for step phenotype_by_region")
        phenotype_by_region(args)


if __name__ == "__main__":
    main()
