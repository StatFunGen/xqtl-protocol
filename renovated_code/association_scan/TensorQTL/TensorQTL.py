#!/usr/bin/env python3
"""
TensorQTL.py
Mirrors: code/association_scan/TensorQTL/TensorQTL.ipynb

Steps (selected via --step):
  cis   — cis-QTL: nominal pass + permutation test across all chromosomes
  trans — trans-QTL: genome-wide association scan

Flags are kept identical to the SoS notebook parameter names.
"""

import argparse
import os
import sys
import glob
import gzip
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np


def read_file_list(path: str) -> list:
    """Read a one-file-per-line manifest, return list of paths."""
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def load_covariates(cov_file: str) -> pd.DataFrame:
    """
    Load covariate file.
    Expected format: rows = covariates/factors, cols = samples.
    First column is the covariate name/ID.
    Returns DataFrame: samples × covariates.
    """
    df = pd.read_csv(cov_file, sep="\t", index_col=0,
                     compression="gzip" if cov_file.endswith(".gz") else None)
    return df.T   # transpose to samples × covariates


def load_phenotype_bed(bed_gz: str):
    """
    Load a phenotype BED.gz file.
    Returns (phenotype_df, phenotype_pos_df).
    phenotype_df: genes × samples
    phenotype_pos_df: DataFrame with columns chr, tss
    """
    df = pd.read_csv(bed_gz, sep="\t", index_col=3,
                     compression="gzip" if bed_gz.endswith(".gz") else None)
    pos_df = df.iloc[:, :3].copy()
    pos_df.columns = ["chr", "start", "end"]
    pos_df["tss"] = pos_df["start"]
    pheno_df = df.iloc[:, 4:].astype(float)
    return pheno_df, pos_df[["chr", "tss"]]


def run_cis(args) -> None:
    """
    Run cis-QTL analysis (nominal + permutation) per chromosome.
    Requires TensorQTL Python package.
    """
    try:
        import tensorqtl
        from tensorqtl import cis, post
    except ImportError:
        sys.exit("ERROR: tensorqtl package not installed. "
                 "Install via: pip install tensorqtl")

    os.makedirs(args.cwd, exist_ok=True)

    geno_list  = read_file_list(args.genotype_file)
    pheno_list = read_file_list(args.phenotype_file)

    covariates_df = load_covariates(args.covariate_file)

    # Map genotype files to chromosome
    geno_by_chrom  = {Path(f).stem.split(".")[-1]: f for f in geno_list}
    pheno_by_chrom = {}
    for f in pheno_list:
        # Phenotype files named: {name}.{chrom}.bed.gz
        stem = Path(f).stem.replace(".bed", "")
        chrom = stem.split(".")[-1]
        pheno_by_chrom[chrom] = f

    chroms = sorted(set(geno_by_chrom) & set(pheno_by_chrom))
    if not chroms:
        sys.exit("ERROR: No matching chromosomes between genotype and phenotype lists.")
    print(f"Running cis-QTL on {len(chroms)} chromosomes: {chroms}", flush=True)

    nominal_results  = []
    permutation_results = []

    for chrom in chroms:
        print(f"\n=== {chrom} ===", flush=True)
        geno_prefix = geno_by_chrom[chrom].replace(".bed", "")
        pheno_file  = pheno_by_chrom[chrom]

        pr = tensorqtl.genotypeio.PlinkReader(geno_prefix)
        pheno_df, pheno_pos_df = load_phenotype_bed(pheno_file)

        # Align samples
        shared = pr.fam.index.intersection(pheno_df.columns).intersection(covariates_df.index)
        pheno_df     = pheno_df[shared]
        covariates_t = covariates_df.loc[shared]

        # MAC/MAF filter
        genotype_df, variant_df = pr.get_all_genotypes(return_df=True)
        genotype_df  = genotype_df[shared]
        mac          = (genotype_df.sum(axis=1).clip(0, 2 * len(shared) - genotype_df.sum(axis=1).clip(0)).min(axis=0))

        # Nominal pass
        nominal_df = cis.map_nominal(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            covariates_df=covariates_t,
            window=args.window,
            maf_threshold=args.maf_threshold,
        )
        nominal_out = os.path.join(args.cwd, f"{chrom}.cis_qtl_pairs.parquet")
        nominal_df.to_parquet(nominal_out)
        nominal_results.append(nominal_out)
        print(f"  Nominal: {len(nominal_df)} pairs → {nominal_out}", flush=True)

        # Permutation pass
        perm_df = cis.map_cis(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            covariates_df=covariates_t,
            window=args.window,
            maf_threshold=args.maf_threshold,
            nperm=1000,
        )
        perm_out = os.path.join(args.cwd, f"{chrom}.cis_qtl.txt.gz")
        perm_df.to_csv(perm_out, sep="\t", index=True, compression="gzip")
        permutation_results.append(perm_out)
        print(f"  Permutation: {len(perm_df)} phenotypes → {perm_out}", flush=True)

    # Aggregate permutation results and compute q-values
    print("\n=== Aggregating permutation results ===", flush=True)
    all_perm = pd.concat([pd.read_csv(f, sep="\t", index_col=0)
                          for f in permutation_results])
    all_perm_out = os.path.join(args.cwd, "all_chroms.cis_qtl.txt.gz")
    all_perm.to_csv(all_perm_out, sep="\t", index=True, compression="gzip")

    # BH correction for q-values
    try:
        from tensorqtl import post as tpost
        sig_df = tpost.calculate_qvalues(all_perm, qvalue_cutoff=0.05)
        sig_out = os.path.join(args.cwd, "all_chroms.cis_qtl.signif_pairs.txt.gz")
        sig_df.to_csv(sig_out, sep="\t", index=True, compression="gzip")
        print(f"Significant pairs: {sig_out}", flush=True)
    except Exception as e:
        print(f"WARNING: Could not compute q-values: {e}", flush=True)

    print(f"\nCIS QTL complete. Results in: {args.cwd}", flush=True)


def run_trans(args) -> None:
    """Run trans-QTL genome-wide association scan."""
    try:
        import tensorqtl
        from tensorqtl import trans
    except ImportError:
        sys.exit("ERROR: tensorqtl package not installed.")

    os.makedirs(args.cwd, exist_ok=True)
    geno_list  = read_file_list(args.genotype_file)
    pheno_list = read_file_list(args.phenotype_file)
    covariates_df = load_covariates(args.covariate_file)

    for geno_f, pheno_f in zip(sorted(geno_list), sorted(pheno_list)):
        chrom = Path(geno_f).stem.split(".")[-1]
        pr = tensorqtl.genotypeio.PlinkReader(geno_f.replace(".bed", ""))
        pheno_df, pheno_pos_df = load_phenotype_bed(pheno_f)
        shared = pr.fam.index.intersection(pheno_df.columns).intersection(covariates_df.index)
        genotype_df, variant_df = pr.get_all_genotypes(return_df=True)
        genotype_df = genotype_df[shared]

        trans_df = trans.map_trans(
            genotype_df, variant_df, pheno_df[shared],
            covariates_df=covariates_df.loc[shared],
            maf_threshold=args.maf_threshold,
        )
        out = os.path.join(args.cwd, f"{chrom}.trans_qtl_pairs.parquet")
        trans_df.to_parquet(out)
        print(f"  Trans {chrom}: {len(trans_df)} pairs → {out}", flush=True)

    print(f"\nTRANS QTL complete. Results in: {args.cwd}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TensorQTL wrapper (mirrors TensorQTL.ipynb)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--step", required=True, choices=["cis", "trans"],
                   help="Which step to run")
    p.add_argument("--genotype-file", required=True, metavar="PATH",
                   help="Path to genotype_by_chrom_files.txt manifest")
    p.add_argument("--phenotype-file", required=True, metavar="PATH",
                   help="Path to phenotype_by_chrom_files.txt manifest")
    p.add_argument("--covariate-file", required=True, metavar="PATH",
                   help="Hidden factor covariate file (gzip TSV, covariates × samples)")
    p.add_argument("--cwd", default="output", metavar="DIR",
                   help="Output directory")
    p.add_argument("--window", type=int, default=1_000_000, metavar="BP",
                   help="CIS window in bp")
    p.add_argument("--MAC", type=int, default=5, metavar="N",
                   help="Minimum minor allele count filter")
    p.add_argument("--maf-threshold", type=float, default=0.0, metavar="F",
                   help="Minimum MAF filter (0 = no filter)")
    p.add_argument("--numThreads", type=int, default=8)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.step == "cis":
        run_cis(args)
    elif args.step == "trans":
        run_trans(args)


if __name__ == "__main__":
    main()
