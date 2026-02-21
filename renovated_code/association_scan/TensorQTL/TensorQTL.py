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
from pathlib import Path

import numpy as np
import pandas as pd


def read_file_list(path: str) -> list:
    """Read a one-file-per-line manifest, return list of paths."""
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def load_covariates(cov_file: str) -> pd.DataFrame:
    """
    Load covariate file.
    Expected format: rows = covariates/factors, cols = samples.
    First column is the covariate name/ID.
    Returns DataFrame: samples × covariates (transposed for TensorQTL).
    """
    df = pd.read_csv(cov_file, sep="\t", index_col=0,
                     compression="gzip" if cov_file.endswith(".gz") else None)
    return df.T   # transpose to samples × covariates


def load_phenotype_bed(bed_gz: str):
    """
    Load a phenotype BED.gz file.
    Returns (phenotype_df, phenotype_pos_df) matching TensorQTL expected format:
      phenotype_df:     DataFrame indexed by phenotype_id, cols = sample_ids
      phenotype_pos_df: DataFrame indexed by phenotype_id, cols = ['chr', 'tss']
    """
    df = pd.read_csv(bed_gz, sep="\t", index_col=3,
                     compression="gzip" if bed_gz.endswith(".gz") else None)
    pos_df = df.iloc[:, :3].copy()
    pos_df.columns = ["chr", "start", "end"]
    pos_df["tss"] = pos_df["start"]
    pheno_df = df.iloc[:, 3:].astype(float)
    return pheno_df, pos_df[["chr", "tss"]]


def apply_mac_filter(genotype_df: pd.DataFrame, variant_df: pd.DataFrame,
                     n_samples: int, mac_min: int):
    """Return genotype_df and variant_df with variants below mac_min removed."""
    if mac_min <= 0:
        return genotype_df, variant_df
    ac = genotype_df.sum(axis=1)
    mac = np.minimum(ac, 2 * n_samples - ac)
    keep = mac >= mac_min
    return genotype_df[keep], variant_df[keep]


def run_cis(args) -> None:
    """
    Run cis-QTL analysis (nominal + permutation) per chromosome.
    Requires TensorQTL Python package.
    """
    try:
        from tensorqtl import genotypeio, cis, post
    except ImportError:
        sys.exit("ERROR: tensorqtl package not installed. "
                 "Install via: pip install tensorqtl")

    os.makedirs(args.cwd, exist_ok=True)

    geno_list  = read_file_list(args.genotype_file)
    pheno_list = read_file_list(args.phenotype_file)

    covariates_df = load_covariates(args.covariate_file)

    # Map files to chromosome by parsing the filename stem
    geno_by_chrom: dict = {}
    for f in geno_list:
        # strip .bed; the last dot-separated token is the chromosome
        stem = Path(f).stem  # e.g. xqtl_protocol_data.plink_qc.chr1
        chrom = stem.split(".")[-1]
        geno_by_chrom[chrom] = f.replace(".bed", "")  # plink prefix without .bed

    pheno_by_chrom: dict = {}
    for f in pheno_list:
        stem = Path(f).stem.replace(".bed", "")  # strip .bed.gz → .bed → stem
        chrom = stem.split(".")[-1]
        pheno_by_chrom[chrom] = f

    chroms = sorted(set(geno_by_chrom) & set(pheno_by_chrom))
    if not chroms:
        sys.exit("ERROR: No matching chromosomes between genotype and phenotype lists.")
    print(f"Running cis-QTL on {len(chroms)} chromosomes: {chroms}", flush=True)

    permutation_results = []

    for chrom in chroms:
        print(f"\n=== {chrom} ===", flush=True)
        geno_prefix = geno_by_chrom[chrom]
        pheno_file  = pheno_by_chrom[chrom]

        # Load genotype using the correct TensorQTL API
        genotype_df, variant_df = genotypeio.load_genotypes(geno_prefix, dosages=True)
        pheno_df, pheno_pos_df = load_phenotype_bed(pheno_file)

        # Align samples across genotype, phenotype, and covariates
        shared = (genotype_df.columns
                  .intersection(pheno_df.columns)
                  .intersection(covariates_df.index))
        shared = list(shared)
        if not shared:
            print(f"  WARNING: No shared samples for {chrom}, skipping.", flush=True)
            continue

        pheno_df     = pheno_df[shared]
        covariates_t = covariates_df.loc[shared]
        genotype_df  = genotype_df[shared]

        # Apply MAC filter
        genotype_df, variant_df = apply_mac_filter(
            genotype_df, variant_df, n_samples=len(shared), mac_min=args.MAC)

        # Nominal pass — writes {cwd}/{chrom}.cis_qtl_pairs.{chr}.parquet
        # map_nominal writes output to disk; it does not return a DataFrame
        nominal_prefix = chrom
        cis.map_nominal(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            nominal_prefix,
            covariates_df=covariates_t,
            window=args.window,
            maf_threshold=args.maf_threshold,
            run_eigenmt=True,
            output_dir=args.cwd,
        )
        print(f"  Nominal pass complete → {args.cwd}/{nominal_prefix}.cis_qtl_pairs.*.parquet",
              flush=True)

        # Permutation pass — returns a DataFrame
        perm_df = cis.map_cis(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            covariates_df=covariates_t,
            window=args.window,
            maf_threshold=args.maf_threshold,
            seed=999,
        )
        perm_out = os.path.join(args.cwd, f"{chrom}.cis_qtl.txt.gz")
        perm_df.to_csv(perm_out, sep="\t", index=True, compression="gzip")
        permutation_results.append(perm_out)
        print(f"  Permutation: {len(perm_df)} phenotypes → {perm_out}", flush=True)

    if not permutation_results:
        print("No permutation results produced.", flush=True)
        return

    # Aggregate permutation results across chromosomes
    print("\n=== Aggregating permutation results ===", flush=True)
    all_perm = pd.concat([pd.read_csv(f, sep="\t", index_col=0)
                          for f in permutation_results])
    all_perm_out = os.path.join(args.cwd, "all_chroms.cis_qtl.txt.gz")
    all_perm.to_csv(all_perm_out, sep="\t", index=True, compression="gzip")
    print(f"All permutation results: {all_perm_out}", flush=True)

    # Compute q-values using tensorqtl.post
    try:
        sig_df = post.calculate_qvalues(all_perm, qvalue_cutoff=0.05)
        sig_out = os.path.join(args.cwd, "all_chroms.cis_qtl.signif_pairs.txt.gz")
        sig_df.to_csv(sig_out, sep="\t", index=True, compression="gzip")
        print(f"Significant pairs: {sig_out}", flush=True)
    except Exception as e:
        print(f"WARNING: Could not compute q-values: {e}", flush=True)

    print(f"\nCIS QTL complete. Results in: {args.cwd}", flush=True)


def run_trans(args) -> None:
    """Run trans-QTL genome-wide association scan."""
    try:
        from tensorqtl import genotypeio, trans
    except ImportError:
        sys.exit("ERROR: tensorqtl package not installed.")

    os.makedirs(args.cwd, exist_ok=True)
    geno_list     = read_file_list(args.genotype_file)
    pheno_list    = read_file_list(args.phenotype_file)
    covariates_df = load_covariates(args.covariate_file)

    for geno_f, pheno_f in zip(sorted(geno_list), sorted(pheno_list)):
        chrom      = Path(geno_f).stem.split(".")[-1]
        geno_prefix = geno_f.replace(".bed", "")

        genotype_df, variant_df = genotypeio.load_genotypes(geno_prefix, dosages=True)
        pheno_df, _ = load_phenotype_bed(pheno_f)

        shared = (genotype_df.columns
                  .intersection(pheno_df.columns)
                  .intersection(covariates_df.index))
        shared = list(shared)
        genotype_df = genotype_df[shared]
        genotype_df, variant_df = apply_mac_filter(
            genotype_df, variant_df, n_samples=len(shared), mac_min=args.MAC)

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
