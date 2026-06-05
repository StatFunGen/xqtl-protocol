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
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def read_file_list(path: str) -> list:
    """Read a manifest and return the resolved file paths."""
    return [entry["path"] for entry in read_manifest_entries(path)]


def infer_manifest_chrom(entry_id: str | None, path: str) -> str:
    """Infer chromosome label from manifest row id or file basename."""
    if entry_id:
        norm = str(entry_id).strip().replace("chr", "")
        if norm.isdigit():
            return norm
    stem = phenotype_prefix(path) if path.endswith(".bed.gz") else Path(path).stem
    tail = stem.split(".")[-1].replace("chr", "")
    return tail


def read_manifest_entries(path: str) -> list[dict]:
    """
    Read either:
      1. a one-path-per-line manifest
      2. a 2-column table such as '#id\\t#path' or '#id\\t#dir'
    Returns a list of {'id': <optional>, 'path': <resolved path>}.
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        return []

    header = lines[0].split("\t")
    if len(header) >= 2 and header[0].startswith("#"):
        entries = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            entries.append({"id": parts[0], "path": parts[1]})
        return entries

    return [{"id": None, "path": line.strip()} for line in lines]


def strip_suffix(text: str, suffix: str) -> str:
    return text[:-len(suffix)] if text.endswith(suffix) else text


def phenotype_prefix(path: str) -> str:
    return strip_suffix(Path(path).name, ".bed.gz")


def local_temp_dir() -> str:
    """Use the workflow temp directory, falling back to ./tmp instead of /tmp."""
    tmpdir = os.environ.get("TMPDIR") or os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    return tmpdir


def direct_input_mode(genotype_file: str, phenotype_file: str) -> bool:
    return (
        genotype_file.endswith((".bed", ".pgen"))
        and phenotype_file.endswith(".bed.gz")
        and os.path.isfile(genotype_file)
        and os.path.isfile(phenotype_file)
    )


def genotype_prefix(path: str) -> str:
    if path.endswith(".bed"):
        return path[:-4]
    if path.endswith(".pgen"):
        return path[:-5]
    return path


def infer_single_chrom_label(phenotype_file: str) -> str:
    try:
        pos_df = pd.read_csv(
            phenotype_file,
            sep="\t",
            usecols=[0],
            compression="gzip" if phenotype_file.endswith(".gz") else None,
        )
    except Exception:
        return "0"
    chroms = sorted({str(x).replace("chr", "") for x in pos_df.iloc[:, 0].dropna().unique()})
    return chroms[0] if len(chroms) == 1 else "0"


def expected_cis_outputs(cwd: str, pheno_file: str, chrom_label: str) -> dict:
    prefix = phenotype_prefix(pheno_file)
    chrom_suffix = "" if str(chrom_label) == "0" else f"_chr{chrom_label}"
    parquet_suffix = "" if str(chrom_label) == "0" else str(chrom_label)
    return {
        "parquet": os.path.join(cwd, f"{prefix}.cis_qtl_pairs.{parquet_suffix}.parquet"),
        "nominal": os.path.join(cwd, f"{prefix}{chrom_suffix}.cis_qtl.pairs.tsv.gz"),
        "regional": os.path.join(cwd, f"{prefix}{chrom_suffix}.cis_qtl.regional.tsv.gz"),
    }


def resolve_cis_inputs(genotype_file: str, phenotype_file: str) -> list[dict]:
    if direct_input_mode(genotype_file, phenotype_file):
        chrom_label = infer_single_chrom_label(phenotype_file)
        return [{
            "chrom": chrom_label,
            "geno_prefix": genotype_prefix(genotype_file),
            "pheno_file": phenotype_file,
            "pheno_prefix": phenotype_prefix(phenotype_file),
        }]

    geno_entries = read_manifest_entries(genotype_file)
    pheno_entries = read_manifest_entries(phenotype_file)

    geno_by_chrom = {}
    for entry in geno_entries:
        chrom = infer_manifest_chrom(entry["id"], entry["path"])
        geno_by_chrom[chrom] = genotype_prefix(entry["path"])

    pheno_by_chrom = {}
    for entry in pheno_entries:
        chrom = infer_manifest_chrom(entry["id"], entry["path"])
        pheno_by_chrom[chrom] = entry["path"]

    chroms = sorted(set(geno_by_chrom) & set(pheno_by_chrom))
    return [{
        "chrom": chrom,
        "geno_prefix": geno_by_chrom[chrom],
        "pheno_file": pheno_by_chrom[chrom],
        "pheno_prefix": phenotype_prefix(pheno_by_chrom[chrom]),
    } for chrom in chroms]


def normalize_chromosomes(chromosomes: list[str] | None) -> list[str]:
    """Normalize CLI chromosome labels to labels without a chr prefix."""
    if not chromosomes:
        return []
    values = []
    for chrom in chromosomes:
        for part in str(chrom).replace(",", " ").split():
            part = part.strip()
            if part:
                values.append(part.replace("chr", ""))
    return values


def resolve_genotype_chrom(genotype_file: str, chrom: str) -> str:
    """Resolve a direct genotype file or manifest to the requested chromosome."""
    chrom = str(chrom).replace("chr", "")
    if genotype_file.endswith((".bed", ".pgen")):
        return genotype_prefix(genotype_file)

    for entry in read_manifest_entries(genotype_file):
        entry_chrom = infer_manifest_chrom(entry["id"], entry["path"])
        if str(entry_chrom).replace("chr", "") == chrom:
            return genotype_prefix(entry["path"])

    raise ValueError(f"No genotype file found for chromosome {chrom}")


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
      phenotype_pos_df: DataFrame indexed by phenotype_id, cols = ['chr', 'start', 'end']
    """
    df = pd.read_csv(bed_gz, sep="\t", index_col=3,
                     compression="gzip" if bed_gz.endswith(".gz") else None)
    pos_df = df.iloc[:, :3].copy()
    pos_df.columns = ["chr", "start", "end"]
    pos_df["chr"] = pos_df["chr"].astype(str).str.replace(r"^chr", "", regex=True)
    # Keep metadata columns such as "strand" until the caller intersects with
    # genotype/covariate sample IDs, matching the SoS notebook behavior.
    pheno_df = df.iloc[:, 3:].copy()
    return pheno_df, pos_df[["chr", "start", "end"]]


def filter_to_chromosome(genotype_df: pd.DataFrame,
                         variant_df: pd.DataFrame,
                         pheno_df: pd.DataFrame,
                         pheno_pos_df: pd.DataFrame,
                         chrom: str):
    """Filter direct multi-chromosome inputs to one requested chromosome."""
    chrom = str(chrom).replace("chr", "")

    variant_chrom_col = None
    for candidate in ("chrom", "chr"):
        if candidate in variant_df.columns:
            variant_chrom_col = candidate
            break
    if variant_chrom_col is not None:
        variant_chrom = variant_df[variant_chrom_col].astype(str).str.replace(r"^chr", "", regex=True)
        variant_df = variant_df.loc[variant_chrom == chrom].copy()
        genotype_df = genotype_df.loc[variant_df.index]

    pheno_chrom = pheno_pos_df["chr"].astype(str).str.replace(r"^chr", "", regex=True)
    pheno_pos_df = pheno_pos_df.loc[pheno_chrom == chrom].copy()
    pheno_df = pheno_df.loc[pheno_pos_df.index]

    return genotype_df, variant_df, pheno_df, pheno_pos_df


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def write_bgzip_table(df: pd.DataFrame, out_gz: str) -> None:
    out_tsv = strip_suffix(out_gz, ".gz")
    df.to_csv(out_tsv, sep="\t", index=False)
    run_command(["bgzip", "--compress-level", "9", "-f", out_tsv])
    run_command(["tabix", "-S", "1", "-s", "1", "-b", "2", "-e", "2", out_gz])


def apply_nominal_qvalues(tsv_path: str) -> None:
    r_code = """
args <- commandArgs(TRUE)
file_path <- args[1]
library(purrr)
library(tidyr)
library(readr)
library(dplyr)
library(qvalue)
compute_qvalues <- function(pvalues) {
    tryCatch({
        if(length(pvalues) < 2) {
            return(pvalues)
        } else {
            return(qvalue(pvalues)$qvalues)
        }
    }, error = function(e) {
        message("Too few p-values to calculate qvalue, fall back to qvalue(pi0 = 1)")
        qvalue(pvalues, pi0 = 1)$qvalues
    })
}
pairs_df = read_delim(file_path, delim = '\\t')
pairs_df = pairs_df %>% group_by(molecular_trait_id) %>% mutate(qvalue = compute_qvalues(pvalue))
pairs_df %>% write_delim(file_path, '\\t')
"""
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, dir=local_temp_dir()) as handle:
        handle.write(r_code)
        script_path = handle.name
    try:
        run_command(["Rscript", script_path, tsv_path])
    finally:
        os.unlink(script_path)


def run_regional_postprocess(regional_files: list[str], out_tsv: str, out_summary: str) -> None:
    r_code = """
args <- commandArgs(TRUE)
out_tsv <- args[1]
out_summary <- args[2]
input_files <- args[-c(1,2)]
library(purrr)
library(tidyr)
library(dplyr)
library(readr)
library(qvalue)
emprical_pd = tibble(map(input_files, ~read_delim(.x,'\\t')))%>%unnest()
emprical_pd['q_beta'] = tryCatch(qvalue(emprical_pd$p_beta)$qvalue, error = function(e){print('Too few pvalue to calculate qvalue, fall back to BH')
                                                                                          qvalue(emprical_pd$p_beta,pi0 = 1 )$qvalue})
emprical_pd['q_perm'] = tryCatch(qvalue(emprical_pd$p_perm)$qvalue, error = function(e){print('Too few pvalue to calculate qvalue, fall back to BH')
                                                                                          qvalue(emprical_pd$p_perm,pi0 = 1 )$qvalue})
emprical_pd['fdr_beta'] = p.adjust(emprical_pd$p_beta,'fdr')
emprical_pd['fdr_perm'] = p.adjust(emprical_pd$p_perm,'fdr')
if (!all(is.na(emprical_pd$p_beta))) {
  lb <- emprical_pd %>% filter(q_beta <= 0.05) %>% pull(p_beta) %>% sort()
  ub <- emprical_pd %>% filter(q_beta > 0.05) %>% pull(p_beta) %>% sort()
  if (length(lb) > 0) {
    lb_val <- tail(lb, 1)
    threshold <- if (length(ub) > 0) (lb_val + head(ub, 1)) / 2 else lb_val
    message(sprintf('min p-value threshold @ FDR 0.05: %g', threshold))
    emprical_pd <- emprical_pd %>% mutate(p_nominal_threshold = qbeta(threshold, beta_shape1, beta_shape2))
  }
}
summary = tibble('fdr_perm_0.05' = sum(emprical_pd['fdr_perm'] < 0.05),
                 'fdr_beta_0.05' = sum(emprical_pd['fdr_beta'] < 0.05),
                 'q_perm_0.05' = sum(emprical_pd['q_perm'] < 0.05),
                 'q_beta_0.05' = sum(emprical_pd['q_beta'] < 0.05),
                 'fdr_perm_0.01' = sum(emprical_pd['fdr_perm'] < 0.01),
                 'fdr_beta_0.01' = sum(emprical_pd['fdr_beta'] < 0.01),
                 'q_perm_0.01' = sum(emprical_pd['q_perm'] < 0.01),
                 'q_beta_0.01' = sum(emprical_pd['q_beta'] < 0.01))
emprical_pd %>% write_delim(out_tsv, '\\t')
summary %>% write_delim(out_summary, '\\t')
"""
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, dir=local_temp_dir()) as handle:
        handle.write(r_code)
        script_path = handle.name
    try:
        run_command(["Rscript", script_path, out_tsv, out_summary, *regional_files])
    finally:
        os.unlink(script_path)


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

    if args.dry_run:
        import sys as _sys
        print("[DRY-RUN] TensorQTL.py cis — would execute:")
        print(f"  python {os.path.abspath(__file__)} \\")
        print(f"    --step cis \\")
        print(f"    --genotype-file {args.genotype_file} \\")
        print(f"    --phenotype-file {args.phenotype_file} \\")
        print(f"    --covariate-file {args.covariate_file} \\")
        print(f"    --cwd {args.cwd} \\")
        if args.chromosome:
            print(f"    --chromosome {' '.join(args.chromosome)} \\")
        print(f"    --window {args.window} --MAC {args.MAC} --maf-threshold {args.maf_threshold} \\")
        print(f"    --numThreads {args.numThreads}")
        print("\n[DRY-RUN] Input file check:")
        for _label, _path in [
            ("genotype manifest", args.genotype_file),
            ("phenotype manifest", args.phenotype_file),
            ("covariate file",    args.covariate_file),
        ]:
            _ok = "\u2713" if os.path.isfile(_path) else "\u2717 NOT FOUND"
            print(f"  {_ok}  {_label}: {_path}")
            if os.path.isfile(_path):
                try:
                    files = read_file_list(_path)
                    print(f"      ({len(files)} entries)")
                    for _i, _f in enumerate(files[:3]):
                        _fok = "\u2713" if os.path.isfile(_f) else "\u2717"
                        print(f"      {_fok} {_f}")
                    if len(files) > 3:
                        print(f"      ... and {len(files)-3} more")
                except Exception:
                    pass
        return

    covariates_df = load_covariates(args.covariate_file)
    input_pairs = resolve_cis_inputs(args.genotype_file, args.phenotype_file)
    requested_chroms = normalize_chromosomes(args.chromosome)
    if requested_chroms:
        expanded_pairs = []
        for pair in input_pairs:
            pair_chrom = str(pair["chrom"]).replace("chr", "")
            if pair_chrom == "0":
                for chrom in requested_chroms:
                    expanded_pair = dict(pair)
                    expanded_pair["chrom"] = chrom
                    expanded_pair["filter_chrom"] = chrom
                    expanded_pairs.append(expanded_pair)
            elif pair_chrom in requested_chroms:
                expanded_pairs.append(pair)
        input_pairs = expanded_pairs
    if not input_pairs:
        sys.exit("ERROR: No matching chromosomes between genotype and phenotype lists.")
    print(f"Running cis-QTL on {len(input_pairs)} input group(s)", flush=True)

    regional_outputs = []

    for pair in input_pairs:
        chrom = str(pair["chrom"])
        geno_prefix = pair["geno_prefix"]
        pheno_file = pair["pheno_file"]
        pheno_prefix = pair["pheno_prefix"]
        print(f"\n=== {chrom} ===", flush=True)

        # Load genotype using the correct TensorQTL API
        genotype_df, variant_df = genotypeio.load_genotypes(geno_prefix, dosages=True)
        pheno_df, pheno_pos_df = load_phenotype_bed(pheno_file)
        if pair.get("filter_chrom"):
            genotype_df, variant_df, pheno_df, pheno_pos_df = filter_to_chromosome(
                genotype_df, variant_df, pheno_df, pheno_pos_df, pair["filter_chrom"])
        expected = expected_cis_outputs(args.cwd, pheno_file, chrom)

        # Align samples across genotype, phenotype, and covariates
        shared = (genotype_df.columns
                  .intersection(pheno_df.columns)
                  .intersection(covariates_df.index))
        shared = list(shared)
        if not shared:
            print(f"  WARNING: No shared samples for {chrom}, skipping.", flush=True)
            continue

        pheno_df     = pheno_df[shared].astype(float)
        covariates_t = covariates_df.loc[shared]
        genotype_df  = genotype_df[shared]

        # Apply MAC filter
        genotype_df, variant_df = apply_mac_filter(
            genotype_df, variant_df, n_samples=len(shared), mac_min=args.MAC)

        # map_nominal writes parquet output to disk; it does not return a DataFrame.
        nominal_prefix = f"{pheno_prefix}{'' if chrom == '0' else f'_chr{chrom}'}"
        cis.map_nominal(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            nominal_prefix,
            covariates_df=covariates_t,
            window=args.window,
            maf_threshold=args.maf_threshold,
            run_eigenmt=True,
            output_dir=args.cwd,
        )
        parquet_candidates = sorted(glob.glob(os.path.join(args.cwd, f"{nominal_prefix}.cis_qtl_pairs.*.parquet")))
        if parquet_candidates:
            Path(expected["parquet"]).parent.mkdir(parents=True, exist_ok=True)
            if os.path.abspath(parquet_candidates[0]) != os.path.abspath(expected["parquet"]):
                os.replace(parquet_candidates[0], expected["parquet"])
            parquet_file = expected["parquet"]
        else:
            pd.DataFrame().to_parquet(expected["parquet"])
            parquet_file = expected["parquet"]
        print(f"  Nominal pass complete -> {parquet_file}", flush=True)

        pairs_df = pd.read_parquet(parquet_file)
        pairs_df["molecular_trait_object_id"] = pairs_df["phenotype_id"]
        if "end_distance" not in pairs_df.columns:
            start_pos = pairs_df.columns.get_loc("start_distance")
            pairs_df.insert(start_pos + 1, "end_distance", pairs_df["start_distance"])
        pairs_df.rename(columns={
            "phenotype_id": "molecular_trait_id",
            "pval_nominal": "pvalue",
            "slope": "bhat",
            "slope_se": "sebhat",
            "start_distance": "tss_distance",
            "end_distance": "tes_distance",
        }, inplace=True)
        pairs_df["n"] = len(shared)
        pairs_df = variant_df.merge(pairs_df, right_on="variant_id", left_index=True)
        pairs_df.rename(columns={"a1": "a2", "a0": "a1"}, inplace=True)
        if not pairs_df["pos"].is_monotonic_increasing:
            pairs_df = pairs_df.sort_values(by=["chrom", "pos"])
        nominal_tsv = strip_suffix(expected["nominal"], ".gz")
        pairs_df.to_csv(nominal_tsv, sep="\t", index=False)
        apply_nominal_qvalues(nominal_tsv)
        run_command(["bgzip", "--compress-level", "9", "-f", nominal_tsv])
        run_command(["tabix", "-S", "1", "-s", "1", "-b", "2", "-e", "2", expected["nominal"]])

        lambda_col = pairs_df.groupby("molecular_trait_object_id").apply(
            lambda x: stats.chi2.ppf(1.0 - np.median(x["pvalue"]), 1) / stats.chi2.ppf(0.5, 1)
        )

        # Permutation pass — returns a DataFrame
        perm_df = cis.map_cis(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            covariates_df=covariates_t,
            window=args.window,
            maf_threshold=args.maf_threshold,
            seed=999,
        )
        perm_df.index.name = "molecular_trait_id"
        if "group_id" not in perm_df.columns:
            perm_df["group_id"] = perm_df.index
            perm_df["group_size"] = 1
        perm_df.rename(columns={
            "group_id": "molecular_trait_object_id",
            "group_size": "n_traits",
            "start_distance": "tss_distance",
            "end_distance": "tes_distance",
            "num_var": "n_variants",
            "pval_nominal": "p_nominal",
            "slope": "bhat",
            "slope_se": "sebhat",
            "pval_true_df": "p_true_df",
            "pval_perm": "p_perm",
            "pval_beta": "p_beta",
        }, inplace=True)
        perm_df["genomic_inflation"] = perm_df["molecular_trait_object_id"].map(lambda_col)
        perm_df = variant_df.merge(perm_df, right_on="variant_id", left_index=True)
        perm_df.rename(columns={"a1": "a2", "a0": "a1"}, inplace=True)
        if not perm_df["pos"].is_monotonic_increasing:
            perm_df = perm_df.sort_values(by=["chrom", "pos"])
        write_bgzip_table(perm_df, expected["regional"])
        regional_outputs.append(expected["regional"])
        print(f"  Regional results -> {expected['regional']}", flush=True)
        print(f"  Nominal results  -> {expected['nominal']}", flush=True)

    if not regional_outputs:
        print("No regional results produced.", flush=True)
        return

    output_prefix = strip_suffix(os.path.basename(regional_outputs[0]), ".cis_qtl.regional.tsv.gz")
    out_tsv = os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.tsv.gz")
    out_summary = os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.summary.txt")
    run_regional_postprocess(regional_outputs, out_tsv, out_summary)
    print(f"Regional significance table: {out_tsv}", flush=True)
    print(f"Regional significance summary: {out_summary}", flush=True)

    print(f"\nCIS QTL complete. Results in: {args.cwd}", flush=True)


def run_cis_postprocess(args) -> None:
    regional_files = sorted(glob.glob(os.path.join(args.cwd, "*.cis_qtl.regional.tsv.gz")))
    if not regional_files:
        sys.exit(f"ERROR: No regional cis-QTL files found in {args.cwd}")

    prefix_candidates = [
        strip_suffix(os.path.basename(path), ".cis_qtl.regional.tsv.gz")
        for path in regional_files
    ]
    output_prefix = prefix_candidates[0]

    out_tsv = os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.tsv.gz")
    out_summary = os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.summary.txt")
    run_regional_postprocess(regional_files, out_tsv, out_summary)
    print(f"Regional significance table: {out_tsv}", flush=True)
    print(f"Regional significance summary: {out_summary}", flush=True)


def run_trans(args) -> None:
    """Run trans-QTL genome-wide association scan."""
    try:
        from tensorqtl import genotypeio, trans
    except ImportError:
        sys.exit("ERROR: tensorqtl package not installed.")

    os.makedirs(args.cwd, exist_ok=True)

    if args.dry_run:
        print("[DRY-RUN] TensorQTL.py trans — would execute:")
        print(f"  python {os.path.abspath(__file__)} \\")
        print(f"    --step trans \\")
        print(f"    --genotype-file {args.genotype_file} \\")
        print(f"    --phenotype-file {args.phenotype_file} \\")
        print(f"    --covariate-file {args.covariate_file} \\")
        print(f"    --cwd {args.cwd} \\")
        if args.trans_geno_chromosome:
            print(f"    --trans-geno-chromosome {args.trans_geno_chromosome} \\")
        print(f"    --MAC {args.MAC} --maf-threshold {args.maf_threshold} \\")
        print(f"    --numThreads {args.numThreads}")
        return

    covariates_df = load_covariates(args.covariate_file)
    trans_geno_chrom = str(args.trans_geno_chromosome).replace("chr", "") if args.trans_geno_chromosome else ""
    forced_geno_prefix = resolve_genotype_chrom(args.genotype_file, trans_geno_chrom) if trans_geno_chrom else ""

    input_pairs = resolve_cis_inputs(args.genotype_file, args.phenotype_file)
    for pair in input_pairs:
        chrom = str(pair["chrom"])
        geno_prefix = forced_geno_prefix or pair["geno_prefix"]
        pheno_f = pair["pheno_file"]

        genotype_df, variant_df = genotypeio.load_genotypes(geno_prefix, dosages=True)
        if trans_geno_chrom:
            chrom_filter = (
                variant_df["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
                == trans_geno_chrom
            )
            if chrom_filter.sum() == 0:
                raise ValueError(f"No variants found for chromosome {trans_geno_chrom} in genotype data")
            variant_df = variant_df[chrom_filter]
            genotype_df = genotype_df.loc[variant_df.index]

        pheno_df, _ = load_phenotype_bed(pheno_f)

        shared = (genotype_df.columns
                  .intersection(pheno_df.columns)
                  .intersection(covariates_df.index))
        shared = list(shared)
        genotype_df = genotype_df[shared]
        genotype_df, variant_df = apply_mac_filter(
            genotype_df, variant_df, n_samples=len(shared), mac_min=args.MAC)

        trans_df = trans.map_trans(
            genotype_df, variant_df, pheno_df[shared].astype(float),
            covariates_df=covariates_df.loc[shared],
            maf_threshold=args.maf_threshold,
        )
        geno_suffix = f"_geno_chr{trans_geno_chrom}" if trans_geno_chrom else ""
        out = os.path.join(args.cwd, f"{chrom}{geno_suffix}.trans_qtl_pairs.parquet")
        trans_df.to_parquet(out)
        print(f"  Trans {chrom}: {len(trans_df)} pairs → {out}", flush=True)

    print(f"\nTRANS QTL complete. Results in: {args.cwd}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TensorQTL wrapper (mirrors TensorQTL.ipynb)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--step", required=True, choices=["cis", "cis_postprocess", "trans"],
                   help="Which step to run")
    p.add_argument("--genotype-file", metavar="PATH",
                   help="Path to a genotype manifest or a direct PLINK .bed/.pgen file")
    p.add_argument("--phenotype-file", metavar="PATH",
                   help="Path to a phenotype manifest or a direct BED.gz file")
    p.add_argument("--covariate-file", metavar="PATH",
                   help="Hidden factor covariate file (gzip TSV, covariates × samples)")
    p.add_argument("--cwd", default="output", metavar="DIR",
                   help="Output directory")
    p.add_argument("--chromosome", nargs="*", default=[],
                   help="Chromosomes to run for cis-QTL; accepts values with or without chr")
    p.add_argument("--window", type=int, default=1_000_000, metavar="BP",
                   help="CIS window in bp")
    p.add_argument("--MAC", type=int, default=5, metavar="N",
                   help="Minimum minor allele count filter")
    p.add_argument("--maf-threshold", type=float, default=0.0, metavar="F",
                   help="Minimum MAF filter (0 = no filter)")
    p.add_argument("--trans-geno-chromosome", default="", metavar="CHR",
                   help="For trans-QTL, use this genotype chromosome instead of the per-phenotype chromosome")
    p.add_argument("--numThreads", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print the full command and validate inputs; do not run TensorQTL.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.step in {"cis", "trans"}:
        missing = [
            flag for flag, value in [
                ("--genotype-file", args.genotype_file),
                ("--phenotype-file", args.phenotype_file),
                ("--covariate-file", args.covariate_file),
            ] if not value
        ]
        if missing:
            parser.error(f"missing required arguments for {args.step}: {' '.join(missing)}")
    if args.step == "cis":
        run_cis(args)
    elif args.step == "cis_postprocess":
        run_cis_postprocess(args)
    elif args.step == "trans":
        run_trans(args)


if __name__ == "__main__":
    main()
