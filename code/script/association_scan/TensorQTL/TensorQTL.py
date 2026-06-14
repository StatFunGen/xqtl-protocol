#!/usr/bin/env python3
"""
TensorQTL.py
Mirrors: code/SoS/association_scan/TensorQTL/TensorQTL.ipynb

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
    return expected_cis_outputs_for_interaction(cwd, pheno_file, chrom_label, "")


def expected_cis_outputs_for_interaction(cwd: str, pheno_file: str, chrom_label: str,
                                         interaction_name: str = "") -> dict:
    prefix = phenotype_prefix(pheno_file)
    chrom_suffix = "" if str(chrom_label) == "0" else f"_chr{chrom_label}"
    parquet_suffix = "" if str(chrom_label) == "0" else str(chrom_label)
    interaction_suffix = f"_{interaction_name}" if interaction_name else ""
    return {
        "parquet": os.path.join(cwd, f"{prefix}{interaction_suffix}.cis_qtl_pairs.{parquet_suffix}.parquet"),
        "nominal": os.path.join(cwd, f"{prefix}{chrom_suffix}{interaction_suffix}.cis_qtl.pairs.tsv.gz"),
        "regional": os.path.join(cwd, f"{prefix}{chrom_suffix}{interaction_suffix}.cis_qtl.regional.tsv.gz"),
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


def apply_covariate_pattern(covariates_df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    if not patterns:
        return covariates_df
    pattern_mapping = {
        "pheno_PC": ["Hidden_Factor_PC"],
        "geno_PC": ["PC"],
    }
    keep_cols = []
    for col in covariates_df.columns:
        if col in patterns:
            keep_cols.append(col)
            continue
        for pattern in patterns:
            for mapped_pattern in pattern_mapping.get(pattern, [pattern]):
                if str(col).startswith(mapped_pattern):
                    keep_cols.append(col)
                    break
            if col in keep_cols:
                break
    if not keep_cols:
        print("WARNING: No covariate columns match --covariate-pattern; keeping all covariates.", flush=True)
        return covariates_df
    return covariates_df[keep_cols]


def apply_keep_samples(covariates_df: pd.DataFrame, keep_sample: str) -> pd.DataFrame:
    if not keep_sample or not os.path.isfile(keep_sample):
        return covariates_df
    sample_df = pd.read_csv(keep_sample, comment="#", header=None, names=["sample_id"], sep="\t")
    sample_ids = sample_df["sample_id"].astype(str).str.strip().tolist()
    return covariates_df.loc[covariates_df.index.intersection(sample_ids)]


def read_region_phenotypes(region_list: str, phenotype_column: int) -> set[str]:
    if not region_list or not os.path.isfile(region_list):
        return set()
    region = pd.read_csv(region_list, comment="#", header=None, sep="\t", dtype=str)
    if region.empty:
        return set()
    column = 0 if len(region.columns) == 1 else max(int(phenotype_column) - 1, 0)
    if column >= len(region.columns):
        raise ValueError(
            f"--region-list-phenotype-column {phenotype_column} is outside the {len(region.columns)}-column region list"
        )
    return set(region.iloc[:, column].astype(str).str.strip())


def filter_phenotypes_by_region(pheno_df: pd.DataFrame, pheno_pos_df: pd.DataFrame,
                                region_list: str, phenotype_column: int):
    keep_region = read_region_phenotypes(region_list, phenotype_column)
    if not keep_region:
        return pheno_df, pheno_pos_df
    keep = pheno_df.index.astype(str).isin(keep_region)
    pheno_df = pheno_df.loc[keep]
    pheno_pos_df = pheno_pos_df.loc[pheno_pos_df.index.astype(str).isin(keep_region)]
    return pheno_df, pheno_pos_df


def apply_custom_cis_windows(pheno_df: pd.DataFrame, pheno_pos_df: pd.DataFrame,
                             customized_cis_windows: str):
    if not customized_cis_windows or not os.path.isfile(customized_cis_windows):
        return pheno_df, pheno_pos_df, None

    phenotype_id = pheno_pos_df.index.name or "ID"
    cis_list = pd.read_csv(
        customized_cis_windows,
        comment="#",
        header=None,
        names=["chr", "start", "end", phenotype_id],
        sep="\t",
    )
    cis_list["chr"] = cis_list["chr"].astype(str).str.replace(r"^chr", "", regex=True)
    if cis_list[["chr", phenotype_id]].duplicated().sum() != 0:
        cis_list = (
            cis_list.groupby([phenotype_id, "chr"])
            .agg({"start": "min", "end": "max"})
            .reset_index()[["chr", "start", "end", phenotype_id]]
        )

    pos_reset = pheno_pos_df.reset_index()
    pos_id_col = pos_reset.columns[0]
    merged_pos = pos_reset.merge(
        cis_list,
        left_on=["chr", pos_id_col],
        right_on=["chr", phenotype_id],
        suffixes=("_default", ""),
    )
    if merged_pos.empty:
        raise ValueError("No phenotypes matched --customized-cis-windows")
    if merged_pos[pos_id_col].duplicated().sum() != 0:
        raise ValueError("customized cis windows did not uniquely match phenotype IDs")

    matched_ids = merged_pos[pos_id_col].astype(str)
    original_missing = set(pheno_df.index.astype(str)) - set(matched_ids)
    if original_missing:
        pheno_df = pheno_df.loc[pheno_df.index.astype(str).isin(matched_ids)]
    if len(pheno_df.index) != len(merged_pos.index):
        raise ValueError("cannot uniquely match all phenotype data to customized cis windows")

    merged_pos = merged_pos.set_index(pos_id_col)[["chr", "start", "end"]]
    merged_pos.index.name = pheno_pos_df.index.name
    return pheno_df.loc[merged_pos.index], merged_pos, 0


def load_phenotype_group(phenotype_group: str):
    if not phenotype_group or not os.path.isfile(phenotype_group):
        return None
    return pd.read_csv(phenotype_group, sep="\t", header=None, index_col=0).squeeze("columns")


def load_interaction(interaction: str, covariates_df: pd.DataFrame):
    if not interaction:
        return None, ""
    if os.path.isfile(interaction):
        interaction_s = pd.read_csv(interaction, sep="\t", index_col=0)
        if interaction_s.shape[1] == 0:
            raise ValueError(f"Interaction file has no value columns: {interaction}")
        interaction_name = str(interaction_s.columns[0])
        return interaction_s.iloc[:, [0]], interaction_name
    if interaction in covariates_df.columns:
        return covariates_df[[interaction]], interaction
    raise ValueError(
        f"--interaction must be either a file or a covariate column name; got {interaction}"
    )


def align_analysis_inputs(genotype_df: pd.DataFrame, pheno_df: pd.DataFrame,
                          covariates_df: pd.DataFrame, interaction_df=None):
    shared = genotype_df.columns.intersection(pheno_df.columns).intersection(covariates_df.index)
    if interaction_df is not None:
        shared = shared.intersection(interaction_df.index)
    shared = list(shared)
    if interaction_df is not None:
        return shared, interaction_df.loc[shared]
    return shared, None


def rename_nominal_columns(pairs_df: pd.DataFrame, interaction_name: str,
                           custom_cis_window: bool) -> pd.DataFrame:
    distance_renames = {
        "start_distance": "cis_window_start_distance" if custom_cis_window else "tss_distance",
        "end_distance": "cis_window_end_distance" if custom_cis_window else "tes_distance",
    }
    if interaction_name:
        interaction_renames = {
            "phenotype_id": "molecular_trait_id",
            "pval_g": "pvalue",
            "b_g": "bhat",
            "b_g_se": "sebhat",
            "pval_i": f"pvalue_{interaction_name}",
            "b_i": f"bhat_{interaction_name}",
            "b_i_se": f"sebhat_{interaction_name}",
            "pval_gi": f"pvalue_{interaction_name}_interaction",
            "b_gi": f"bhat_{interaction_name}_interaction",
            "b_gi_se": f"sebhat_{interaction_name}_interaction",
        }
        interaction_renames.update(distance_renames)
        return pairs_df.rename(columns=interaction_renames)

    plain_renames = {
        "phenotype_id": "molecular_trait_id",
        "pval_nominal": "pvalue",
        "slope": "bhat",
        "slope_se": "sebhat",
    }
    plain_renames.update(distance_renames)
    return pairs_df.rename(columns=plain_renames)


def genomic_inflation_by_trait(pairs_df: pd.DataFrame, pvalue_col: str = "pvalue") -> pd.Series:
    return pairs_df.groupby("molecular_trait_object_id").apply(
        lambda x: stats.chi2.ppf(1.0 - np.median(x[pvalue_col]), 1) / stats.chi2.ppf(0.5, 1)
    )


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


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value}")


def tensorqtl_r_script() -> str:
    script_path = str(Path(__file__).with_suffix(".R"))
    if not os.path.isfile(script_path):
        sys.exit(f"ERROR: TensorQTL R helper missing: {script_path}")
    return script_path


def write_bgzip_table(df: pd.DataFrame, out_gz: str) -> None:
    out_tsv = strip_suffix(out_gz, ".gz")
    df.to_csv(out_tsv, sep="\t", index=False)
    run_command(["bgzip", "--compress-level", "9", "-f", out_tsv])
    run_command(["tabix", "-S", "1", "-s", "1", "-b", "2", "-e", "2", out_gz])


def apply_nominal_qvalues(tsv_path: str, interaction: str = "") -> None:
    cmd = ["Rscript", tensorqtl_r_script(), "nominal_qvalues", tsv_path]
    if interaction:
        cmd.append(interaction)
    run_command(cmd)


def apply_trans_qvalues(tsv_path: str) -> None:
    run_command(["Rscript", tensorqtl_r_script(), "trans_qvalues", tsv_path])


def run_regional_postprocess(regional_files: list[str], out_tsv: str, out_summary: str) -> None:
    run_command(["Rscript", tensorqtl_r_script(), "regional_postprocess",
                 out_tsv, out_summary, *regional_files])


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
    covariates_df = apply_covariate_pattern(covariates_df, args.covariate_pattern)
    covariates_df = apply_keep_samples(covariates_df, args.keep_sample)
    interaction_df, interaction_name = load_interaction(args.interaction, covariates_df)
    if interaction_name and interaction_name in covariates_df.columns:
        covariates_df = covariates_df.drop(columns=[interaction_name])
    group_s = load_phenotype_group(args.phenotype_group)
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
        pheno_df, pheno_pos_df = filter_phenotypes_by_region(
            pheno_df, pheno_pos_df, args.region_list, args.region_list_phenotype_column)
        if pair.get("filter_chrom"):
            genotype_df, variant_df, pheno_df, pheno_pos_df = filter_to_chromosome(
                genotype_df, variant_df, pheno_df, pheno_pos_df, pair["filter_chrom"])
        pheno_df, pheno_pos_df, custom_window = apply_custom_cis_windows(
            pheno_df, pheno_pos_df, args.customized_cis_windows)
        effective_window = args.window if custom_window is None else custom_window
        expected = expected_cis_outputs_for_interaction(args.cwd, pheno_file, chrom, interaction_name)

        # Align samples across genotype, phenotype, and covariates
        shared, interaction_t = align_analysis_inputs(genotype_df, pheno_df, covariates_df, interaction_df)
        if not shared:
            print(f"  WARNING: No shared samples for {chrom}, skipping.", flush=True)
            continue
        if pheno_df.empty:
            print(f"  WARNING: No phenotypes left for {chrom}, skipping.", flush=True)
            continue

        pheno_df     = pheno_df[shared].astype(float)
        covariates_t = covariates_df.loc[shared]
        genotype_df  = genotype_df[shared]

        # Apply MAC filter
        genotype_df, variant_df = apply_mac_filter(
            genotype_df, variant_df, n_samples=len(shared), mac_min=args.MAC)

        # map_nominal writes parquet output to disk; it does not return a DataFrame.
        nominal_prefix = f"{pheno_prefix}{'_' + interaction_name if interaction_name else ''}"
        if not (args.skip_nominal_if_exist and os.path.isfile(expected["parquet"])):
            if interaction_t is not None:
                cis.map_nominal(
                    genotype_df, variant_df, pheno_df, pheno_pos_df,
                    nominal_prefix,
                    covariates_df=covariates_t,
                    interaction_df=interaction_t,
                    maf_threshold_interaction=args.maf_threshold,
                    window=effective_window,
                    group_s=group_s,
                    run_eigenmt=True,
                    output_dir=args.cwd,
                )
            else:
                cis.map_nominal(
                    genotype_df, variant_df, pheno_df, pheno_pos_df,
                    nominal_prefix,
                    covariates_df=covariates_t,
                    window=effective_window,
                    maf_threshold=args.maf_threshold,
                    run_eigenmt=not args.permutation,
                    group_s=group_s,
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
        if interaction_t is not None and "pval_gi" in pairs_df.columns:
            pairs_df = pairs_df.dropna(subset=["pval_gi"])
        pairs_df["molecular_trait_object_id"] = pairs_df["phenotype_id"]
        if "end_distance" not in pairs_df.columns:
            start_pos = pairs_df.columns.get_loc("start_distance")
            pairs_df.insert(start_pos + 1, "end_distance", pairs_df["start_distance"])
        pairs_df = rename_nominal_columns(
            pairs_df, interaction_name, custom_cis_window=custom_window is not None)
        pairs_df["n"] = len(shared)
        pairs_df = variant_df.merge(pairs_df, right_on="variant_id", left_index=True)
        pairs_df.rename(columns={"a1": "a2", "a0": "a1"}, inplace=True)
        if not pairs_df["pos"].is_monotonic_increasing:
            pairs_df = pairs_df.sort_values(by=["chrom", "pos"])
        nominal_tsv = strip_suffix(expected["nominal"], ".gz")
        pairs_df.to_csv(nominal_tsv, sep="\t", index=False)
        apply_nominal_qvalues(nominal_tsv, args.interaction or interaction_name)
        run_command(["bgzip", "--compress-level", "9", "-f", nominal_tsv])
        run_command(["tabix", "-S", "1", "-s", "1", "-b", "2", "-e", "2", expected["nominal"]])

        test_regional_association = args.permutation and interaction_t is None
        if not test_regional_association:
            print(f"  Nominal results  -> {expected['nominal']}", flush=True)
            continue

        lambda_col = genomic_inflation_by_trait(pairs_df)

        # Permutation pass — returns a DataFrame
        perm_df = cis.map_cis(
            genotype_df, variant_df, pheno_df, pheno_pos_df,
            covariates_df=covariates_t,
            window=effective_window,
            maf_threshold=args.maf_threshold,
            group_s=group_s,
            seed=999,
        )
        perm_df.index.name = "molecular_trait_id"
        if "group_id" not in perm_df.columns:
            perm_df["group_id"] = perm_df.index
            perm_df["group_size"] = 1
        regional_distance_renames = {
            "start_distance": "cis_window_start_distance" if custom_window is not None else "tss_distance",
            "end_distance": "cis_window_end_distance" if custom_window is not None else "tes_distance",
        }
        regional_renames = {
            "group_id": "molecular_trait_object_id",
            "group_size": "n_traits",
            "num_var": "n_variants",
            "pval_nominal": "p_nominal",
            "slope": "bhat",
            "slope_se": "sebhat",
            "pval_true_df": "p_true_df",
            "pval_perm": "p_perm",
            "pval_beta": "p_beta",
        }
        regional_renames.update(regional_distance_renames)
        perm_df.rename(columns=regional_renames, inplace=True)
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
    if args.skip_postprocess:
        print("\nCIS QTL complete. Regional postprocess skipped.", flush=True)
        return

    output_prefix = strip_suffix(os.path.basename(regional_outputs[0]), ".cis_qtl.regional.tsv.gz")
    out_tsv = os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.tsv.gz")
    out_summary = os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.summary.txt")
    run_regional_postprocess(regional_outputs, out_tsv, out_summary)
    print(f"Regional significance table: {out_tsv}", flush=True)
    print(f"Regional significance summary: {out_summary}", flush=True)

    print(f"\nCIS QTL complete. Results in: {args.cwd}", flush=True)


def run_cis_postprocess(args) -> None:
    regional_files = args.regional_files or sorted(glob.glob(os.path.join(args.cwd, "*.cis_qtl.regional.tsv.gz")))
    if not regional_files:
        sys.exit(f"ERROR: No regional cis-QTL files found in {args.cwd}")

    prefix_candidates = [
        strip_suffix(os.path.basename(path), ".cis_qtl.regional.tsv.gz")
        for path in regional_files
    ]
    output_prefix = prefix_candidates[0]

    out_tsv = args.output_tsv or os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.tsv.gz")
    out_summary = args.output_summary or os.path.join(args.cwd, f"{output_prefix}.cis_qtl_regional_significance.summary.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_tsv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_summary)), exist_ok=True)
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
    covariates_df = apply_covariate_pattern(covariates_df, args.covariate_pattern)
    covariates_df = apply_keep_samples(covariates_df, args.keep_sample)
    trans_geno_chrom = str(args.trans_geno_chromosome).replace("chr", "") if args.trans_geno_chromosome else ""
    forced_geno_prefix = resolve_genotype_chrom(args.genotype_file, trans_geno_chrom) if trans_geno_chrom else ""

    all_results = []
    input_pairs = resolve_cis_inputs(args.genotype_file, args.phenotype_file)
    for pair in input_pairs:
        chrom = str(pair["chrom"])
        geno_prefix = forced_geno_prefix or pair["geno_prefix"]
        pheno_f = pair["pheno_file"]

        genotype_df, variant_df = genotypeio.load_genotypes(geno_prefix, dosages=True)
        variant_df["chrom"] = variant_df["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
        if trans_geno_chrom:
            chrom_filter = (
                variant_df["chrom"].astype(str).str.replace(r"^chr", "", regex=True) == trans_geno_chrom
            )
            if chrom_filter.sum() == 0:
                raise ValueError(f"No variants found for chromosome {trans_geno_chrom} in genotype data")
            variant_df = variant_df[chrom_filter]
            genotype_df = genotype_df.loc[variant_df.index]

        pheno_df, pheno_pos_df = load_phenotype_bed(pheno_f)
        pheno_df, pheno_pos_df = filter_phenotypes_by_region(
            pheno_df, pheno_pos_df, args.region_list, args.region_list_phenotype_column)
        if pair.get("filter_chrom"):
            genotype_df, variant_df, pheno_df, pheno_pos_df = filter_to_chromosome(
                genotype_df, variant_df, pheno_df, pheno_pos_df, pair["filter_chrom"])
        pheno_df, pheno_pos_df, custom_window = apply_custom_cis_windows(
            pheno_df, pheno_pos_df, args.customized_cis_windows)
        effective_window = args.window if custom_window is None else custom_window

        pheno_pos_df["chr"] = pheno_pos_df["chr"].astype(str).str.replace(r"^chr", "", regex=True)
        pheno_chroms = sorted(pheno_pos_df["chr"].unique().tolist())
        if chrom != "0" and chrom in pheno_chroms:
            pheno_chroms = [chrom]
        geno_chroms = sorted(variant_df["chrom"].astype(str).str.replace(r"^chr", "", regex=True).unique().tolist())

        for pheno_chrom in pheno_chroms:
            phenotype_pos_df_filtered = pheno_pos_df.loc[pheno_pos_df["chr"] == pheno_chrom]
            phenotype_df_filtered = pheno_df.loc[pheno_df.index.isin(phenotype_pos_df_filtered.index)]
            if phenotype_df_filtered.empty:
                print(f"  No phenotypes found for chromosome {pheno_chrom}, skipping.", flush=True)
                continue

            for geno_chrom in geno_chroms:
                chrom_variants = variant_df.index[
                    variant_df["chrom"].astype(str).str.replace(r"^chr", "", regex=True) == geno_chrom
                ].tolist()
                if not chrom_variants:
                    print(f"  No variants found for chromosome {geno_chrom}, skipping.", flush=True)
                    continue

                genotype_df_chr = genotype_df.loc[chrom_variants]
                variant_df_chr = variant_df.loc[chrom_variants]
                shared = (
                    genotype_df_chr.columns
                    .intersection(phenotype_df_filtered.columns)
                    .intersection(covariates_df.index)
                )
                shared = list(shared)
                if not shared:
                    print(
                        f"  No common samples for phenotype chr{pheno_chrom} x genotype chr{geno_chrom}, skipping.",
                        flush=True,
                    )
                    continue

                phenotype_df_final = phenotype_df_filtered[shared].astype(float)
                genotype_df_final = genotype_df_chr[shared]
                covariates_df_final = covariates_df.loc[shared]
                genotype_df_final, variant_df_chr = apply_mac_filter(
                    genotype_df_final, variant_df_chr, n_samples=len(shared), mac_min=args.MAC)
                if genotype_df_final.empty:
                    print(
                        f"  No variants left after MAC filter for phenotype chr{pheno_chrom} x genotype chr{geno_chrom}.",
                        flush=True,
                    )
                    continue

                trans_df = trans.map_trans(
                    genotype_df_final,
                    phenotype_df_final,
                    covariates_df_final,
                    batch_size=args.batch_size,
                    return_sparse=True,
                    return_r2=True,
                    pval_threshold=args.pval_threshold,
                    maf_threshold=args.maf_threshold,
                )
                if trans_df is None or trans_df.empty:
                    print(f"  No trans-QTLs found for phenotype chr{pheno_chrom} x genotype chr{geno_chrom}.", flush=True)
                    continue

                trans_df = trans.filter_cis(trans_df, phenotype_pos_df_filtered, variant_df_chr, window=effective_window)
                if trans_df is None or trans_df.empty:
                    print(
                        f"  No trans-QTLs found after cis filtering for phenotype chr{pheno_chrom} x genotype chr{geno_chrom}.",
                        flush=True,
                    )
                    continue

                trans_df.rename(columns={
                    "phenotype_id": "molecular_trait_id",
                    "pval": "pvalue",
                    "b": "bhat",
                    "b_se": "sebhat",
                }, inplace=True)
                trans_df["n"] = len(shared)
                trans_df = variant_df_chr.merge(trans_df, right_on="variant_id", left_index=True)
                trans_df.rename(columns={"a1": "a2", "a0": "a1"}, inplace=True)
                trans_df["pheno_chrom"] = pheno_chrom
                trans_df["geno_chrom"] = geno_chrom
                all_results.append(trans_df)
                print(
                    f"  Trans phenotype chr{pheno_chrom} x genotype chr{geno_chrom}: {len(trans_df)} pairs",
                    flush=True,
                )

    output_gz = args.output or os.path.join(args.cwd, "trans_qtl.pairs.tsv.gz")
    os.makedirs(os.path.dirname(os.path.abspath(output_gz)), exist_ok=True)
    output_tsv = strip_suffix(output_gz, ".gz")

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        lambda_col = combined_results.groupby("molecular_trait_id").apply(
            lambda x: stats.chi2.ppf(1.0 - np.median(x["pvalue"]), 1) / stats.chi2.ppf(0.5, 1)
        ).reset_index()
        lambda_col.columns = ["molecular_trait_id", "genomic_inflation_lambda"]
        inflation_prefix = strip_suffix(strip_suffix(strip_suffix(output_gz, ".gz"), ".tsv"), ".pairs")
        lambda_col.to_csv(
            f"{inflation_prefix}.genomic_inflation.tsv.gz",
            sep="\t",
            index=False,
            compression={"method": "gzip", "compresslevel": 9},
        )

        combined_results = combined_results.sort_values(by=["chrom", "pos", "molecular_trait_id"])
        if args.pval > 0:
            initial_n = len(combined_results)
            pval_percentiles = np.percentile(combined_results["pvalue"], np.arange(0, 110, 10))
            combined_results = combined_results[combined_results["pvalue"] < args.pval]
            summary_df = pd.DataFrame({
                "metric": ["initial_n", "after_filtering"] + [f"{i}%" for i in range(0, 110, 10)],
                "value": [initial_n, len(combined_results)] + list(pval_percentiles),
            })
            summary_prefix = strip_suffix(output_tsv, ".tsv")
            summary_df.to_csv(f"{summary_prefix}.summary.tsv", sep="\t", index=False)
    else:
        combined_results = pd.DataFrame(columns=[
            "chrom", "pos", "variant_id", "a1", "a2", "molecular_trait_id",
            "pvalue", "bhat", "sebhat", "r2", "af", "n", "pheno_chrom", "geno_chrom",
        ])

    combined_results.to_csv(output_tsv, sep="\t", index=False)
    apply_trans_qvalues(output_tsv)
    run_command(["bgzip", "--compress-level", "9", "-f", output_tsv])
    run_command(["tabix", "-S", "1", "-s", "1", "-b", "2", "-e", "2", output_gz])
    print(f"\nTRANS QTL complete. Results: {output_gz}", flush=True)


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
    p.add_argument("--covariate-pattern", nargs="*", default=[],
                   help="Covariate prefixes or exact names to retain")
    p.add_argument("--cwd", default="output", metavar="DIR",
                   help="Output directory")
    p.add_argument("--output", default="", metavar="PATH",
                   help="Nominal output path for --step trans")
    p.add_argument("--region-list", default="", metavar="PATH",
                   help="Optional list of phenotypes/regions to analyze")
    p.add_argument("--region-list-phenotype-column", type=int, default=4,
                   help="One-based phenotype ID column in --region-list")
    p.add_argument("--keep-sample", default="", metavar="PATH",
                   help="Optional one-column sample keep list")
    p.add_argument("--interaction", default="",
                   help="Interaction file or covariate column name for cis-QTL")
    p.add_argument("--customized-cis-windows", default="", metavar="PATH",
                   help="Optional chr/start/end/phenotype custom cis window table")
    p.add_argument("--phenotype-group", default="", metavar="PATH",
                   help="Optional phenotype-to-group mapping")
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
    p.add_argument("--batch-size", type=int, default=10000,
                   help="TensorQTL trans batch size")
    p.add_argument("--pval-threshold", type=float, default=1.0,
                   help="TensorQTL trans sparse p-value threshold")
    p.add_argument("--pval", type=float, default=0.0,
                   help="Optional post-qvalue trans p-value filter")
    p.add_argument("--pvalue-cutoff", default="",
                   help="Compatibility option from the notebook interface")
    p.add_argument("--qvalue-cutoff", default="",
                   help="Compatibility option from the notebook interface")
    p.add_argument("--skip-nominal-if-exist", action="store_true", default=False,
                   help="Reuse an existing nominal parquet file if present")
    p.add_argument("--permutation", type=str2bool, default=True,
                   help="Whether to run cis permutation/regional association")
    p.add_argument("--skip-postprocess", action="store_true", default=False,
                   help="For --step cis, produce per-chromosome outputs only; run --step cis_postprocess separately.")
    p.add_argument("--output-tsv", default="", metavar="PATH",
                   help="Output table for --step cis_postprocess")
    p.add_argument("--output-summary", default="", metavar="PATH",
                   help="Summary table for --step cis_postprocess")
    p.add_argument("--regional-files", nargs="*", default=[],
                   help="Regional cis-QTL files for --step cis_postprocess")
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
