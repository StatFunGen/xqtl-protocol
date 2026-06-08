#!/usr/bin/env python3
"""
gene_annotation.py
Mirrors selected task cells from gene_annotation.ipynb.

Steps (selected via --step):
  annotate_coord
  map_leafcutter_cluster_to_gene
  annotate_leafcutter_isoforms
  annotate_psichomics_isoforms
"""

import argparse
import gzip
import os
import subprocess
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_qtl_io():
    import qtl.io

    return qtl.io


def load_qtl_annotation():
    import qtl.annotation

    return qtl.annotation


def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def run_command(args):
    print("+", " ".join(str(x) for x in args), flush=True)
    subprocess.run(args, check=True)


def maybe_normalize_gene_type_gtf(annotation_gtf):
    """Return a path whose gene records expose gene_type, plus optional temp path."""
    has_gene_type = True
    with open_text(annotation_gtf) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            if len(row) < 9:
                continue
            if row[2] == "gene":
                has_gene_type = "gene_type" in line
                break
    if has_gene_type:
        return str(annotation_gtf), None

    with open_text(annotation_gtf) as handle:
        contents = handle.read().replace("gene_biotype", "gene_type")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".gtf", delete=False)
    try:
        tmp.write(contents)
        tmp.close()
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise
    return tmp.name, tmp.name


def gtf_to_bed(annotation_gtf, feature="gene", exclude_chrs=None, phenotype_id="gene_id"):
    """
    Parse a GTF into a BED-like dataframe with min start and max end per identifier.
    """
    exclude_chrs = exclude_chrs or []
    gene_data = defaultdict(lambda: {"chr": "", "start": float("inf"), "end": 0, "strand": ""})
    gene_names = {}

    with open_text(annotation_gtf) as gtf:
        for row in gtf:
            row = row.strip().split("\t")
            if not row or row[0].startswith("#") or len(row) < 9 or row[2] != feature:
                continue

            attributes = defaultdict(list)
            for attr in row[8].replace('"', "").split(";")[:-1]:
                parts = attr.strip().split(" ")
                if not parts:
                    continue
                key = parts[0]
                value = parts[1] if len(parts) > 1 else ""
                if key == "tag":
                    attributes["tags"].append(value)
                else:
                    attributes[key] = value

            curr_gene_id = attributes["gene_id"]
            curr_gene_name = attributes.get("gene_name", curr_gene_id)
            gene_names[curr_gene_id] = curr_gene_name

            data = gene_data[curr_gene_id]
            if not data["chr"]:
                data["chr"] = row[0]
                data["strand"] = row[6]

            start_pos = int(row[3]) - 1
            end_pos = int(row[4]) - 1
            data["start"] = min(data["start"], start_pos)
            data["end"] = max(data["end"], end_pos)

    chrom, start, end, ids, names, strand = [], [], [], [], [], []
    for gene_id, data in gene_data.items():
        chrom.append(data["chr"])
        start.append(data["start"])
        end.append(data["end"])
        ids.append(gene_id)
        names.append(gene_names[gene_id])
        strand.append(data["strand"])

    if phenotype_id == "gene_id":
        bed_df = pd.DataFrame(
            {"chr": chrom, "start": start, "end": end, "gene_id": ids, "strand": strand},
            columns=["chr", "start", "end", "gene_id", "strand"],
            index=ids,
        )
    elif phenotype_id == "gene_name":
        bed_df = pd.DataFrame(
            {"chr": chrom, "start": start, "end": end, "gene_id": names, "strand": strand},
            columns=["chr", "start", "end", "gene_id", "strand"],
            index=names,
        )
    else:
        raise ValueError(f"Unsupported phenotype_id: {phenotype_id}")

    for chrom_name in exclude_chrs:
        bed_df = bed_df[bed_df["chr"] != chrom_name]
    bed_df = sort_by_chrom_position(bed_df, "chr")
    return bed_df


def sort_by_chrom_position(df, chrom_col):
    if df.empty:
        return df
    return pd.concat(
        [group.sort_values(["start", "end"]) for _, group in df.groupby(chrom_col, sort=False)],
        axis=0,
    )


def prepare_bed(df, bed_template_df, chr_subset=None):
    bed_df = pd.merge(bed_template_df, df, left_index=True, right_index=True)
    bed_df = sort_by_chrom_position(bed_df, "#chr")
    if chr_subset is not None:
        bed_df = bed_df[bed_df["#chr"].isin(chr_subset)]
    return bed_df


def load_and_preprocess_data(input_path, drop_columns, sep):
    df = pd.read_csv(input_path, sep=sep, skiprows=0)
    removable = [col for col in df.columns if col in drop_columns]
    df = df.drop(removable, axis=1)
    if len(df.columns) < 2:
        raise ValueError(
            "There are too few columns in the loaded dataframe; check the input delimiter."
        )
    return df


def rename_samples_using_lookup(df, lookup_path):
    lookup_file = Path(lookup_path)
    if not lookup_file.is_file():
        return df

    if lookup_file.suffix == ".csv":
        sep = ","
    else:
        sep = "\t"

    try:
        lookup_df = pd.read_csv(lookup_file, sep=sep, dtype=str)
        if "genotype_id" in lookup_df.columns and lookup_df.shape[1] >= 2:
            lookup_df = lookup_df.set_index(lookup_df.columns[1])
            df.rename(columns=lookup_df.to_dict()["genotype_id"], inplace=True)
        else:
            lookup_df = pd.read_csv(lookup_file, sep=sep, header=None, dtype=str)
            rename_dict = dict(zip(lookup_df[0], lookup_df[1]))
            df.rename(columns=rename_dict, inplace=True)
    except Exception as exc:
        print(f"Warning: Error processing sample lookup file: {exc}", flush=True)
    return df


def load_bed_template_from_gtf(input_path, phenotype_id_column):
    if gtf_to_bed(input_path, feature="gene", phenotype_id="gene_id").index.duplicated().sum() > 0:
        raise ValueError(
            f"gtf file {input_path} needs to be collapsed into gene model by reference data processing module"
        )

    bed_template_df_id = gtf_to_bed(input_path, feature="transcript", phenotype_id="gene_id")
    bed_template_df_name = gtf_to_bed(input_path, feature="transcript", phenotype_id="gene_name")
    bed_template_df = bed_template_df_id.merge(
        bed_template_df_name, on=["chr", "start", "end", "strand"]
    )
    bed_template_df.columns = ["#chr", "start", "end", "gene_id", "strand", "gene_name"]
    if phenotype_id_column not in bed_template_df.columns:
        raise ValueError(f"{phenotype_id_column} is not available in the GTF-derived template")
    return bed_template_df.set_index(phenotype_id_column, drop=False)


def derive_gene_list_output(pheno_file, output_bed, explicit_path=""):
    if explicit_path:
        return explicit_path
    stem = Path(pheno_file).name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    return str(Path(output_bed).with_name(f"{stem}.gene_list.tsv"))


def annotate_coord(args):
    qtl_io = load_qtl_io()
    drop_cols = ["#chr", "chr", "start", "end", "stop", "annot.seqnames", "annot.start", "annot.end"]
    df = load_and_preprocess_data(args.phenoFile, drop_cols, args.sep)
    phenotype_id = df.columns[0]
    df.set_index(phenotype_id, inplace=True)

    if args.sample_participant_lookup:
        df = rename_samples_using_lookup(df, args.sample_participant_lookup)

    if args.molecular_trait_type == "gene":
        if args.strip_id:
            df.index = df.index.map(lambda x: x.split("|")[-1].strip() if "|" in x else x)
        bed_template_df = load_bed_template_from_gtf(args.coordinate_annotation, args.phenotype_id_column)
        bed_df = prepare_bed(df, bed_template_df).drop("gene_name", axis=1)
        bed_df = bed_df.drop_duplicates("gene_id", keep=False).rename(columns={"gene_id": "ID"})
    elif args.molecular_trait_type == "protein":
        expr_df = df.reset_index()
        aux_path = Path(args.auxiliary_id_mapping) if args.auxiliary_id_mapping else None
        if aux_path is not None and aux_path.is_file():
            df_info = pd.read_csv(aux_path).rename(
                columns={phenotype_id: phenotype_id, "EntrezGeneSymbol": "gene_name"}
            )[["gene_name", phenotype_id, "UniProt"]]
            expr_df = df_info.merge(expr_df, on=phenotype_id).drop(phenotype_id, axis=1)
        else:
            split_df = expr_df[phenotype_id].astype(str).str.split("|", n=1, expand=True)
            expr_df["gene_name"] = split_df[0]
            expr_df["UniProt"] = split_df[1]
            expr_df = expr_df.drop(columns=[phenotype_id])
        expr_df = expr_df.set_index("gene_name", drop=False)
        bed_template_df = load_bed_template_from_gtf(args.coordinate_annotation, args.phenotype_id_column)
        bed_df = prepare_bed(expr_df, bed_template_df)
        bed_df["ID"] = bed_df["gene_id"] + "_" + bed_df["UniProt"]
        keep_cols = ["#chr", "start", "end", "ID"] + expr_df.drop(["UniProt"], axis=1).columns.tolist()
        bed_df = bed_df.drop_duplicates("ID", keep=False)[keep_cols]
    elif args.molecular_trait_type == "atac":
        bed_template_df = pd.read_csv(args.coordinate_annotation, sep="\t").set_index("ID")
        bed_template_df = bed_template_df.assign(ID=bed_template_df.index)
        if "chr" in bed_template_df.columns and "#chr" not in bed_template_df.columns:
            bed_template_df = bed_template_df.rename(columns={"chr": "#chr"})
        bed_df = prepare_bed(df, bed_template_df).drop_duplicates("ID", keep=False)
    else:
        raise ValueError(f"Unsupported molecular_trait_type: {args.molecular_trait_type}")

    os.makedirs(Path(args.output_bed).parent, exist_ok=True)
    qtl_io.write_bed(bed_df, args.output_bed)
    region_cols = ["#chr", "start", "end", "ID"]
    if "strand" in bed_df.columns:
        region_cols.append("strand")
    bed_df[region_cols].assign(path=args.output_bed).to_csv(
        args.output_region_list, sep="\t", index=False
    )

    if args.molecular_trait_type in {"gene", "protein"}:
        gene_list_output = derive_gene_list_output(args.phenoFile, args.output_bed, args.gene_list_output)
        region_list = bed_template_df[bed_template_df[args.phenotype_id_column].isin(df.index)]
        region_list.to_csv(gene_list_output, sep="\t", index=False)
    else:
        warnings.warn(
            "Gene partitioning is not applicable for ATAC; no gene_list.tsv will be generated."
        )


def extract_exon_table(annotation_gtf):
    gtf_path, temp_path = maybe_normalize_gene_type_gtf(annotation_gtf)
    qtl_annotation = load_qtl_annotation()
    try:
        annot = qtl_annotation.Annotation(gtf_path)
        rows = []
        for gene in annot.genes:
            if not gene.transcripts:
                continue
            for exon in gene.transcripts[0].exons:
                rows.append([gene.chr, exon.start_pos, exon.end_pos, gene.strand, gene.id, gene.name])
        exon_df = pd.DataFrame(
            rows, columns=["chr", "start", "end", "strand", "gene_id", "gene_name"]
        )
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
    return exon_df


def get_intron_meta(introns):
    parts = [str(x).split(":") for x in introns]
    intron_meta = pd.DataFrame(parts)
    if intron_meta.shape[1] == 5:
        intron_meta.columns = ["chr", "start", "end", "clu", "category"]
    else:
        intron_meta.columns = ["chr", "start", "end", "clu"]
    intron_meta["start"] = pd.to_numeric(intron_meta["start"])
    intron_meta["end"] = pd.to_numeric(intron_meta["end"])
    return intron_meta


def harmonize_chr_style(intron_meta, exon_table):
    intron_has_chr = str(intron_meta.iloc[0]["chr"]).startswith("chr")
    exon_has_chr = str(exon_table.iloc[0]["chr"]).startswith("chr")
    if not intron_has_chr and exon_has_chr:
        exon_table = exon_table.copy()
        exon_table["chr"] = exon_table["chr"].astype(str).str.replace("^chr", "", regex=True)
    elif intron_has_chr and not exon_has_chr:
        exon_table = exon_table.copy()
        exon_table["chr"] = "chr" + exon_table["chr"].astype(str)
    return exon_table


def map_clusters_to_genes_site(intron_meta, exon_table):
    pieces = []
    for chrom in sorted(intron_meta["chr"].dropna().unique()):
        intron_chr = intron_meta[intron_meta["chr"] == chrom].copy()
        exons_chr = exon_table[exon_table["chr"] == chrom].copy()
        if intron_chr.empty or exons_chr.empty:
            continue

        exons_start = exons_chr[["start", "gene_name"]].rename(columns={"start": "temp"})
        intron_end = intron_chr[["end", "clu"]].rename(columns={"end": "temp"})
        three_prime = intron_end.merge(exons_start, on="temp", how="inner")

        exons_end = exons_chr[["end", "gene_name"]].rename(columns={"end": "temp"})
        intron_start = intron_chr[["start", "clu"]].rename(columns={"start": "temp"})
        five_prime = intron_start.merge(exons_end, on="temp", how="inner")

        all_matches = pd.concat([three_prime, five_prime], ignore_index=True)
        if all_matches.empty:
            continue
        all_matches = all_matches[["clu", "gene_name"]].drop_duplicates()
        all_matches["clu"] = chrom + ":" + all_matches["clu"].astype(str)
        pieces.append(all_matches)

    if not pieces:
        return pd.DataFrame(columns=["clu", "genes"])

    gene_df = pd.concat(pieces, ignore_index=True)
    return (
        gene_df.groupby("clu", as_index=False)["gene_name"]
        .apply(lambda x: ",".join(x.astype(str)))
        .rename(columns={"gene_name": "genes"})
    )


def map_clusters_to_genes_region(intron_meta, exon_table, overlap_ratio):
    merged_df = (
        exon_table.groupby(["chr", "gene_id"], as_index=False)
        .agg({"start": "min", "end": "max"})
        [["chr", "start", "end", "gene_id"]]
    )
    results = []
    for chrom in sorted(intron_meta["chr"].dropna().unique()):
        intron_chr = intron_meta[intron_meta["chr"] == chrom]
        gene_chr = merged_df[merged_df["chr"] == chrom]
        if intron_chr.empty or gene_chr.empty:
            continue
        for _, intron in intron_chr.iterrows():
            intron_len = max(int(intron["end"]) - int(intron["start"]), 1)
            genes = []
            for _, gene in gene_chr.iterrows():
                overlap = max(
                    0,
                    min(int(intron["end"]), int(gene["end"])) - max(int(intron["start"]), int(gene["start"])),
                )
                if overlap / intron_len >= overlap_ratio:
                    genes.append(str(gene["gene_id"]))
            if genes:
                results.append({"clu": f'{chrom}:{intron["clu"]}', "genes": ",".join(dict.fromkeys(genes))})
    return pd.DataFrame(results, columns=["clu", "genes"])


def map_leafcutter_cluster_to_gene(args):
    exon_df = extract_exon_table(args.annotation_gtf)
    exon_df.to_csv(args.output_exon_list, sep="\t", index=False)

    intron_counts = pd.read_csv(args.intron_count, sep="\t", header=0, dtype=str)
    intron_meta = get_intron_meta(intron_counts.iloc[:, 3].tolist())
    exon_table = pd.read_csv(args.output_exon_list, sep="\t", header=0, dtype={"chr": str})
    exon_table = harmonize_chr_style(intron_meta, exon_table)

    if "gene_id" not in exon_table.columns:
        raise ValueError("gene_id must be present in the exon table")
    exon_table["gene_name"] = exon_table["gene_id"]

    if args.map_stra == "site":
        mapping = map_clusters_to_genes_site(intron_meta, exon_table)
    elif args.map_stra == "region":
        mapping = map_clusters_to_genes_region(intron_meta, exon_table, args.overlap_ratio)
    else:
        raise ValueError("map_stra must be 'site' or 'region'")

    mapping.to_csv(args.output_cluster_map, sep="\t", index=False)


def read_lookup_participant_mapping(lookup_path):
    if not lookup_path or not Path(lookup_path).is_file():
        return {}
    lookup_df = pd.read_csv(lookup_path, sep="\t", index_col=0, dtype=str)
    if "participant_id" not in lookup_df.columns:
        return {}
    return dict(zip(lookup_df.index, lookup_df["participant_id"]))


def load_tss_df(annotation_gtf, feature="gene", phenotype_id="gene_id"):
    gtf_path, temp_path = maybe_normalize_gene_type_gtf(annotation_gtf)
    qtl_io = load_qtl_io()
    try:
        return qtl_io.gtf_to_tss_bed(gtf_path, feature=feature, phenotype_id=phenotype_id)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def annotate_leafcutter_isoforms(args):
    qtl_io = load_qtl_io()
    tss_df = load_tss_df(args.annotation_gtf)
    bed_df = pd.read_csv(args.phenoFile, sep="\t", skiprows=0)
    bed_df.columns.values[0] = "#chr"
    cluster2gene_dict = pd.read_csv(args.cluster_map, sep="\t", index_col=0).to_dict().get("genes", {})
    sample_mapping = read_lookup_participant_mapping(args.sample_participant_lookup)

    print("    ** assigning introns to gene mapping(s)", flush=True)
    discarded = 0
    gene_bed_rows = []
    group_s = {}
    for _, row in bed_df.iterrows():
        parts = str(row["ID"]).split(":")
        if len(parts) < 4:
            discarded += 1
            continue
        cluster_id = parts[0] + ":" + parts[3]
        if cluster_id not in cluster2gene_dict:
            discarded += 1
            continue
        gene_ids = [x for x in str(cluster2gene_dict[cluster_id]).split(",") if x]
        for gene_id in gene_ids:
            if gene_id not in tss_df.index:
                discarded += 1
                continue
            gene_isoform_id = str(row["ID"]) + ":" + gene_id
            gene_bed_rows.append(
                tss_df.loc[gene_id, ["chr", "start", "end"]].tolist() + [gene_isoform_id] + row.iloc[4:].tolist()
            )
            group_s[gene_isoform_id] = gene_id

    if discarded > 0:
        print(f"    ** discarded {discarded} introns without a gene mapping", flush=True)

    print("  * writing BED files for QTL mapping", flush=True)
    gene_bed_df = pd.DataFrame(gene_bed_rows, columns=bed_df.columns)
    gene_bed_df = gene_bed_df.groupby("#chr", sort=False, group_keys=False).apply(
        lambda x: x.sort_values("start")
    )
    gene_bed_df.columns = list(gene_bed_df.columns[:4]) + [
        name.split(".")[0] if "junc" in name else name for name in gene_bed_df.columns[4:]
    ]

    if sample_mapping:
        column_names = gene_bed_df.columns[4:]
        new_column_names = [sample_mapping.get(col, col) for col in column_names]
        gene_bed_df.rename(columns=dict(zip(column_names, new_column_names)), inplace=True)

    gene_bed_df = gene_bed_df.drop_duplicates()
    qtl_io.write_bed(gene_bed_df, args.output_bed)

    group_s_df = pd.Series(group_s).sort_values().reset_index()
    group_s_df.columns = ["ID", "gene"]
    group_s_df.to_csv(args.output_phenotype_group, sep="\t", index=False, header=True)


def annotate_psichomics_isoforms(args):
    qtl_io = load_qtl_io()
    tss_df = load_tss_df(args.annotation_gtf, feature="gene", phenotype_id="gene_id")
    bed_df = pd.read_csv(args.phenoFile, sep="\t", skiprows=0)
    bed_df["gene_id"] = bed_df["ID"].astype(str).str.split("_").map(lambda x: x[-1])
    drop_cols = [col for col in ["#Chr", "#chr", "start", "end"] if col in bed_df.columns]
    if drop_cols:
        bed_df = bed_df.drop(columns=drop_cols)

    output = tss_df.merge(bed_df, on="gene_id", how="right").sort_values(["chr", "start"])
    sample_mapping = read_lookup_participant_mapping(args.sample_participant_lookup)
    if sample_mapping:
        column_names = output.columns[4:]
        new_column_names = [sample_mapping.get(col, col) for col in column_names]
        output.rename(columns=dict(zip(column_names, new_column_names)), inplace=True)

    output = output.assign(
        gene_type_id=output["ID"].astype(str).str.split("_").map(lambda x: x[-1] + "_" + x[0])
    )
    phenotype_group = output[["ID", "gene_type_id"]]
    bed_output = output.drop("gene_id", axis=1)
    if "chr" in bed_output.columns and "#chr" not in bed_output.columns:
        bed_output = bed_output.rename(columns={"chr": "#chr"})

    qtl_io.write_bed(bed_output, args.output_bed)
    phenotype_group.to_csv(args.output_phenotype_group, sep="\t", index=False, header=False)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Gene annotation helpers extracted from gene_annotation.ipynb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "annotate_coord",
            "map_leafcutter_cluster_to_gene",
            "annotate_leafcutter_isoforms",
            "annotate_psichomics_isoforms",
        ],
    )

    parser.add_argument("--phenoFile", default="")
    parser.add_argument("--coordinate-annotation", default="")
    parser.add_argument("--sample-participant-lookup", default="")
    parser.add_argument("--molecular-trait-type", default="gene")
    parser.add_argument("--phenotype-id-column", default="gene_id")
    parser.add_argument("--auxiliary-id-mapping", default="")
    parser.add_argument("--strip-id", action="store_true", default=False)
    parser.add_argument("--sep", default="\t")
    parser.add_argument("--output-bed", default="")
    parser.add_argument("--output-region-list", default="")
    parser.add_argument("--gene-list-output", default="")

    parser.add_argument("--intron-count", default="")
    parser.add_argument("--annotation-gtf", default="")
    parser.add_argument("--map-stra", default="site")
    parser.add_argument("--overlap-ratio", type=float, default=0.8)
    parser.add_argument("--output-exon-list", default="")
    parser.add_argument("--output-cluster-map", default="")
    parser.add_argument("--cluster-map", default="")
    parser.add_argument("--output-phenotype-group", default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.step == "annotate_coord":
        required = ["phenoFile", "coordinate_annotation", "output_bed", "output_region_list"]
    elif args.step == "map_leafcutter_cluster_to_gene":
        required = ["intron_count", "annotation_gtf", "output_exon_list", "output_cluster_map"]
    elif args.step == "annotate_leafcutter_isoforms":
        required = ["phenoFile", "annotation_gtf", "cluster_map", "output_bed", "output_phenotype_group"]
    else:
        required = ["phenoFile", "annotation_gtf", "output_bed", "output_phenotype_group"]

    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing required arguments for step {args.step}: {', '.join('--' + x.replace('_', '-') for x in missing)}")

    if args.step == "annotate_coord":
        annotate_coord(args)
    elif args.step == "map_leafcutter_cluster_to_gene":
        map_leafcutter_cluster_to_gene(args)
    elif args.step == "annotate_leafcutter_isoforms":
        annotate_leafcutter_isoforms(args)
    elif args.step == "annotate_psichomics_isoforms":
        annotate_psichomics_isoforms(args)


if __name__ == "__main__":
    main()
