#!/usr/bin/env python3
"""
bulk_expression_normalization.py
Mirrors: code/molecular_phenotypes/QC/bulk_expression_normalization.ipynb

Steps (selected via --step):
  normalize — TMM/CPM/voom (or quantile) normalization of bulk expression data

This implementation is adapted from the GTEx pipeline's
eqtl_prepare_expression.py script.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def get_args():
    p = argparse.ArgumentParser(description="Bulk expression normalization")
    p.add_argument("--step", required=True, choices=["normalize"],
                   help="Step to run: normalize")
    p.add_argument("--cwd", default="output", help="Output directory")
    p.add_argument("--tpm-gct", required=True, help="TPM GCT.gz file")
    p.add_argument("--counts-gct", required=True, help="Raw counts GCT.gz file")
    p.add_argument("--annotation-gtf", required=True, help="Gene annotation GTF file")
    p.add_argument("--sample-participant-lookup", required=True,
                   help="TSV mapping sample_id -> participant_id")
    p.add_argument("--tpm-threshold", type=float, default=0.1)
    p.add_argument("--count-threshold", type=float, default=6)
    p.add_argument("--sample-frac-threshold", type=float, default=0.2)
    p.add_argument("--normalization-method", default="tmm_cpm_voom",
                   choices=["tmm_cpm_voom", "tmm_cpm_edger", "qn"])
    p.add_argument("--quantile-normalize", action="store_true", default=True)
    p.add_argument("--no-quantile-normalize", dest="quantile_normalize",
                   action="store_false")
    p.add_argument("--numThreads", type=int, default=20)
    return p.parse_args()


def gtf_to_gene_bed_fallback(gtf_path):
    import pandas as pd

    rows = {}
    with open(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in {"gene", "exon"}:
                continue
            attrs = {}
            for item in fields[8].strip().split(";"):
                item = item.strip()
                if not item:
                    continue
                key, _, value = item.partition(" ")
                attrs[key] = value.strip().strip('"')
            gene_id = attrs.get("gene_id")
            if gene_id is None:
                continue
            start = int(fields[3]) - 1
            end = int(fields[4])
            if gene_id not in rows:
                rows[gene_id] = {
                    "chr": fields[0],
                    "start": start,
                    "end": end,
                    "gene_id": gene_id,
                }
            else:
                rows[gene_id]["start"] = min(rows[gene_id]["start"], start)
                rows[gene_id]["end"] = max(rows[gene_id]["end"], end)
    return pd.DataFrame(rows.values(), columns=["chr", "start", "end", "gene_id"])


def normalize_ensembl_gene_ids(gene_ids):
    """Strip Ensembl version suffixes while preserving optional PAR_Y labels."""
    return gene_ids.astype(str).str.replace(r"\.(\d+)(?=_PAR_Y$|$)", "", regex=True)


def sort_bed_for_tabix(bed_df):
    """Sort BED records in the order required by tabix indexing."""
    chrom = bed_df["#chr"].astype(str).str.replace(r"^chr", "", regex=True)
    chrom_rank = {str(i): i for i in range(1, 23)}
    chrom_rank.update({"X": 23, "Y": 24, "M": 25, "MT": 25})

    sortable = bed_df.assign(
        _chrom_rank=chrom.str.upper().map(chrom_rank).fillna(1000).astype(int),
        _start=pd.to_numeric(bed_df["start"], errors="raise"),
        _end=pd.to_numeric(bed_df["end"], errors="raise"),
    )
    sortable = sortable.sort_values(
        ["_chrom_rank", "#chr", "_start", "_end", "gene_id"],
        kind="mergesort",
    )
    return sortable.drop(columns=["_chrom_rank", "_start", "_end"])


def run_normalize(args):
    import numpy as np
    import pandas as pd
    import qtl.io
    import qtl.norm

    os.makedirs(args.cwd, exist_ok=True)

    # Derive output prefix from tpm_gct filename
    bname = Path(args.tpm_gct).name
    for suffix in [".gct.gz", ".gct"]:
        if bname.endswith(suffix):
            bname = bname[:-len(suffix)]
            break
    for suffix in [".gene_tpm", ".tpm"]:
        if bname.endswith(suffix):
            bname = bname[:-len(suffix)]
            break

    qnorm = "qnorm" if args.quantile_normalize else "none"
    norm_method = args.normalization_method

    def prepare_expression(counts_df, tpm_df,
                           sample_frac_threshold=0.2,
                           count_threshold=6, tpm_threshold=0.1,
                           mode="tmm_cpm_voom", qnorm="none"):
        ns   = tpm_df.shape[1]
        mask = (
            (np.sum(tpm_df >= tpm_threshold,   axis=1) >= sample_frac_threshold * ns) &
            (np.sum(counts_df >= count_threshold, axis=1) >= sample_frac_threshold * ns)
        ).values

        mode_l = mode.lower()
        if mode_l == "tmm_cpm_edger":
            norm_counts = qtl.norm.edger_cpm(counts_df, normalized_lib_sizes=True)
        elif mode_l == "tmm_cpm_voom":
            voom_fn = getattr(qtl.norm, "voom", None) or getattr(qtl.norm, "voom_transform", None)
            if voom_fn is None:
                raise AttributeError("qtl.norm is missing voom/voom_transform")
            norm_counts = voom_fn(counts_df)
        elif mode_l == "qn":
            norm_counts = qtl.norm.quantile_normalize(counts_df)
        else:
            raise ValueError(f"Unknown normalization method: {mode}")

        norm_counts = norm_counts.loc[mask]
        if qnorm == "qnorm":
            norm_counts = qtl.norm.inverse_normal_transform(norm_counts)
        return norm_counts

    def eqtl_prepare_expression(tpm_gct, counts_gct, annotation_gtf,
                                 sample_to_participant, prefix,
                                 output_dir=".", sample_ids=None, chrs=None,
                                 tpm_threshold=0.1, count_threshold=6,
                                 sample_frac_threshold=0.2,
                                 normalization_method="tmm_cpm_voom",
                                 qnorm="none"):
        sample_participant_df = pd.read_csv(
            sample_to_participant, sep="\t", index_col=0)

        tpm_df    = qtl.io.read_gct(tpm_gct,    sample_ids=sample_ids)
        counts_df = qtl.io.read_gct(counts_gct, sample_ids=sample_ids)

        # Restrict to shared samples and participant mapping
        shared = tpm_df.columns.intersection(sample_participant_df.index)
        tpm_df    = tpm_df[shared]
        counts_df = counts_df[shared]

        norm_df = prepare_expression(
            counts_df, tpm_df,
            sample_frac_threshold=sample_frac_threshold,
            count_threshold=count_threshold,
            tpm_threshold=tpm_threshold,
            mode=normalization_method,
            qnorm=qnorm)

        # RNASeQC GCTs often carry versioned Ensembl IDs (for example ENSG....5)
        # while modular_sos gene annotations use unversioned gene IDs.
        norm_df.index = normalize_ensembl_gene_ids(pd.Index(norm_df.index))
        if norm_df.index.has_duplicates:
            dup_ids = norm_df.index[norm_df.index.duplicated()].unique().tolist()
            preview = ", ".join(dup_ids[:10])
            raise ValueError(
                "Duplicate gene IDs after stripping Ensembl versions; "
                f"cannot build BED safely. Examples: {preview}"
            )

        # Map sample IDs to participant IDs
        norm_df.columns = [
            sample_participant_df.loc[s, "participant_id"]
            if s in sample_participant_df.index else s
            for s in norm_df.columns
        ]

        # Load gene annotations
        gtf_to_bed = getattr(qtl.io, "gtf_to_bed", None)
        if gtf_to_bed is not None:
            gene_df = gtf_to_bed(annotation_gtf, feature="gene")
        else:
            gene_df = gtf_to_gene_bed_fallback(annotation_gtf)
        if chrs is not None:
            gene_df = gene_df[gene_df["chr"].isin(chrs)]

        # Build output BED
        bed_df = gene_df.merge(
            norm_df, left_on="gene_id", right_index=True, how="inner")
        if bed_df.empty:
            expr_ids = set(norm_df.index)
            annot_ids = set(gene_df["gene_id"])
            overlap = len(expr_ids & annot_ids)
            raise ValueError(
                "No expression genes overlapped the annotation GTF after gene ID "
                f"normalization (expression={len(expr_ids)}, annotation={len(annot_ids)}, "
                f"overlap={overlap})."
            )
        bed_df.rename(columns={"chr": "#chr"}, inplace=True)
        bed_df = sort_bed_for_tabix(bed_df)

        no_qnorm_part = ".no_qnorm" if qnorm == "none" else ""
        out_name = f"{prefix}.{normalization_method}{no_qnorm_part}.expression.bed.gz"
        out_path = Path(output_dir) / out_name
        out_plain = Path(str(out_path)[:-3])
        bed_df.to_csv(out_plain, sep="\t", index=False)
        subprocess.run(["bgzip", "-f", str(out_plain)], check=True)
        subprocess.run(["tabix", "-f", "-p", "bed", str(out_path)], check=True)
        print(f"Output: {out_path}", flush=True)

    eqtl_prepare_expression(
        tpm_gct=args.tpm_gct,
        counts_gct=args.counts_gct,
        annotation_gtf=args.annotation_gtf,
        sample_to_participant=args.sample_participant_lookup,
        prefix=bname,
        output_dir=args.cwd,
        tpm_threshold=args.tpm_threshold,
        count_threshold=args.count_threshold,
        sample_frac_threshold=args.sample_frac_threshold,
        normalization_method=args.normalization_method,
        qnorm=qnorm,
    )


def main():
    args = get_args()
    os.makedirs(args.cwd, exist_ok=True)
    if args.step == "normalize":
        run_normalize(args)
    else:
        print(f"Unknown step: {args.step}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
