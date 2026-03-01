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
import sys
from pathlib import Path


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
            norm_counts = qtl.norm.voom(counts_df)
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

        # Map sample IDs to participant IDs
        norm_df.columns = [
            sample_participant_df.loc[s, "participant_id"]
            if s in sample_participant_df.index else s
            for s in norm_df.columns
        ]

        # Load gene annotations
        gene_df = qtl.io.gtf_to_bed(annotation_gtf, feature="gene")
        if chrs is not None:
            gene_df = gene_df[gene_df["chr"].isin(chrs)]

        # Build output BED
        bed_df = gene_df.merge(
            norm_df, left_on="gene_id", right_index=True, how="inner")
        bed_df.rename(columns={"chr": "#chr"}, inplace=True)

        no_qnorm_part = ".no_qnorm" if qnorm == "none" else ""
        out_name = f"{prefix}.{normalization_method}{no_qnorm_part}.expression.bed.gz"
        out_path = os.path.join(output_dir, out_name)
        bed_df.to_csv(out_path, sep="\t", index=False)
        os.system(f"tabix -p bed {out_path}")
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
