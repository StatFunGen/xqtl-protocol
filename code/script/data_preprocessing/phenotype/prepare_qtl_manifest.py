#!/usr/bin/env python3
"""Build the single-context phenotype manifest for pecotmr's QtlDataset.

Every feature on the configured chromosomes points at the same phenotype BED:
``loadQtlDatasetFromManifest`` resolves one context to one phenotype matrix.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path


def _open_text(path: Path):
    if path.name.endswith((".gz", ".bgz")):
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def _canonical(chromosome: str) -> str:
    return re.sub(r"^chr", "", chromosome, flags=re.IGNORECASE).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic single-context QtlDataset manifest"
    )
    parser.add_argument("--bed", required=True, type=Path)
    parser.add_argument("--covariates", required=True, type=Path)
    parser.add_argument("--context", required=True)
    parser.add_argument("--chromosomes", required=True, nargs="+")
    parser.add_argument("--phenotype-id-column", default="ID")
    parser.add_argument("--phenotype-manifest", required=True, type=Path)
    parser.add_argument("--region-ids", required=True, type=Path)
    parser.add_argument("--sample-ids", required=True, type=Path)
    args = parser.parse_args()

    bed = args.bed.resolve()
    covariates = args.covariates.resolve()
    order = {_canonical(c): i for i, c in enumerate(args.chromosomes)}

    rows = []
    with _open_text(bed) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        expected = ["#chr", "start", "end", args.phenotype_id_column]
        if header[:4] != expected:
            raise ValueError(f"{bed} must begin with " + ", ".join(expected))
        for chromosome, start, end, feature_id in (row[:4] for row in reader):
            if _canonical(chromosome) in order:
                rows.append([chromosome, int(start), int(end), feature_id,
                             str(bed), args.context, str(covariates)])
    if not rows:
        raise ValueError(f"No features on the configured chromosomes: {bed}")
    rows.sort(key=lambda row: (order[_canonical(row[0])], row[1], row[2], row[3]))

    with _open_text(covariates) as handle:
        covariate_header = next(csv.reader(handle, delimiter="\t"))
    samples = sorted(set(header[4:]) & set(covariate_header[1:]))
    if not samples:
        raise ValueError("Phenotype BED and covariate file share no sample IDs")

    for path in [args.phenotype_manifest, args.region_ids, args.sample_ids]:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.phenotype_manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["#chr", "start", "end", "ID", "path", "cond", "cov_path"])
        writer.writerows(rows)
    args.region_ids.write_text("".join(f"{row[3]}\n" for row in rows))
    args.sample_ids.write_text("".join(f"{sample}\n" for sample in samples))


if __name__ == "__main__":
    main()
