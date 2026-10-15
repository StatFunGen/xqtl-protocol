#!/usr/bin/env python3
"""Split a LeafCutter intron count table into one table per chromosome.

leafcutter_ds.R tests every cluster on its own, so the clusters can be tested in
separate jobs and the results joined afterwards. Only the multiple-testing
correction needs every cluster at once; leafcutter_ds_merge.py recomputes it.

Rows go to parts 1-22 and X by their chromosome (a leading "chr" is ignored) and
everything else (Y, MT, unplaced contigs) to "other". All 24 parts are always
written, with the header, so the set of output files is fixed in advance; a part
with no clusters holds the header only.

With --groups, the groups file is checked against the count table first, so a
naming mismatch stops the run here rather than in every per-part job.

Usage: leafcutter_ds_split.py --counts <perind_numers.counts.gz> --output-prefix <dir/prefix> [--groups <groups.txt>]
Writes <dir/prefix>.part_<name>.counts.gz for each name in PARTS.
"""
import argparse
import collections
import gzip
import os

PARTS = [str(i) for i in range(1, 23)] + ["X", "other"]


def part_of(intron_id):
    chrom = intron_id.split(":", 1)[0]
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom if chrom in PARTS[:-1] else "other"


def check_groups(groups, header):
    """Fail early on the mistakes leafcutter_ds.R would otherwise hit in every part."""
    columns = set(header.split())
    rows = [l.split() for l in open(groups) if l.strip()]
    if any(len(r) < 2 for r in rows):
        raise SystemExit(f"ERROR: {groups} needs at least two columns: sample and group")
    used = [r for r in rows if r[0] in columns]
    if not used:
        raise SystemExit(
            f"ERROR: no sample in {groups} matches a column of the count table. "
            f"Column names look like: {' '.join(sorted(columns)[:3])}")
    labels = sorted({r[1] for r in used})
    if len(labels) != 2:
        raise SystemExit(f"ERROR: {groups} gives {len(labels)} groups ({', '.join(labels)}) "
                         "among the matched samples; leafcutter_ds compares exactly two")
    n = collections.Counter(r[1] for r in used)
    print(f"groups: {len(used)} of {len(rows)} samples matched; "
          + ", ".join(f"{g}={n[g]}" for g in labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counts", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--groups")
    a = ap.parse_args()
    d = os.path.dirname(a.output_prefix)
    if d:
        os.makedirs(d, exist_ok=True)
    n = collections.Counter()
    with gzip.open(a.counts, "rt") as f:
        header = f.readline()
        if not header.strip():
            raise SystemExit(f"ERROR: {a.counts} is empty")
        if a.groups:
            check_groups(a.groups, header)
        out = {p: gzip.open(f"{a.output_prefix}.part_{p}.counts.gz", "wt") for p in PARTS}
        try:
            for o in out.values():
                o.write(header)
            for line in f:
                if not line.strip():
                    continue
                p = part_of(line)
                out[p].write(line)
                n[p] += 1
        finally:
            for o in out.values():
                o.close()
    print("introns per part:", " ".join(f"{p}={n[p]}" for p in PARTS))
    print("total", sum(n.values()))


if __name__ == "__main__":
    main()
