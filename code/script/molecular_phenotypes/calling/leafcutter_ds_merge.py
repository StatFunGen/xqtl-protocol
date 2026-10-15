#!/usr/bin/env python3
"""Join per-part leafcutter_ds.R results and recompute the FDR over all clusters.

Each part was tested separately, so its p.adjust column was corrected only within
that part. The p-values themselves do not depend on the split, so the column is
recomputed here with the Benjamini-Hochberg method over every cluster that has a
p-value, as R's p.adjust(p, method = "fdr") does in an unsplit run. Clusters
without a p-value keep NA.

Usage: leafcutter_ds_merge.py --output-prefix <dir/prefix> <part>_cluster_significance.txt ...
Each significance file is read with its <part>_effect_sizes.txt sibling. Other
paths are ignored, so a step's whole output list can be passed. Empty files (parts
with no clusters) are skipped. Writes <prefix>_cluster_significance.txt and
<prefix>_effect_sizes.txt.
"""
import argparse
import os

SIG = "_cluster_significance.txt"
EFF = "_effect_sizes.txt"


def read_table(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    return (lines[0], lines[1:]) if lines else (None, [])


def bh(pvalues):
    """Benjamini-Hochberg adjusted values in input order, as R's p.adjust(method = "fdr")."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i], reverse=True)
    adj = [0.0] * m
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, pvalues[i] * m / rank)
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    sig_files = [f for f in a.files if f.endswith(SIG)]
    if not sig_files:
        raise SystemExit("ERROR: no *" + SIG + " files given")
    sig_hdr = eff_hdr = None
    rows, eff_rows, used = [], [], 0
    for s in sig_files:
        e = s[: -len(SIG)] + EFF
        if not os.path.exists(e):
            raise SystemExit(f"ERROR: {e} is missing")
        h, body = read_table(s)
        eh, ebody = read_table(e)
        if h is None:
            continue
        if sig_hdr is not None and h != sig_hdr:
            raise SystemExit(f"ERROR: {s} has a different header")
        if eh is not None and eff_hdr is not None and eh != eff_hdr:
            raise SystemExit(f"ERROR: {e} has a different header (were the groups the same?)")
        sig_hdr = h
        eff_hdr = eh or eff_hdr
        rows += [r.split("\t") for r in body]
        eff_rows += ebody
        used += 1
    if sig_hdr is None:
        raise SystemExit("ERROR: every part was empty; nothing was tested")
    cols = sig_hdr.split("\t")
    ip, iq = cols.index("p"), cols.index("p.adjust")
    tested = [i for i, r in enumerate(rows) if r[ip] not in ("NA", "")]
    adj = bh([float(rows[i][ip]) for i in tested])
    for r in rows:
        r[iq] = "NA"
    for i, q in zip(tested, adj):
        rows[i][iq] = repr(q)
    with open(a.output_prefix + SIG, "w") as f:
        f.write(sig_hdr + "\n")
        f.writelines("\t".join(r) + "\n" for r in rows)
    with open(a.output_prefix + EFF, "w") as f:
        f.write((eff_hdr or "") + "\n")
        f.writelines(r + "\n" for r in eff_rows)
    print(f"parts with clusters {used} of {len(sig_files)}; clusters {len(rows)}; "
          f"tested {len(tested)}; FDR < 0.05 {sum(q < 0.05 for q in adj)}")


if __name__ == "__main__":
    main()
