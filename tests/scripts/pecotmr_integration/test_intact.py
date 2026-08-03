"""Tier B: intact.R -> INTACT-integrated gene table from a PTWAS + fastenloc
pair. Joins on gene (distinct (GENE, SUBCLASS); inner_join drops the unmatched
gene), runs INTACT::intact + INTACT::fdr_rst, writes intact_pip + fdr_sig
sorted by descending posterior.

`read_rds` only reports structure (class / nrow / columns), so the test also
reads the RDS values to assert fdr_sig is actually functional ({TRUE, FALSE})
-- guarding the alpha-as-number contract: intact.R passes --alpha through as a
number, and passing it as a string made fdr_rst's FDR<=alpha a char comparison
that flagged nothing (fdr_sig uniformly FALSE).
"""
from __future__ import annotations

import csv
import subprocess

from helpers.r_runner import rscript_bin

FX = "tests/fixtures/intact"
SCRIPT = "code/script/pecotmr_integration/intact.R"


def _rows(rds):
    """The result data.frame as list-of-dict rows (Rscript -> TSV), so the test
    can assert on cell values `read_rds` does not surface."""
    p = subprocess.run(
        [rscript_bin(), "-e",
         f'write.table(readRDS("{rds}"), sep="\\t", quote=FALSE, row.names=FALSE)'],
        capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    return list(csv.DictReader([ln for ln in p.stdout.splitlines() if ln.strip()],
                               delimiter="\t"))


def test_intact(run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "intact.rds"
    p = run_r(repo_root / SCRIPT,
              ["--ptwas-file", repo_root / FX / "protocol_example.ptwas.output",
               "--fastenloc-file", repo_root / FX / "protocol_example.fastenloc.gene.out",
               "--alpha", "0.05", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    # structure: six PTWAS rows -> distinct() drops the duplicate -> inner_join
    # drops the gene missing from fastenloc -> five genes with the two added cols.
    info = read_rds(out)
    assert info["class"] == "data.frame" and info["nrow"] == 5, info
    for col in ("gene", "zscore", "GLCP", "intact_pip", "fdr_sig"):
        assert col in info["names"], info

    # values: posteriors in [0, 1], sorted descending, and fdr_sig functional
    # (both TRUE and FALSE), with the two strong genes the significant ones.
    rows = _rows(out)
    pips = [float(r["intact_pip"]) for r in rows]
    assert pips == sorted(pips, reverse=True), pips
    assert all(0.0 <= x <= 1.0 for x in pips), pips
    assert {r["fdr_sig"] for r in rows} == {"TRUE", "FALSE"}, rows
    assert {r["gene"] for r in rows if r["fdr_sig"] == "TRUE"} == \
        {"ENSG00000128272", "ENSG00000100060"}, rows
