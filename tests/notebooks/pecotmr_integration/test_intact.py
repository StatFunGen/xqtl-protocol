"""Notebook tier: intact.ipynb (TWAS x COLOC integration via INTACT).

The `intact` step is a thin wrapper: the cell shells out to
`code/script/pecotmr_integration/intact.R`, which inner-joins PTWAS output
(GENE / STAT z-score / SUBCLASS) with fastenloc output (Gene / GLCP), runs the
Bioconductor INTACT package (`intact()`: z -> Bayes factor, GLCP -> linear prior
-> posterior; then `fdr_rst()`), and writes one RDS augmented with `intact_pip`
and an FDR flag `fdr_sig`, sorted by descending posterior. The worker itself is
covered directly at Tier B (tests/scripts/pecotmr_integration/test_intact.py);
this test asserts the notebook wires it end-to-end.

The committed fixture is a small hand-authored PTWAS + fastenloc pair (there is
no upstream PTWAS/fastenloc data in-repo to derive from). It is shaped to
exercise the wrapper's join logic and both FDR outcomes:
  * a duplicate (GENE, SUBCLASS) row  -> collapsed by distinct()
  * a gene present only in PTWAS      -> dropped by the inner_join (5 of 6 kept)
  * two strong genes + three weak ones -> fdr_sig has both TRUE and FALSE

That last point guards a real bug that was fixed alongside this test: the step
used to pass `alpha` as the *string* "0.05", so `fdr_rst`'s `FDR <= alpha`
compared numeric-vs-character, `which()` came back empty, and **every** gene got
`fdr_sig = FALSE` regardless of signal. `read_rds` only sees structure (class /
nrow / columns), which is identical either way, so the test reads the actual
`fdr_sig` / `intact_pip` values to catch a regression.
"""
from __future__ import annotations

import csv
import subprocess

import pytest

from helpers.r_runner import rscript_bin

NB = "pipeline/intact.ipynb"
FX = "tests/fixtures/intact"


def _have_intact():
    """Whether the Bioconductor INTACT package is loadable (skip guard)."""
    p = subprocess.run(
        [rscript_bin(), "-e", 'quit(status = !requireNamespace("INTACT", quietly = TRUE))'],
        capture_output=True, text=True, timeout=120)
    return p.returncode == 0


def _read_table(rds):
    """Read the INTACT result data.frame as a list of dict rows (via Rscript ->
    TSV), so the test can assert on cell values `read_rds` doesn't surface."""
    p = subprocess.run(
        [rscript_bin(), "-e",
         f'write.table(readRDS("{rds}"), sep="\\t", quote=FALSE, row.names=FALSE)'],
        capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return list(csv.DictReader(lines, delimiter="\t"))


@pytest.mark.skipif(not _have_intact(), reason="Bioconductor INTACT not installed")
def test_intact(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / FX
    cwd = tmp_path / "intact"
    p = run_sos(repo_root / NB, "intact",
                {"ptwas_file": fx / "protocol_example.ptwas.output",
                 "fastenloc_file": fx / "protocol_example.fastenloc.gene.out",
                 "tissue": "DLPFC",
                 "alpha": 0.05,
                 "cwd": cwd},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr

    out = cwd / "DLPFC.INTACT.rds"
    assert out.exists(), p.stdout

    # structure: the merged table with the two INTACT-added columns. Five genes:
    # six PTWAS rows -> distinct() drops the duplicate -> inner_join drops the one
    # gene missing from fastenloc.
    info = read_rds(out)
    assert info["class"] == "data.frame"
    assert info["nrow"] == 5, info
    for col in ("gene", "zscore", "GLCP", "intact_pip", "fdr_sig"):
        assert col in info["names"], info

    # values: sorted by descending posterior; posteriors in [0, 1]; and fdr_sig
    # is actually functional (both TRUE and FALSE) -- the alpha-as-string bug
    # made this column uniformly FALSE.
    rows = _read_table(out)
    assert len(rows) == 5
    pips = [float(r["intact_pip"]) for r in rows]
    assert pips == sorted(pips, reverse=True), pips
    assert all(0.0 <= x <= 1.0 for x in pips), pips
    flags = {r["fdr_sig"] for r in rows}
    assert flags == {"TRUE", "FALSE"}, rows
    # the two strong genes (high |z| + high GLCP) are the significant ones
    sig = {r["gene"] for r in rows if r["fdr_sig"] == "TRUE"}
    assert sig == {"ENSG00000128272", "ENSG00000100060"}, rows
