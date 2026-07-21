"""Tier B: colocboost_manifest.R resolves per-locus colocboost units from the
phenotype manifest + association windows."""
from __future__ import annotations

import pytest


def test_colocboost_manifest(run_r, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/qtl_mini"
    out = tmp_path / "manifest.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/colocboost_manifest.R",
              ["--pheno-manifest", fx / "pheno_manifest_multicontext.tsv",
               "--customized-association-windows", fx / "association_windows.bed",
               "--region-name", "ENSG00000283047", "--cis-window", "1000000",
               "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert len(rows) >= 2
    assert any("ENSG00000283047" in "\t".join(r) for r in rows[1:])
