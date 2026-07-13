"""Tier B: colocboost.R -> xQTL colocalization list over a qtl_mini gene."""
from __future__ import annotations

import pytest


def test_colocboost(qtl_dataset, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "coloc.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/colocboost.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", "ENSG00000283047",
               "--output", out], timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "list" and "xqtl_coloc" in info["names"]
