"""Tier B: fine_mapping.R -> QtlFineMappingResult over a qtl_mini gene (susie,
one row per context)."""
from __future__ import annotations

import pytest

GENE = "ENSG00000283047"


def test_fine_mapping(qtl_dataset, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "fmr.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", GENE, "--output", out],
              timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "QtlFineMappingResult"
    assert info["nrow"] == 2                      # one row per context
    assert info["MethodNames"] == ["susie"]
