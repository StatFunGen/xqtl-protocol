"""Tier B: twas_weights.R -> TwasWeights (enet, one row per context)."""
from __future__ import annotations

import pytest

GENE = "ENSG00000283047"


def test_twas_weights(qtl_dataset, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "tw.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/twas_weights.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", GENE,
               "--methods", "enet", "--output", out], timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "TwasWeights"
    assert info["nrow"] == 2
    assert info["MethodNames"] == ["enet"]
