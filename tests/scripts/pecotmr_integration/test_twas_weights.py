"""Tier B: twas_weights.R -> TwasWeights (enet, one row per context)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

GENE = "ENSG00000283047"
EXPECTED = "tests/fixtures/twas_weights/expected/twas_weights.enet.rds"


def test_twas_weights(qtl_dataset, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "tw.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/twas_weights.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", GENE,
               "--methods", "enet", "--seed", "1", "--output", out], timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "TwasWeights"
    assert info["nrow"] == 2
    assert info["MethodNames"] == ["enet"]
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
