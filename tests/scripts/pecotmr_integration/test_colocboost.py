"""Tier B: colocboost.R -> xQTL ColocBoostResult over a qtl_mini gene."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/susie_enloc/expected/colocboost.rds"


def test_colocboost(qtl_dataset, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "coloc.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/colocboost.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", "ENSG00000283047",
               "--seed", "1", "--output", out], timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "ColocBoostResult" and "isColocalized" in info["colnames"]
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
