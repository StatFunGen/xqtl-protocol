"""Tier B: mash_vhat.R -> residual correlation (Vhat) via
pecotmr::mashResidualCorrelation, over the MWE-derived mash input."""
from __future__ import annotations

import pytest

FX = "tests/fixtures/mash/mashr_input.rds"


@pytest.mark.parametrize("method", ["identity", "simple", "corshrink", "simple_specific"])
def test_mash_vhat(method, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / f"vhat_{method}.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_vhat.R",
              ["--data", repo_root / FX, "--method", method,
               "--effect-model", "EE", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    # conditions x conditions correlation matrix (8 conditions in the fixture)
    assert probe["class"] == "matrix"
    assert probe["nrow"] == 8
