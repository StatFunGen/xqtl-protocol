"""Tier B: legacy_twas_weights_convert.R -> S4 TwasWeights from a legacy
univariate_twas_weights RDS."""
from __future__ import annotations

import pytest


def test_legacy_twas_weights_convert(run_r, read_rds, repo_root, tmp_path):
    legacy = repo_root / ("tests/fixtures/twas/"
        "protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.univariate_twas_weights.rds")
    out = tmp_path / "tw.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/legacy_twas_weights_convert.R",
              ["--legacy", legacy, "--study", "protocol_example", "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "TwasWeights"
