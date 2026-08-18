"""Tier B: mash_covariance.R -> mixture-prior covariance components via
pecotmr::mashCovarianceComponents, over the MWE-derived mash input."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

FX = "tests/fixtures/mash/mashr_input.rds"
EXPECTED_DIR = "tests/fixtures/mash/expected"


def _run(run_r, repo_root, out, component, extra=()):
    return run_r(repo_root / "code/script/pecotmr_integration/mash_covariance.R",
                 ["--data", repo_root / FX, "--component", component,
                  "--effect-model", "EE", "--output", out, *extra], timeout=300)


@pytest.mark.parametrize("component", ["canonical", "pca", "flash", "flash_nonneg"])
def test_mash_covariance_runs(component, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / f"cov_{component}.rds"
    extra = ["--npc", "3"] if component == "pca" else ()
    p = _run(run_r, repo_root, out, component, extra)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "list"   # a list of covariance matrices
    assert_matches_expected(out, repo_root / f"{EXPECTED_DIR}/cov.{component}.EE.rds",
                            mode="tolerant", rtol=1e-6, atol=1e-8)


def test_mash_covariance_canonical_components(run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "cov_canonical.rds"
    p = _run(run_r, repo_root, out, "canonical")
    assert p.returncode == 0, p.stdout + p.stderr
    names = read_rds(out)["names"]
    # canonical always yields identity + equal_effects + per-condition singletons
    assert "identity" in names and "equal_effects" in names
