"""Notebook tier: mixture_prior.ipynb -- end-to-end MASH prior estimation.

Running the `ed_bovy` (cov_ed) prior engine chains the component steps
(canonical/pca) and the Vhat step via SoS auto-dependency resolution, so one
run exercises mash_covariance.R + mash_vhat.R + mash_prior.R together.
"""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/mash/expected/mixture_prior.EE.prior.rds"


def test_mixture_prior_ed_bovy(run_sos, read_rds, repo_root, tmp_path):
    data = repo_root / "tests/fixtures/mash/mashr_input.rds"
    cwd = tmp_path / "mixprior"
    p = run_sos(repo_root / "pipeline/mixture_prior.ipynb", "ed_bovy",
                dict(data=data, cwd=cwd, output_prefix="toy", effect_model="EE",
                     vhat="simple", mixture_components=["canonical", "pca"],
                     modular_script_dir=repo_root / "code/script"),
                cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    out = cwd / "toy.EE.prior.rds"
    assert out.exists(), p.stdout
    names = set(read_rds(out)["names"])
    assert "U" in names and "w" in names
    # regression: the prior (cov_ed + canonical/pca, deterministic) reproduces the
    # committed expected list within tolerance (deep all.equal via rds_compare.R).
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
