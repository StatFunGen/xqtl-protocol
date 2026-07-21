"""Tier B: mash_plot_prior.R -> covariance-sharing heatmap PDF for a MASH prior
(list(U, w)) produced by mash_prior.R."""
from __future__ import annotations

import pytest

FX = "tests/fixtures/mash/mashr_input.rds"


def test_mash_plot_prior(run_r, repo_root, tmp_path):
    vhat = tmp_path / "vhat.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_vhat.R",
              ["--data", repo_root / FX, "--method", "simple",
               "--effect-model", "EE", "--output", vhat], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    prior = tmp_path / "prior.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_prior.R",
              ["--data", repo_root / FX, "--engine", "cov_ed",
               "--components", "canonical", "--vhat-data", vhat,
               "--effect-model", "EE", "--output", prior], timeout=400)
    assert p.returncode == 0, p.stdout + p.stderr

    pdf = tmp_path / "prior.pdf"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_plot_prior.R",
              ["--data", prior, "--tol", "1E-6", "--output", pdf], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert pdf.exists() and pdf.stat().st_size > 1000
