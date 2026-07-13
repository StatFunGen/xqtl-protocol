"""Tier B: mash_prior.R -> refined MASH prior (U + weights) via
pecotmr::mashPriorCovariances, over the MWE-derived mash input. Covers both the
self-contained (build components here) and the pipeline (refine pre-built
component RDS) modes with the dependency-light cov_ed engine."""
from __future__ import annotations

import pytest

FX = "tests/fixtures/mash/mashr_input.rds"


def _vhat(run_r, repo_root, tmp_path):
    out = tmp_path / "vhat.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_vhat.R",
              ["--data", repo_root / FX, "--method", "simple",
               "--effect-model", "EE", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def test_mash_prior_self_contained(run_r, read_rds, repo_root, tmp_path):
    vhat = _vhat(run_r, repo_root, tmp_path)
    out = tmp_path / "prior.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_prior.R",
              ["--data", repo_root / FX, "--engine", "cov_ed",
               "--components", "canonical,pca", "--npc", "3",
               "--vhat-data", vhat, "--effect-model", "EE", "--output", out],
              timeout=400)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "list"
    assert "U" in probe["names"] and "w" in probe["names"]


def test_mash_prior_component_files(run_r, read_rds, repo_root, tmp_path):
    # pipeline mode: build components separately, then refine them
    canon = tmp_path / "canon.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_covariance.R",
              ["--data", repo_root / FX, "--component", "canonical",
               "--effect-model", "EE", "--output", canon], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    out = tmp_path / "prior.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_prior.R",
              ["--data", repo_root / FX, "--engine", "cov_ed",
               "--component-files", str(canon), "--effect-model", "EE",
               "--output", out], timeout=400)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "U" in read_rds(out)["names"]
