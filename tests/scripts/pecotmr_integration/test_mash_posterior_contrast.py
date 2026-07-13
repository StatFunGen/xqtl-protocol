"""Tier B: mash_posterior_contrast.R + _summary.R + _plot.R.

Per-region posterior contrasts (via pecotmr::mashPosteriorContrast), the
cross-region significance summary, and the ggnewscale heatmap. Fixtures are a
small synthetic posterior (PosteriorMean/PosteriorCov) + aligned effect matrix.
"""
from __future__ import annotations

import pytest

FX = "tests/fixtures/mash_posterior"
CELLS = "ALL,Ast,End,Exc,Inh,Mic,OPC,Oli"   # the MWE mashr_input conditions


def _contrast(run_r, repo_root, tmp_path):
    out = tmp_path / "contrast.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_posterior_contrast.R",
              ["--posterior", repo_root / FX / "posterior.rds",
               "--orig-data", repo_root / FX / "orig.rds", "--orig-key", "bhat",
               "--cells", CELLS, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def test_mash_posterior_contrast(run_r, read_rds, repo_root, tmp_path):
    out = _contrast(run_r, repo_root, tmp_path)
    probe = read_rds(out)
    assert probe["class"] == "data.frame"
    names = probe["names"]
    # deviation + pairwise contrasts across mean / se / p
    assert any(n.startswith("mean_contrast_") for n in names)
    assert any(n.startswith("se_contrast_") for n in names)
    assert any(n.startswith("p_contrast_") for n in names)
    assert any("deviation" in n for n in names) and any("_vs_" in n for n in names)


def test_mash_posterior_contrast_summary_and_plot(run_r, repo_root, tmp_path):
    contrast = _contrast(run_r, repo_root, tmp_path)

    summ = tmp_path / "posterior_sum.csv"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_posterior_contrast_summary.R",
              ["--contrast", contrast, "--cells", CELLS, "--p-cutoff", "0.5",
               "--output", summ], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    header = summ.read_text().splitlines()[0]
    assert "Ast_vs_Mic" in header  # a pairwise contrast column

    png = tmp_path / "posterior_sum.png"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_posterior_contrast_plot.R",
              ["--data", summ, "--output", png], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert png.exists() and png.stat().st_size > 1000
