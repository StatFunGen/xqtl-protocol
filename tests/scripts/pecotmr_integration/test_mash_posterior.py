"""Script tier: mash_posterior.R -> MASH posterior matrices via pecotmr::mashPosterior
(the [mash_2] step). Uses the mash_model_chain fixture (Vhat + fitted model) over
the MWE input."""
from __future__ import annotations


def _posterior(mash_model_chain, run_r, repo_root, out, extra=()):
    return run_r(repo_root / "code/script/pecotmr_integration/mash_posterior.R",
                 ["--data", mash_model_chain["data"],
                  "--bhat-key", "strong.b", "--shat-key", "strong.s",
                  "--vhat-data", mash_model_chain["vhat"],
                  "--mash-model", mash_model_chain["model"],
                  "--effect-model", "EE", "--output", out, *extra], timeout=400)


def test_mash_posterior(mash_model_chain, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "posterior.rds"
    p = _posterior(mash_model_chain, run_r, repo_root, out)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "list"
    assert {"PosteriorMean", "PosteriorCov"}.issubset(probe["names"])


def test_mash_posterior_exclude_condition(mash_model_chain, run_r, read_rds,
                                          repo_root, tmp_path):
    # dropping a condition (by 1-based index) yields a posterior over one fewer
    out = tmp_path / "posterior_excl.rds"
    p = _posterior(mash_model_chain, run_r, repo_root, out,
                   extra=["--exclude-condition", "1", "--no-posterior-cov"])
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "list"
    assert "PosteriorMean" in probe["names"]
