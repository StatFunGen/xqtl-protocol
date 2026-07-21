"""Notebook tier: mash_fit.ipynb `mash` workflow chains [mash_1] fit -> model and
[mash_2] posterior, given a Vhat + prior (from the mash_model_chain fixture)."""
from __future__ import annotations


def test_mash_fit(mash_model_chain, run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "mashfit"
    p = run_sos(repo_root / "pipeline/mash_fit.ipynb", "mash",
                dict(data=mash_model_chain["data"], vhat_data=mash_model_chain["vhat"],
                     prior_data=mash_model_chain["prior"], cwd=cwd,
                     output_prefix="toy", effect_model="EE"),
                cwd=repo_root, timeout=700)
    assert p.returncode == 0, p.stdout + p.stderr
    model = cwd / "toy.EE.mash_model.rds"
    post = cwd / "toy.EE.posterior.rds"
    assert model.exists() and post.exists(), p.stdout
    assert "PosteriorMean" in read_rds(post)["names"]
