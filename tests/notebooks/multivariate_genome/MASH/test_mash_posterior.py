"""Notebook tier: mash_posterior.ipynb `posterior` workflow -> per-region
posterior RDS. Stages the Vhat + model (from mash_model_chain) at the notebook's
derived paths."""
from __future__ import annotations

import shutil


def test_mash_posterior(mash_model_chain, run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "mashpost"
    cwd.mkdir()
    shutil.copy(mash_model_chain["vhat"], cwd / "toy.EE.V_simple.rds")
    shutil.copy(mash_model_chain["model"], cwd / "toy.EE.V_simple.mash_model.rds")
    region = repo_root / "tests/fixtures/mash/region_strong.rds"
    units = tmp_path / "units.txt"
    units.write_text(f"{region}\n")
    p = run_sos(repo_root / "pipeline/mash_posterior.ipynb", "posterior",
                dict(name="toy", data=region, output_prefix="toy", cwd=cwd,
                     analysis_units=units, bhat_table_name="bhat",
                     shat_table_name="sbhat", effect_model="EE", vhat="simple"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    post = cwd / "cache/toy.region_strong.posterior.rds"
    assert post.exists(), p.stdout
    assert "PosteriorMean" in read_rds(post)["names"]
