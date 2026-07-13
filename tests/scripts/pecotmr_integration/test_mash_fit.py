"""Script tier: mash_fit.R -> fitted MASH mixture model via pecotmr::mashModelFit
(the [mash_1] step). Driven by the mash_model_chain fixture (Vhat -> prior -> fit)
over the MWE-derived mash input."""
from __future__ import annotations


def test_mash_fit(mash_model_chain, read_rds):
    probe = read_rds(mash_model_chain["model"])
    assert probe["class"] == "list"
    # mash_fit.R saves list(mash_model, vhat_file, prior_file)
    assert {"mash_model", "vhat_file", "prior_file"}.issubset(probe["names"])
