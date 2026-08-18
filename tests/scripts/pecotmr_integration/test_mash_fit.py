"""Script tier: mash_fit.R -> fitted MASH mixture model via pecotmr::mashModelFit
(the [mash_1] step). Driven by the mash_model_chain fixture (Vhat -> prior -> fit)
over the MWE-derived mash input."""
from __future__ import annotations

from helpers.expected import assert_matches_expected

# The saved list embeds the tmp vhat_file / prior_file paths of the chain build,
# so the comparison basename-normalizes them (normalize_paths=True).
EXPECTED = "tests/fixtures/mash/expected/mash_model.EE.rds"


def test_mash_fit(mash_model_chain, read_rds, repo_root):
    probe = read_rds(mash_model_chain["model"])
    assert probe["class"] == "list"
    # mash_fit.R saves list(mash_model, vhat_file, prior_file)
    assert {"mash_model", "vhat_file", "prior_file"}.issubset(probe["names"])
    assert_matches_expected(mash_model_chain["model"], repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8,
                            normalize_paths=True)
