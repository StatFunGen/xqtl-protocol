"""Tier B: ctwas_est.R -> estimated cTWAS priors (from the ctwas_chain)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/ctwas/expected/ctwas_est.rds"


def test_ctwas_est(ctwas_chain, read_rds, repo_root):
    assert read_rds(ctwas_chain["est"])["class"] == "list"

    # deterministic --fallback-to-prefit path (pinned --niter 50); embeds absolute
    # reference paths -> normalize_paths (basename compare).
    assert_matches_expected(ctwas_chain["est"], repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8, normalize_paths=True)
