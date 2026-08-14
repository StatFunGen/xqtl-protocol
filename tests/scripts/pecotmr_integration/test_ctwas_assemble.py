"""Tier B: ctwas_assemble.R -> assembled cTWAS inputs (from the ctwas_chain)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/ctwas/expected/ctwas_assemble.rds"


def test_ctwas_assemble(ctwas_chain, read_rds, repo_root):
    assert read_rds(ctwas_chain["inputs"])["class"] == "list"

    # embeds absolute LD/snp-map reference paths -> normalize_paths (basename compare).
    assert_matches_expected(ctwas_chain["inputs"], repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8, normalize_paths=True)
