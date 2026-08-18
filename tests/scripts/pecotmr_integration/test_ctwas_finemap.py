"""Tier B: ctwas_finemap.R -> CtwasResult with the blessed gene pip/z (from the
ctwas_chain)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/ctwas/expected/ctwas_finemap.rds"


def test_ctwas_finemap(ctwas_chain, read_rds, repo_root):
    info = read_rds(ctwas_chain["finemap"])
    assert info["class"] == "CtwasResult"
    assert info["geneMaxPip"] == pytest.approx(1.0, abs=1e-3), info
    # geneTopZ is ctwas 0.6.0's LD-adjusted burden Z, w'z / sqrt(w'Rw) = 3.168.
    # The pre-0.6.0 5.462 under-adjusted the LD (it exceeded the max single-SNP
    # |z| = 4.13, which these SNPs' mutual LD makes impossible).
    assert info["geneTopZ"] == pytest.approx(3.168, abs=0.01), info

    # deterministic single-region susie fine-map; no embedded paths.
    assert_matches_expected(ctwas_chain["finemap"], repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8)
