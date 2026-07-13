"""Tier B: ctwas_finemap.R -> CtwasResult with the blessed gene pip/z (from the
ctwas_chain)."""
from __future__ import annotations

import pytest


def test_ctwas_finemap(ctwas_chain, read_rds):
    info = read_rds(ctwas_chain["finemap"])
    assert info["class"] == "CtwasResult"
    assert info["geneMaxPip"] == pytest.approx(1.0, abs=1e-3), info
    assert info["geneTopZ"] == pytest.approx(5.462, abs=0.01), info
