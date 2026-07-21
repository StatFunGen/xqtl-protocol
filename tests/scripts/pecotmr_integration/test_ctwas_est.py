"""Tier B: ctwas_est.R -> estimated cTWAS priors (from the ctwas_chain)."""
from __future__ import annotations

import pytest


def test_ctwas_est(ctwas_chain, read_rds):
    assert read_rds(ctwas_chain["est"])["class"] == "list"
