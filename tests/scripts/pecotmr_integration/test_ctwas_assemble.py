"""Tier B: ctwas_assemble.R -> assembled cTWAS inputs (from the ctwas_chain)."""
from __future__ import annotations

import pytest


def test_ctwas_assemble(ctwas_chain, read_rds):
    assert read_rds(ctwas_chain["inputs"])["class"] == "list"
