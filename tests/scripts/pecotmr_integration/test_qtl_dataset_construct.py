"""Tier B: qtl_dataset_construct.R builds a QtlDataset (2 contexts) from the
qtl_mini phenotype manifest + shared genotype."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/qtl_dataset/expected/qtl_dataset.rds"


def test_qtl_dataset_construct(qtl_dataset, read_rds, repo_root):
    info = read_rds(qtl_dataset)
    assert info["class"] == "QtlDataset"
    assert set(info["Contexts"]) == {"context1", "context2"}
    assert info["Study"] == ["test_study"]
    assert_matches_expected(qtl_dataset, repo_root / EXPECTED, mode="tolerant",
                            rtol=1e-6, atol=1e-8, normalize_paths=True)
