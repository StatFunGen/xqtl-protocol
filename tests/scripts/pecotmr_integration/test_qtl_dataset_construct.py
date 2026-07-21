"""Tier B: qtl_dataset_construct.R builds a QtlDataset (2 contexts) from the
qtl_mini phenotype manifest + shared genotype."""
from __future__ import annotations

import pytest


def test_qtl_dataset_construct(qtl_dataset, read_rds):
    info = read_rds(qtl_dataset)
    assert info["class"] == "QtlDataset"
    assert set(info["Contexts"]) == {"context1", "context2"}
    assert info["Study"] == ["test_study"]
