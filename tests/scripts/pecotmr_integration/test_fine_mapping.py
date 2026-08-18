"""Tier B: fine_mapping.R -> QtlFineMappingResult over a qtl_mini gene (susie,
one row per context)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

GENE = "ENSG00000283047"
EXPECTED = "tests/fixtures/fine_mapping/expected/fine_mapping.ENSG00000283047.rds"


def test_fine_mapping(qtl_dataset, run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "fmr.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", GENE, "--output", out],
              timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "QtlFineMappingResult"
    assert info["nrow"] == 2                      # one row per context
    assert info["MethodNames"] == ["susie"]


def test_fine_mapping_matches_expected(qtl_dataset, run_r, repo_root, tmp_path):
    """Regression: a --seed-pinned susie fit reproduces the committed expected
    QtlFineMappingResult within tolerance (deep all.equal via rds_compare.R).
    Seeding makes it bit-reproducible on one machine; `tolerant` mode absorbs the
    x86/ARM float drift across the CI matrix."""
    out = tmp_path / "fmr.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping.R",
              ["--qtl-dataset", qtl_dataset, "--gene-id", GENE, "--seed", "1", "--output", out],
              timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
