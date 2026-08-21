"""Tier B: manifest-builder wrappers produce well-formed TSVs from the fixtures."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/region_manifest/expected/region_manifest.tsv"


def test_region_manifest(run_r, repo_root, qtl_mini, tmp_path):
    out = tmp_path / "regions.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/region_manifest.R",
              ["--pheno-manifest", qtl_mini / "protocol_example.pheno_manifest_context.tsv",
               "--customized-association-windows", qtl_mini / "association_windows.bed",
               "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = out.read_text().splitlines()
    assert lines, "region manifest is empty"
    cols = lines[0].split("\t")
    assert {"region_id", "chr", "start", "end"} <= set(cols)
    assert any("ENSG00000283047" in ln for ln in lines[1:])
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
