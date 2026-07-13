"""Tier B: twas_manifest.R resolves the twas metas into per-region TWAS units."""
from __future__ import annotations

import pytest


def test_twas_manifest(run_r, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/twas"
    out = tmp_path / "manifest.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/twas_manifest.R",
              ["--gwas-meta", fx / "protocol_example.twas.gwas_meta.tsv",
               "--xqtl-meta", fx / "protocol_example.twas.xqtl_meta.TAB.tsv",
               "--region-name", "chr22_10000000_19000000", "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert rows[0] == ["region_id", "chrom", "start", "stop", "genes", "weight_files",
                       "gwas_studies", "gwas_files", "gwas_mappings"]
    assert len(rows) == 2
    assert rows[1][0] == "chr22_10000000_19000000" and "ENSG00000130538" in rows[1][4]
