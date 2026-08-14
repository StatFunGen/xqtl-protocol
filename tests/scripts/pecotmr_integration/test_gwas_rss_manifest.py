"""Tier B: gwas_rss_manifest.R builds the study x region GWAS-RSS manifest."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/gwas_rss_manifest/expected/gwas_rss_manifest.tsv"


def test_gwas_rss_manifest(run_r, repo_root, tmp_path):
    out = tmp_path / "manifest.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/gwas_rss_manifest.R",
              ["--gwas-meta", repo_root / "tests/fixtures/rss_analysis/gwas_meta_data.tsv",
               "--regions", "chr22:49355984-50799822", "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert "study_id" in rows[0] and "region_id" in rows[0]
    assert len(rows) >= 2           # header + >=1 study x region row
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant",
                            rtol=1e-6, atol=1e-8, normalize_paths=True)
