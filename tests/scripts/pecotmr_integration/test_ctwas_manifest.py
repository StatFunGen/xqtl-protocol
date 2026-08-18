"""Tier B: ctwas_manifest.R enumerates the per-LD-block grid from the ld-meta."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/ctwas/expected/ctwas_manifest.tsv"


def test_ctwas_manifest(run_r, repo_root, tmp_path):
    out = tmp_path / "ctwas_manifest.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/ctwas_manifest.R",
              ["--ld-meta", repo_root / "tests/fixtures/ld_reference/ld_meta_file.ctwas.tsv",
               "--chrom", "chr22", "--gwas-sumstats-dir", tmp_path, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert rows[0] == ["region_id", "region", "gwas_sumstats_rds"]
    assert len(rows) == 21          # header + 20 chr22 blocks
    assert all(r[1].startswith("chr22:") for r in rows[1:])

    # gwas_sumstats_rds is an absolute tmp path -> normalize_paths (basename compare).
    assert_matches_expected(out, repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8, normalize_paths=True)
