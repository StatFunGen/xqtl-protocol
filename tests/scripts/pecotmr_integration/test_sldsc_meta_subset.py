"""Tier B: sldsc_meta_subset.R re-metas a trait subset off a postprocess RDS
(built here from sldsc_postprocess.R)."""
from __future__ import annotations

import pytest


def test_sldsc_meta_subset(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/sldsc_enrichment"
    pp = tmp_path / "pp.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/sldsc_postprocess.R",
              ["--traits-file", fx / "sumstats_test_all.txt",
               "--heritability-cwd", fx / "sldsc_heritability",
               "--annotation-name", "protocol_example",
               "--target-anno-dir", fx / "target_anno", "--maf-cutoff", "0",
               "--target-categories", "ANNOT_0", "--output", pp], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "meta.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/sldsc_meta_subset.R",
              ["--postprocess-rds", pp,
               "--subset-traits-file", fx / "sumstats_test_category1.txt",
               "--target-categories", "ANNOT_0", "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "enrichment" in read_rds(out)["names"]
