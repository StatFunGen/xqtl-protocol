"""Tier B: sldsc_postprocess.R -> per_trait / meta / params from polyfun's
per-trait sLDSC outputs (--maf-cutoff 0, no frq)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/sldsc_enrichment/expected/sldsc_postprocess.rds"


def test_sldsc_postprocess(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/sldsc_enrichment"
    out = tmp_path / "pp.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/sldsc_postprocess.R",
              ["--traits-file", fx / "sumstats_test_all.txt",
               "--heritability-cwd", fx / "sldsc_heritability",
               "--annotation-name", "protocol_example",
               "--target-anno-dir", fx / "target_anno", "--maf-cutoff", "0",
               "--target-categories", "ANNOT_0",
               "--target-categories-label", "protocol_example_annotation",
               "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert {"per_trait", "meta", "params"}.issubset(set(read_rds(out)["names"]))
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
