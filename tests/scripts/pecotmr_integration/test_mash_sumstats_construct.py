"""Tier B: mash_sumstats_construct.R -> multi-context QtlSumStats from
per-condition tensorqtl-style z-score TSVs for one region."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/mash/expected/mash_sumstats.region1.rds"


def test_mash_sumstats_construct(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/mash"
    out = tmp_path / "region1.qss.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_sumstats_construct.R",
              ["--tensorqtl-paths", fx / "Mic_De_Jager_eQTL.tsv",
               fx / "Ast_De_Jager_eQTL.tsv",
               "--conditions", "Mic_De_Jager_eQTL,Ast_De_Jager_eQTL",
               "--region", "chr22:15528191-15529138",
               "--study", "protocol_example", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "QtlSumStats"
    assert set(probe["Contexts"]) == {"Mic_De_Jager_eQTL", "Ast_De_Jager_eQTL"}
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
