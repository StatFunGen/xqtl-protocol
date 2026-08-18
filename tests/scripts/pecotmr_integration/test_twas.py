"""Tier B: twas.R -> TWAS-Z GRanges from a committed S4 TwasWeights + a
GwasSumStats built here (the legacy twas-weights conversion has been retired)."""
from __future__ import annotations

from helpers.expected import assert_matches_expected

FX = "tests/fixtures"
TW = FX + "/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.twas_weights.s4.rds"
EXPECTED = FX + "/twas/expected/twas.chr22.rds"


def test_twas(run_r, read_rds, repo_root, tmp_path):
    r = repo_root
    gss = tmp_path / "gss.rds"
    assert run_r(r / "code/script/pecotmr_integration/gwas_sumstats_construct.R",
        ["--study", "protocol_example_twas_chr22",
         "--gwas-tsv", r / f"{FX}/twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
         "--ld-block", "chr22:10000000-19000000",
         "--ld-meta", r / f"{FX}/ld_reference/ld_meta_file.tsv", "--output", gss], timeout=300).returncode == 0
    out = tmp_path / "twas.rds"
    p = run_r(r / "code/script/pecotmr_integration/twas.R",
              ["--twas-weights", r / TW, "--gwas-sumstats", gss, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "GRanges"
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)
