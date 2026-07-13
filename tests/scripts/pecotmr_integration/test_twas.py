"""Tier B: twas.R -> TWAS-Z GRanges from a TwasWeights + GwasSumStats built here."""
from __future__ import annotations

import pytest

FX = "tests/fixtures"


def test_twas(run_r, read_rds, repo_root, tmp_path):
    r = repo_root
    tw = tmp_path / "tw.rds"
    assert run_r(r / "code/script/pecotmr_integration/legacy_twas_weights_convert.R",
        ["--legacy", r / f"{FX}/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.univariate_twas_weights.rds",
         "--study", "protocol_example", "--output", tw], timeout=120).returncode == 0
    gss = tmp_path / "gss.rds"
    assert run_r(r / "code/script/pecotmr_integration/gwas_sumstats_construct.R",
        ["--study", "protocol_example_twas_chr22",
         "--gwas-tsv", r / f"{FX}/twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
         "--ld-block", "chr22:10000000-19000000",
         "--ld-meta", r / f"{FX}/ld_reference/ld_meta_file.tsv", "--output", gss], timeout=300).returncode == 0
    out = tmp_path / "twas.rds"
    p = run_r(r / "code/script/pecotmr_integration/twas.R",
              ["--twas-weights", tw, "--gwas-sumstats", gss, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "GRanges"
