"""Tier B: gwas_rss_plot.R renders a PNG from a GWAS-RSS GwasFineMappingResult
(built here: gwas_sumstats_construct -> fine_mapping)."""
from __future__ import annotations

import pytest

S = "code/script/pecotmr_integration"


def test_gwas_rss_plot(run_r, repo_root, tmp_path):
    r = repo_root
    fx = r / "tests/fixtures"
    gss = tmp_path / "gss.rds"
    assert run_r(r / f"{S}/gwas_sumstats_construct.R",
        ["--study", "protocol_example_twas_chr22",
         "--gwas-tsv", fx / "twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
         "--ld-block", "chr22:10000000-19000000",
         "--ld-meta", fx / "ld_reference/ld_meta_file.tsv", "--output", gss], timeout=300).returncode == 0
    fmr = tmp_path / "gwas_fmr.rds"
    assert run_r(r / f"{S}/fine_mapping.R",
        ["--gwas-sumstats", gss, "--methods", "susie", "--output", fmr], timeout=600).returncode == 0
    out = tmp_path / "plot.png"
    p = run_r(r / f"{S}/gwas_rss_plot.R", ["--input", fmr, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.stat().st_size > 0
