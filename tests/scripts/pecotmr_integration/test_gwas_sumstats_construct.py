"""Tier B: gwas_sumstats_construct.R -> GwasSumStats over one LD block from a
bgzipped GWAS TSV + the genotype LD sketch."""
from __future__ import annotations

import pytest


def test_gwas_sumstats_construct(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures"
    out = tmp_path / "gss.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/gwas_sumstats_construct.R",
              ["--study", "protocol_example_twas_chr22",
               "--gwas-tsv", fx / "twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
               "--ld-block", "chr22:10000000-19000000",
               "--ld-meta", fx / "ld_reference/ld_meta_file.tsv", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "GwasSumStats"
