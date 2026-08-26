"""Notebook tier: colocboost.ipynb (xQTL colocalization) on qtl_mini.

xQTL-only coloc (--no-separate-gwas --xqtl-coloc) needs no GWAS, so it reuses the
committed qtl_mini fixtures. The colocboost_3 cell forwards the notebook's ``seed``
parameter (default 999, the repo-wide seed convention) to ``colocboost.R --seed``, so
the stochastic colocboost fit is reproducible run-to-run. The output list still carries
a live ``computing_time`` (wall-clock difftimes), but the regression comparator strips
time-like leaves (tests/helpers/rds_compare.R), so a tolerant RDS compare is stable.
"""
from __future__ import annotations

from helpers.expected import assert_matches_expected

GENE = "ENSG00000283047"
EXPECTED = "tests/fixtures/colocboost/expected"


def test_colocboost_xqtl(run_sos, read_rds, repo_root, qtl_mini, tmp_path):
    cwd = tmp_path / "coloc"
    p = run_sos(
        repo_root / "pipeline/colocboost.ipynb", "colocboost",
        {
            "name": "test_coloc",
            "cwd": cwd,
            "genoFile": qtl_mini / "protocol_example.genotype.chr22.bed",
            "phenoFile": qtl_mini / "protocol_example.pheno_manifest_context.tsv",
            "covFile": qtl_mini / "example_covariates.tsv",
            "transpose-covariates": True,
            "customized-association-windows": qtl_mini / "association_windows.bed",
            "region-name": GENE,
            "no-separate-gwas": True,
            "xqtl-coloc": True,
            "modular_script_dir": repo_root / "code/script",
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    out = cwd / f"colocboost/test_coloc.{GENE}.colocboost.rds"
    assert out.exists(), p.stdout
    info = read_rds(out)
    assert info["class"] == "ColocBoostResult"
    assert "isColocalized" in info["colnames"]
    # regression: the seeded (seed=999) colocboost result reproduces the committed
    # snapshot within tolerance; the comparator strips computing_time difftimes.
    assert_matches_expected(out, repo_root / EXPECTED / f"test_coloc.{GENE}.colocboost.rds",
                            mode="tolerant", rtol=1e-6, atol=1e-8)
