"""Notebook tier: colocboost.ipynb (xQTL colocalization) on qtl_mini.

xQTL-only coloc (--no-separate-gwas --xqtl-coloc) needs no GWAS, so it reuses the
committed qtl_mini fixtures.
"""
from __future__ import annotations

import pytest

GENE = "ENSG00000283047"


def test_colocboost_xqtl(run_sos, read_rds, repo_root, qtl_mini, tmp_path):
    cwd = tmp_path / "coloc"
    p = run_sos(
        repo_root / "pipeline/colocboost.ipynb", "colocboost",
        {
            "name": "test_coloc",
            "cwd": cwd,
            "genoFile": qtl_mini / "example.chr22.bed",
            "phenoFile": qtl_mini / "pheno_manifest_multicontext.tsv",
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
    assert info["class"] == "list"
    assert "xqtl_coloc" in info["names"]
