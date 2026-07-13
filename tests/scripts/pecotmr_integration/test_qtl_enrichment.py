"""Tier B: qtl_enrichment.R -> enrichment data.frame from the converted enloc
QTL + GWAS S4 sides."""
from __future__ import annotations

import pytest

def _convert(run_r, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/susie_enloc"
    conv = repo_root / "code/script/pecotmr_integration/legacy_enloc_finemap_convert.R"
    qtl = tmp_path / "qtl.rds"
    assert run_r(conv, ["--mode", "qtl",
        "--rds-files", fx / "protocol_example.enloc.MiGA_eQTL.ENSG00000142798.univariate_susie_twas_weights.rds",
        "--finemapping-obj", "preset_variants_result susie_result_trimmed",
        "--varname-obj", "preset_variants_result variant_names",
        "--study", "protocol_example", "--output", qtl], timeout=300).returncode == 0
    gwas = tmp_path / "gwas.rds"
    assert run_r(conv, ["--mode", "gwas",
        "--rds-files", fx / "protocol_example.enloc.RSS_QC_RAISS_imputed.chr1_20110062_22020160.univariate_susie_rss.rds",
        "--finemapping-obj", "AD_Bellenguez_2022 RSS_QC_RAISS_imputed susie_result_trimmed",
        "--varname-obj", "AD_Bellenguez_2022 RSS_QC_RAISS_imputed variant_names",
        "--output", gwas], timeout=300).returncode == 0
    return qtl, gwas


def test_qtl_enrichment(run_r, read_rds, repo_root, tmp_path):
    qtl, gwas = _convert(run_r, repo_root, tmp_path)
    out = tmp_path / "enrich.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/qtl_enrichment.R",
              ["--qtl-fine-mapping", qtl, "--gwas-fine-mapping", gwas, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "data.frame"
