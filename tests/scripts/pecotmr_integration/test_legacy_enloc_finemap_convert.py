"""Tier B: legacy_enloc_finemap_convert.R extracts the QTL SuSiE fit from a
legacy enloc RDS into the pecotmr S4 collection coloc/enrichment consume."""
from __future__ import annotations

import pytest

S = "code/script/pecotmr_integration"


def test_legacy_enloc_finemap_convert_qtl(run_r, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/susie_enloc"
    out = tmp_path / "qtl.rds"
    p = run_r(repo_root / f"{S}/legacy_enloc_finemap_convert.R",
              ["--mode", "qtl",
               "--rds-files", fx / "protocol_example.enloc.MiGA_eQTL.ENSG00000142798.univariate_susie_twas_weights.rds",
               "--finemapping-obj", "preset_variants_result susie_result_trimmed",
               "--varname-obj", "preset_variants_result variant_names",
               "--study", "protocol_example", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.stat().st_size > 0
