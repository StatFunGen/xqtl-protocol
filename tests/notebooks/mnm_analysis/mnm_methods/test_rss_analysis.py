"""Notebook tier: rss_analysis.ipynb (GWAS RSS fine-mapping) on committed fixtures.

Drives the full manifest -> gwas-sumstats -> fine-mapping chain from a toy chr22
GWAS + LD reference panel (tests/fixtures/rss_analysis + ld_reference).
"""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

STUB = "AD_Bellenguez_2022.chr22_49355984_50799822"


def test_gwas_finemapping(run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "rss"
    fx = repo_root / "tests" / "fixtures"
    p = run_sos(
        repo_root / "pipeline/rss_analysis.ipynb",
        "generate_manifest+generate_gwas_sumstats+gwas_fine_mapping",
        {
            "cwd": cwd,
            "modular-script-dir": repo_root / "code/script",
            "gwas-meta": fx / "rss_analysis/gwas_meta_data.tsv",
            "regions": "chr22:49355984-50799822",
            "ld-meta": fx / "ld_reference/ld_meta_file.tsv",
        },
        cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr

    ss = cwd / f"sumstats/{STUB}.gwas_sumstats.rds"
    fm = cwd / f"fine_mapping/{STUB}.gwas_finemap.rds"
    assert ss.exists() and fm.exists(), p.stdout
    assert read_rds(ss)["class"] == "GwasSumStats"
    info = read_rds(fm)
    assert info["class"] == "GwasFineMappingResult"
    assert info["MethodNames"] == ["susie"]

    # regression: the GwasSumStats (deterministic QC/reshaping) and the susie_rss
    # GwasFineMappingResult (deterministic for this fixed input+init) reproduce the
    # committed expected RDS. normalize_paths basename-normalizes the embedded
    # ld_reference LD-panel path (resolved from --ld-meta at the checkout location).
    exp = repo_root / "tests/fixtures/rss_analysis/expected"
    assert_matches_expected(ss, exp / "gwas_sumstats.rds", mode="tolerant",
                            rtol=1e-6, atol=1e-8, normalize_paths=True)
    assert_matches_expected(fm, exp / "gwas_finemap.rds", mode="tolerant",
                            rtol=1e-6, atol=1e-8, normalize_paths=True)
