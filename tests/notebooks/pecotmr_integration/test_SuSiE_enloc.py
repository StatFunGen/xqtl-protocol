"""Notebook tier: SuSiE_enloc.ipynb (xQTL x GWAS enrichment + colocalization).

Drives both wrapper steps on a downsized fixture (one gene ENSG00000142798, its
MiGA + KNIGHT QTL fine-maps, and the two overlapping chr1 GWAS RSS blocks -- 21M
vs the 524M full set). Exercises the shared-var enloc_manifest.R providers
(get_analysis_regions / get_overlapped_analysis_regions), the per-condition study
routing (MiGA conditions -> MiGA file, Knight -> KNIGHT file), and the modern
FMR path: the *.fmr.tsv metas point at committed S4 FineMappingResult fixtures,
so qtl_enrichment.R / coloc.R consume them directly (no legacy conversion).
"""
from __future__ import annotations

import pytest


def test_enrichment_and_coloc(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures" / "susie_enloc"
    cwd = tmp_path / "enloc"
    base = dict(
        gwas_meta_data=fx / "protocol_example.enloc.gwas_meta.fmr.tsv",
        xqtl_meta_data=fx / "protocol_example.enloc.xqtl_meta.fmr.tsv",
        context_meta=fx / "protocol_example.enloc.context_meta.tsv",
        qtl_path=fx, gwas_path=fx, cwd=cwd, name="protocol_example")

    # (1) enrichment: get_analysis_regions builds a 4-unit manifest (MiGA x3 +
    # Knight, each context-routed to its study's QTL file); xqtl_gwas_enrichment
    # fans out into one enrichment.rds per condition.
    p = run_sos(repo_root / "pipeline/SuSiE_enloc.ipynb", "xqtl_gwas_enrichment",
                base, cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    for cond in ("MiGA_GTS_eQTL", "MiGA_SVZ_eQTL", "MiGA_THA_eQTL", "Knight_eQTL_brain"):
        assert (cwd / f"protocol_example.{cond}.enrichment.rds").exists(), p.stdout
    info = read_rds(cwd / "protocol_example.MiGA_GTS_eQTL.enrichment.rds")
    assert info["class"] == "data.frame"
    assert "enrichment" in info["names"], info

    # (2) coloc: get_overlapped_analysis_regions builds a 4-unit manifest
    # (condition@region); susie_coloc reuses the enrichment.rds and emits one
    # coloc data.frame (PP.H0..H4) per unit.
    p = run_sos(repo_root / "pipeline/SuSiE_enloc.ipynb", "susie_coloc",
                base, cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    cf = cwd / "susie_coloc/protocol_example.MiGA_GTS_eQTL@ENSG00000142798.coloc.rds"
    assert cf.exists(), p.stdout
    cinfo = read_rds(cf)
    assert cinfo["class"] == "data.frame"
    assert "PP.H4.abf" in cinfo["names"], cinfo
