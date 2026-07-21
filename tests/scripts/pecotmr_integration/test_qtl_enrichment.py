"""Tier B: qtl_enrichment.R -> enrichment data.frame from the committed enloc
QTL + GWAS S4 FineMappingResult fixtures (legacy conversion retired)."""
from __future__ import annotations

FX = "tests/fixtures/susie_enloc"
QTL = FX + "/protocol_example.enloc.MiGA_eQTL.ENSG00000142798.fine_mapping.rds"
GWAS = FX + "/protocol_example.enloc.gwas.chr1_20110062_22020160.fine_mapping.rds"


def test_qtl_enrichment(run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "enrich.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/qtl_enrichment.R",
              ["--qtl-fine-mapping", repo_root / QTL,
               "--gwas-fine-mapping", repo_root / GWAS, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "data.frame"
