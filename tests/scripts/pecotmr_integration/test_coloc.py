"""Tier B: coloc.R -> per-pair coloc data.frame (PP.H0..H4) from the committed
enloc QTL + GWAS S4 FineMappingResult fixtures (the modern format the pipeline
produces; the legacy-conversion step has been retired)."""
from __future__ import annotations

FX = "tests/fixtures/susie_enloc"
QTL = FX + "/protocol_example.enloc.MiGA_eQTL.ENSG00000142798.fine_mapping.rds"
GWAS = FX + "/protocol_example.enloc.gwas.chr1_20110062_22020160.fine_mapping.rds"


def test_coloc(run_r, read_rds, repo_root, tmp_path):
    out = tmp_path / "coloc.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/coloc.R",
              ["--qtl-fine-mapping", repo_root / QTL,
               "--gwas-input", repo_root / GWAS, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "data.frame" and "PP.H4.abf" in info["names"]
