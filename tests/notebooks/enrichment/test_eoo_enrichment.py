"""Notebook tier: eoo_enrichment.ipynb enrichment step via SoS on committed fixtures.
The worker is code/script/enrichment/eoo_enrichment.R (LOCO block-jackknife OR/enrichment).
"""
from __future__ import annotations

NB = "pipeline/eoo_enrichment.ipynb"
FX = "tests/fixtures/eoo_enrichment"


def test_enrichment(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "enrichment",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     significant_variants_path=repo_root / FX / "protocol_example.eoo_significant_variants.tsv.gz",
                     baseline_anno_path=repo_root / FX / "protocol_example.eoo_baseline_annotation.tsv.gz",
                     trait="protocol_example", annotation_name="baseline"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    d = out / "enrichment"
    assert (d / "protocol_example.baseline.enrichment_results.rds").exists()
    assert (d / "protocol_example.baseline.enrichment_results_summary.tsv.gz").exists()
