"""Notebook tier: gsea.ipynb pathway_analysis driven via SoS on the committed fixture.
The worker is code/script/enrichment/gsea.R (clusterProfiler KEGG + GO). KEGG queries the
online KEGG API (degrades gracefully via tryCatch); GO uses the local org.Hs.eg.db, so the
step still produces results offline. Requires clusterProfiler + org.Hs.eg.db in the env.
"""
from __future__ import annotations

NB = "pipeline/gsea.ipynb"
FX = "tests/fixtures/gsea"


def test_pathway_analysis(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "pathway_analysis",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     genes_file=repo_root / FX / "protocol_example.pathway_genes.tsv",
                     name="protocol_example", pvalue_cutoff=1, organism="hsa"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "pathway_analysis" / "protocol_example.combined_pathway_results.rds").exists()
