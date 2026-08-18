"""Notebook tier: gsea.ipynb pathway_analysis driven via SoS on the committed fixture.
The worker is code/script/enrichment/gsea.R (clusterProfiler KEGG + GO). KEGG queries the
online KEGG API (degrades gracefully via tryCatch); GO uses the local org.Hs.eg.db, so the
step still produces results offline. Requires clusterProfiler + org.Hs.eg.db in the env.

Value regression is on the GO-only rows: enrichGO runs entirely off the local
org.Hs.eg.db and is deterministic run-to-run, whereas the KEGG rows come from the
online KEGG REST API and their row count drifts, so KEGG is excluded from the
comparison (existence of the combined RDS is still asserted).
"""
from __future__ import annotations

import subprocess

from helpers.expected import assert_matches_expected
from helpers.r_runner import rscript_bin

NB = "pipeline/gsea.ipynb"
FX = "tests/fixtures/gsea"

# Filter the combined result to analysis_type=="GO", sort by (group, ont_category, ID)
# and reset row.names, then save — the same stable GO-only view the fixture was built as.
_GO_PROJECT = r'''
a <- commandArgs(TRUE)
x <- readRDS(a[1])
go <- x[x$analysis_type == "GO", , drop = FALSE]
go <- go[order(go$group, go$ont_category, go$ID), , drop = FALSE]
rownames(go) <- NULL
saveRDS(go, a[2], compress = "xz")
'''


def test_pathway_analysis(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "pathway_analysis",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     genes_file=repo_root / FX / "protocol_example.pathway_genes.tsv",
                     name="protocol_example", pvalue_cutoff=1, organism="hsa"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    combined = out / "pathway_analysis" / "protocol_example.combined_pathway_results.rds"
    assert combined.exists()
    # regression: compare only the deterministic GO rows (KEGG is network-varying).
    go_only = tmp_path / "go_only.rds"
    r = subprocess.run([rscript_bin(), "-e", _GO_PROJECT, str(combined), str(go_only)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert_matches_expected(go_only, repo_root / FX / "expected" / "pathway_go_results.rds",
                            mode="tolerant", rtol=1e-6, atol=1e-8)
