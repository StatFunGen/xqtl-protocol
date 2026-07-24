"""Notebook tier: generalized_TADB.ipynb default workflow via SoS on committed chr22 fixtures.
The worker is code/script/reference_data/generalized_TADB.R (recursive TAD merge + TADB windows).
"""
from __future__ import annotations

NB = "pipeline/generalized_TADB.ipynb"
FX = "tests/fixtures/generalized_TADB"


def test_default(run_sos, repo_root, tmp_path):
    out = tmp_path / "tadb"
    p = run_sos(repo_root / NB, "default",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     tad_input=repo_root / FX / "protocol_example.brain_TADs.txt",
                     gene_coords=repo_root / FX / "protocol_example.gene_start_end.tsv"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    for f in ("generalized_TAD.tsv", "generalized_TADB.tsv",
              "TADB_enhanced_cis.bed", "extended_TADB.bed"):
        out_f = out / f
        assert out_f.exists() and out_f.stat().st_size > 0, f
