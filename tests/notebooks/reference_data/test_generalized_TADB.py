"""Notebook tier: generalized_TADB.ipynb default workflow via SoS on committed chr22 fixtures.
The worker is code/script/reference_data/generalized_TADB.R (recursive TAD merge + TADB windows).
"""
from __future__ import annotations

from helpers.expected import assert_matches_expected

NB = "pipeline/generalized_TADB.ipynb"
FX = "tests/fixtures/generalized_TADB"
EXPECTED = "tests/fixtures/generalized_TADB/expected"
# All four outputs are deterministic (dplyr recursive TAD merge, no RNG) with no embedded paths.
OUTPUTS = ("generalized_TAD.tsv", "generalized_TADB.tsv",
           "TADB_enhanced_cis.bed", "extended_TADB.bed")


def test_default(run_sos, repo_root, tmp_path):
    out = tmp_path / "tadb"
    p = run_sos(repo_root / NB, "default",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     tad_input=repo_root / FX / "protocol_example.brain_TADs.txt",
                     gene_coords=repo_root / FX / "protocol_example.gene_start_end.tsv"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    for f in OUTPUTS:
        out_f = out / f
        assert out_f.exists() and out_f.stat().st_size > 0, f
        # regression: deterministic TAD-merge outputs reproduce the committed tables (cell-wise
        # numeric tolerance; string cells exact). No embedded absolute paths.
        assert_matches_expected(out_f, repo_root / EXPECTED / f, mode="tolerant",
                                rtol=1e-6, atol=1e-8)
