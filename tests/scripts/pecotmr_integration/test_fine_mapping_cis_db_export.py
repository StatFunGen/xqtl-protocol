"""Script tier: fine_mapping_cis_db_export.R — combine per-region FineMappingResult
RDS(s) into a cis-results DB (combined FMR) + a meta TSV."""
from __future__ import annotations

from helpers.expected import assert_matches_expected

FMR = "tests/fixtures/mnm_postprocessing/protocol_example.ENSG00000283047.fine_mapping.rds"
# Both outputs are a deterministic combine/passthrough of a committed FMR (no RNG).
# The meta TSV carries only basenames (original_data / combined_data), and the
# combined_data column is the fixed output filename this test names — no embedded
# absolute paths in either output.
EXPECTED = "tests/fixtures/fine_mapping/expected"


def test_fine_mapping_cis_db_export(run_r, read_rds, repo_root, tmp_path):
    combined = tmp_path / "cis_results_db.rds"
    meta = tmp_path / "cis_results_db.meta.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_cis_db_export.R",
              ["--input", repo_root / FMR,
               "--combined-data-output", combined,
               "--meta-output", meta, "--signal-cutoff", "0"], timeout=200)
    assert p.returncode == 0, p.stdout + p.stderr
    assert combined.exists() and meta.exists()
    assert meta.read_text().splitlines()                     # header at least
    assert_matches_expected(combined, repo_root / EXPECTED / "cis_results_db.rds",
                            mode="tolerant", rtol=1e-6, atol=1e-8)
    assert_matches_expected(meta, repo_root / EXPECTED / "cis_results_db.meta.tsv",
                            mode="tolerant", rtol=1e-6, atol=1e-8)
