"""Script tier: fine_mapping_cis_db_export.R — combine per-region FineMappingResult
RDS(s) into a cis-results DB (combined FMR) + a meta TSV."""
from __future__ import annotations

FMR = "tests/fixtures/mnm_postprocessing/protocol_example.ENSG00000283047.fine_mapping.rds"


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
