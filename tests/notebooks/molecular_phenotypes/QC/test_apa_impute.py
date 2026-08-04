"""Notebook tier: apa_impute.ipynb APAimpute + APArename via SoS on the committed fixture.
The worker is code/script/molecular_phenotypes/QC/apa_impute.R. The toy DaPars2 result is
empty (no APA events called), so this exercises the graceful empty-output path.
"""
from __future__ import annotations

import shutil

NB = "pipeline/apa_impute.ipynb"
FX = "tests/fixtures/apa_impute"


def test_apa_impute(run_sos, repo_root, tmp_path):
    cwd = tmp_path / "apa"
    (cwd / "apa_chr22").mkdir(parents=True)
    shutil.copy(repo_root / FX / "apa_chr22" / "Dapars_result_result_temp.chr22.txt",
                cwd / "apa_chr22" / "Dapars_result_result_temp.chr22.txt")
    base = dict(cwd=cwd, modular_script_dir=repo_root / "code/script", chrlist="chr22")

    p = run_sos(repo_root / NB, "APAimpute", base, cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (cwd / "apa_chr22" / "Dapars_result_impute_chr22.bed").exists()
    assert (cwd / "Dapars_allchrom.bed").exists()

    p = run_sos(repo_root / NB, "APArename",
                {**base, "match": repo_root / FX / "protocol_example.apa_matchtable.txt"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (cwd / "apa_chr22" / "Dapars_result_impute_renamed_chr22.bed.gz").exists()
    assert (cwd / "Dapars_allchrom_renamed.bed").exists()
