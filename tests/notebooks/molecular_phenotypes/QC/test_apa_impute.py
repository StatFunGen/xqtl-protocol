"""Notebook tier: apa_impute.ipynb APAimpute + APArename via SoS on the committed fixture.
The worker is code/script/molecular_phenotypes/QC/apa_impute.R. The toy DaPars2 result is
empty (no APA events called), so this exercises the graceful empty-output path.
"""
from __future__ import annotations

import shutil

from helpers.expected import assert_matches_expected

NB = "pipeline/apa_impute.ipynb"
FX = "tests/fixtures/apa_impute"
EXP = "tests/fixtures/apa_impute/expected"


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
    # regression: on the empty toy DaPars2 result the graceful empty-output path is
    # deterministic -> both BEDs are the bare `#chr start end Gene` header, exact.
    assert_matches_expected(cwd / "apa_chr22" / "Dapars_result_impute_chr22.bed",
                            repo_root / EXP / "expected.Dapars_result_impute_chr22.bed", mode="exact")
    assert_matches_expected(cwd / "Dapars_allchrom.bed",
                            repo_root / EXP / "expected.Dapars_allchrom.bed", mode="exact")

    p = run_sos(repo_root / NB, "APArename",
                {**base, "match": repo_root / FX / "protocol_example.apa_matchtable.txt"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (cwd / "apa_chr22" / "Dapars_result_impute_renamed_chr22.bed.gz").exists()
    assert (cwd / "Dapars_allchrom_renamed.bed").exists()
    # regression: the rename/bgzip of the empty BED is deterministic (bare header).
    # .bed.gz is bgzip'd -> assert_matches_expected decompresses before the byte compare.
    assert_matches_expected(cwd / "apa_chr22" / "Dapars_result_impute_renamed_chr22.bed.gz",
                            repo_root / EXP / "expected.Dapars_result_impute_renamed_chr22.bed.gz", mode="exact")
    assert_matches_expected(cwd / "Dapars_allchrom_renamed.bed",
                            repo_root / EXP / "expected.Dapars_allchrom_renamed.bed", mode="exact")
