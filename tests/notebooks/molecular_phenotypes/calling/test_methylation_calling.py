"""Notebook tier: methylation_calling.ipynb sesame + minfi workflows via SoS.
The worker is code/script/molecular_phenotypes/calling/methylation_calling.R (SeSAMe/minfi
450K calling + probe->gene BED). The repo has no committed methylation IDATs (the notebook's
data/MWE/ was never provided), so IDATs are sourced from the installed `minfiData` package
(450K, matching the installed manifest). sesame also needs a one-time sesameData cache (network).
minfi's preprocessQuantile needs several samples, so the minfi test uses all 6.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

NB = "pipeline/methylation_calling.ipynb"
CROSS = "data/cross_reactive_probe_Hop2020.txt"
SAMPLES = [("5723646052", "R02C02", "GroupA_3"), ("5723646052", "R04C01", "GroupA_2"),
           ("5723646052", "R05C02", "GroupB_3"), ("5723646053", "R04C02", "GroupB_1"),
           ("5723646053", "R05C02", "GroupA_1"), ("5723646053", "R06C02", "GroupB_2")]


def _minfidata_extdata(repo_root) -> Path:
    out = subprocess.run(
        ["pixi", "run", "Rscript", "-e", 'cat(system.file("extdata", package="minfiData"))'],
        cwd=repo_root, capture_output=True, text=True)
    for tok in out.stdout.split():
        if "minfiData" in tok and "extdata" in tok:
            return Path(tok)
    raise RuntimeError("minfiData extdata not found: " + out.stdout + out.stderr)


def _stage(repo_root, work: Path, n: int):
    ext = _minfidata_extdata(repo_root)
    for sid, pos, _ in SAMPLES[:n]:
        for c in ("Grn", "Red"):
            shutil.copy(ext / sid / f"{sid}_{pos}_{c}.idat", work / f"{sid}_{pos}_{c}.idat")
    lines = ["Sample_Name,Sentrix_ID,Sentrix_Position"]
    lines += [f"{name},{sid},{pos}" for sid, pos, name in SAMPLES[:n]]
    (work / "sample_sheet.csv").write_text("\n".join(lines) + "\n")


def test_sesame(run_sos, repo_root, tmp_path):
    work = tmp_path / "data"; work.mkdir()
    _stage(repo_root, work, n=2)
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "sesame",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     sample_sheet=work / "sample_sheet.csv", sample_sheet_header_rows=0),
                cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "sample_sheet.sesame.beta.tsv").exists()
    assert (out / "sample_sheet.sesame.beta.bed.gz").exists()
    assert (out / "sample_sheet.sesame.beta.bed.gz.tbi").exists()


def test_minfi(run_sos, repo_root, tmp_path):
    work = tmp_path / "data"; work.mkdir()
    _stage(repo_root, work, n=6)
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "minfi",
                dict(cwd=out, modular_script_dir=repo_root / "code/script",
                     sample_sheet=work / "sample_sheet.csv"),
                cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "sample_sheet.minfi.beta.tsv").exists()
    assert (out / "sample_sheet.minfi.beta.bed.gz").exists()
