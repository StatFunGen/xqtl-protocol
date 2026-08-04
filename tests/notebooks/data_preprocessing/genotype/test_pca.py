"""Notebook tier (integration): PCA.ipynb flashpca step over PCA.R (flashpcaR).

Runs flashPCA end-to-end on the LD-pruned unrelated bed and checks the RDS holds a
flashpca model + one PC-score row per sample. flashpcaR's numeric PCs aren't pinned
to a committed reference here (the MWE reference predates this PCA.R output shape), so
this is a structural integration check that the worker + package run.

Fixture: tests/fixtures/pca/protocol_example.unrelated.prune.{bed,bim,fam} — the MWE
59-sample LD-pruned unrelated genotype.
"""
from __future__ import annotations

import os
import subprocess

WORKER = "code/script/data_preprocessing/genotype/PCA.R"
NB = "pipeline/PCA.ipynb"
FIX = "tests/fixtures/pca"
BASE = "protocol_example.unrelated.prune"


def _rscript(expr):
    r = subprocess.run([os.environ.get("XQTL_RSCRIPT", "Rscript"), "-e", expr],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def test_pca_flashpca(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    out = tmp_path / "out.pca.rds"
    p = run_r(repo_root / WORKER, [
        "--step", "flashpca", "--cwd", tmp_path,
        "--genoFile", fix / f"{BASE}.bed", "--phenoFile", fix / f"{BASE}.fam",
        "--output", out, "--stand", "binom2", "--min-pop-size", "2",
        "--homogeneous", "True", "--pop-col", "", "--label-col", "", "--pops", "",
        "--k", "20", "--maha-k", "5", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    # RDS: a flashpca model + one PC-score row per sample (the bed has 59)
    n = _rscript(
        f'r <- readRDS("{out}"); '
        'stopifnot(is.list(r), "flashpca" %in% class(r$pca_model), '
        'is.data.frame(r$pc_scores)); cat(nrow(r$pc_scores))')
    assert int(n) == 59


def test_pca_project_samples(run_r, repo_root, tmp_path):
    # project samples onto the committed flashpca model
    fix = repo_root / FIX
    out = tmp_path / "proj.rds"
    p = run_r(repo_root / WORKER, [
        "--step", "project_samples", "--cwd", tmp_path, "--genoFile", fix / f"{BASE}.bed",
        "--phenoFile", fix / f"{BASE}.fam", "--output", out, "--stand", "binom2",
        "--pop-col", "", "--label-col", "", "--pops", "", "--pca-model", fix / f"{BASE}.pca.rds"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()


def test_pca_detect_outliers(run_r, repo_root, tmp_path):
    # Mahalanobis outlier detection on the flashpca model
    fix = repo_root / FIX
    maha = tmp_path / "maha.rds"
    outliers = tmp_path / "outliers.txt"
    p = run_r(repo_root / WORKER, [
        "--step", "detect_outliers", "--cwd", tmp_path, "--pca-result", fix / f"{BASE}.pca.rds",
        "--prob", "0.975", "--pval", "0.05", "--robust", "FALSE", "--pop-col", "", "--k", "20",
        "--distance-output", maha, "--identified-outliers-output", outliers])
    assert p.returncode == 0, p.stdout + p.stderr
    assert maha.exists() and outliers.exists()


def test_pca_plot(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    p = run_r(repo_root / WORKER, [
        "--step", "plot_pca", "--cwd", tmp_path, "--plot-data", fix / f"{BASE}.pca.rds",
        "--outlier-file", "", "--min-axis", "", "--max-axis", "", "--pop-col", "",
        "--label-col", "", "--pops", "", "--k", "20"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / f"{BASE}.pca.pc.png").exists()          # PC scatter PNG written


def test_pca_plink(run_sos, repo_root, tmp_path):
    # inline plink2 --pca step (no worker) -> eigenvec, via the SoS cell
    bed = repo_root / FIX / f"{BASE}.bed"
    p = run_sos(repo_root / NB, "pca_plink", {
        "genoFile": bed, "cwd": tmp_path, "name": "test",
        "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    eigenvec = tmp_path / f"{BASE}.pca.eigenvec"
    assert eigenvec.exists() and sum(1 for _ in open(eigenvec)) == 60   # 59 samples + header
