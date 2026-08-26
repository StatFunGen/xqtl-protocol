"""Notebook tier (integration): PCA.ipynb flashpca step over PCA.R (flashpcaR).

Runs flashPCA end-to-end on the LD-pruned unrelated bed and regression-compares the
EIGENVALUES + structure — NOT the eigenvectors. On this data flashpcaR's eigenvectors
are numerically ill-determined (see test_pca_flashpca), so only the eigenvalues are a
cross-platform-stable invariant worth pinning.

Fixture: tests/fixtures/pca/protocol_example.unrelated.prune.{bed,bim,fam} — the MWE
59-sample LD-pruned unrelated genotype.
"""
from __future__ import annotations

import os
import subprocess

from helpers.expected import assert_matches_expected

WORKER = "code/script/data_preprocessing/genotype/PCA.R"
NB = "pipeline/PCA.ipynb"
FIX = "tests/fixtures/pca"
EXP = f"{FIX}/expected"
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
    # structure: a flashpca model + one PC-score row per sample (the bed has 59), with 20
    # eigenvalues and the eigenvector matrices present at the expected shapes (loadings
    # n_variants x 20, vectors n_samples x 20) — so a regression that drops or reshapes
    # them is still caught even though their VALUES are not compared (see below).
    n = _rscript(
        f'r <- readRDS("{out}"); pm <- r$pca_model; '
        'stopifnot(is.list(r), "flashpca" %in% class(pm), is.data.frame(r$pc_scores), '
        'length(pm$values) == 20L, ncol(as.matrix(pm$loadings)) == 20L, '
        'ncol(as.matrix(pm$vectors)) == 20L); cat(nrow(r$pc_scores))')
    assert int(n) == 59

    # regression: value-compare only the EIGENVALUES (+ pve), NOT the eigenvectors.
    # flashpcaR's eigenVECTORS (vectors / loadings / projection / PC scores) are
    # numerically ILL-DETERMINED on this data: the eigenVALUES are near-degenerate (all in
    # [1.13, 1.58], ~2% gaps), so a tiny cross-platform BLAS rounding difference rotates and
    # sign-flips them. Verified against CI: macOS vs Linux differ ~30% on the eigenvectors
    # with full sign flips on PC9/11/14, WHILE the eigenvalues match to < 1e-6. The
    # eigenvalues/pve are therefore the well-determined, cross-platform-stable invariant.
    eig = tmp_path / "eigenvalues.tsv"
    _rscript(
        f'r <- readRDS("{out}"); pm <- r$pca_model; '
        f'writeLines(c("component\\tvalue\\tpve", '
        f'paste(seq_along(pm$values), pm$values, pm$pve, sep="\\t")), "{eig}")')
    assert_matches_expected(eig, repo_root / EXP / "flashpca.eigenvalues.tsv",
                            mode="tolerant", rtol=1e-5, atol=1e-8)


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
    # regression: the projected PC scores reproduce the committed fixture (deterministic).
    assert_matches_expected(out, repo_root / EXP / "project_samples.rds", mode="tolerant", rtol=1e-6, atol=1e-8)


def test_pca_detect_outliers(run_r, repo_root, tmp_path):
    # Mahalanobis outlier detection on the flashpca model
    fix = repo_root / FIX
    maha = tmp_path / "maha.rds"
    outliers = tmp_path / "outliers.txt"
    p = run_r(repo_root / WORKER, [
        "--step", "detect_outliers", "--cwd", tmp_path, "--pca-result", fix / f"{BASE}.pca.rds",
        "--prob", "0.975", "--pval", "0.05", "--robust", "FALSE", "--pop-col", "", "--k", "20",
        "--distance-output", maha, "--identified-outliers-output", outliers,
        # Both plot paths default to <pca-result>.mahalanobis_{qq,hist}.png, i.e.
        # *inside* the committed fixture dir; point them at tmp_path so a suite
        # run does not rewrite the checked-in PNGs.
        "--qqplot-output", tmp_path / "maha_qq.png",
        "--hist-output", tmp_path / "maha_hist.png"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert maha.exists() and outliers.exists()
    # regression: the Mahalanobis distances (tolerant) and the identified-outlier
    # sample list (exact) reproduce the committed fixtures (deterministic).
    assert_matches_expected(maha, repo_root / EXP / "detect_outliers.maha.rds", mode="tolerant", rtol=1e-6, atol=1e-8)
    assert_matches_expected(outliers, repo_root / EXP / "detect_outliers.outliers.txt", mode="exact")


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
    # regression: the plink2 --pca eigenvectors (FID/IID + PC1..PC20) reproduce the
    # committed fixture; PC loadings are floats -> tolerant across plink2 builds.
    assert_matches_expected(eigenvec, repo_root / EXP / "pca_plink.eigenvec", mode="tolerant", rtol=1e-6, atol=1e-8)
