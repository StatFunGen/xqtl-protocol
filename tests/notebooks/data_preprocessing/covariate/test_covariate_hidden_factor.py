"""Notebook tier: covariate_hidden_factor.ipynb — hidden factor analysis.

Worker: code/script/data_preprocessing/covariate/covariate_hidden_factor.R
  (compute_residual / Marchenko_PC, the latter also providing the Buja_Eyuboglu
  permutation route via --choose-k-method). Fixtures: the committed 200-gene rnaseq
  phenotype BED and a #id-headed covariate matrix derived from qtl_mini (49 shared
  samples).
"""
from __future__ import annotations

import gzip

import pytest

from helpers.expected import assert_matches_expected

R = "code/script/data_preprocessing/covariate/covariate_hidden_factor.R"
NB = "pipeline/covariate_hidden_factor.ipynb"
PHENO = "tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz"
COV = "tests/fixtures/covariate_hidden_factor/covariates.tsv"
MSD = "code/script"
EXPECTED_RESIDUAL = "tests/fixtures/covariate_hidden_factor/expected/residual.bed.gz"
EXPECTED_MARCHENKO = "tests/fixtures/covariate_hidden_factor/expected/Marchenko_PC.gz"
EXPECTED_BUJA = "tests/fixtures/covariate_hidden_factor/expected/Buja_Eyuboglu_PC.gz"


def _col1(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [ln.split("\t", 1)[0] for ln in fh if ln.strip()]


def _residual(run_r, repo_root, tmp_path):
    """compute_residual -> residual BED (shared by several tests)."""
    out_dir = tmp_path / "resid"
    p = run_r(repo_root / R, ["--step", "compute_residual", "--cwd", out_dir,
                              "--phenoFile", repo_root / PHENO, "--covFile", repo_root / COV])
    assert p.returncode == 0, p.stdout + p.stderr
    resid = out_dir / "protocol_example.rnaseq.bed.covariates.residual.bed.gz"
    assert resid.exists()
    return resid


def test_compute_residual(run_r, repo_root, tmp_path):
    resid = _residual(run_r, repo_root, tmp_path)
    # 200 genes retained; header + coordinate columns preserved
    with gzip.open(resid, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        n_rows = sum(1 for _ in fh)
    assert header[:4] == ["#chr", "start", "end", "ID"]
    assert n_rows == 200
    assert len(header) - 4 == 49                     # 49 samples shared with covariates
    # regression: the residualized BED (.lm.fit, deterministic, no RNG/paths)
    # reproduces the committed fixture cell-for-cell within numeric tolerance.
    assert_matches_expected(resid, repo_root / EXPECTED_RESIDUAL, mode="tolerant",
                            rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", ["Marchenko", "Buja_Eyuboglu"])
def test_marchenko_pc(run_r, repo_root, tmp_path, method):
    resid = _residual(run_r, repo_root, tmp_path)
    out = tmp_path / f"out.{method}_PC.gz"
    p = run_r(repo_root / R, ["--step", "Marchenko_PC", "--cwd", tmp_path, "--residFile", resid,
                              "--covFile", repo_root / COV, "--choose-k-method", method,
                              "--output", out, "--N", "0", "--seed", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    rows = _col1(out)
    assert rows[0] == "#id"
    # known covariates are stacked above the inferred hidden factors
    assert {"sex", "age", "PC1"}.issubset(set(rows))
    assert any(r.startswith("Hidden_Factor_PC") for r in rows)
    # regression: both routes reproduce their committed factor table within tolerance.
    # Marchenko is closed-form; Buja_Eyuboglu goes through jackstraw::permutationPA(B=100)
    # whose B permutations are now seeded (--seed 1), making the selected factor count and
    # the PCs reproducible run-to-run. (Residual cross-arch risk: a permutation-threshold
    # eigenvalue sitting on the knife's edge could flip the factor count on a different
    # arch — that surfaces as a shape mismatch on CI and is calibrated there, not locally.)
    expected = {"Marchenko": EXPECTED_MARCHENKO, "Buja_Eyuboglu": EXPECTED_BUJA}[method]
    assert_matches_expected(out, repo_root / expected, mode="tolerant",
                            rtol=1e-6, atol=1e-8)


def test_marchenko_workflow_via_sos(run_sos, repo_root, tmp_path):
    p = run_sos(repo_root / NB, "Marchenko_PC", {
        "phenoFile": repo_root / PHENO, "covFile": repo_root / COV, "cwd": tmp_path,
        "N": 0, "numThreads": 1, "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(list(tmp_path.glob("*.Marchenko_PC.gz"))) == 1
