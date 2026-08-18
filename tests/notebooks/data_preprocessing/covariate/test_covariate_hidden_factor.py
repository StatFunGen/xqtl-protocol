"""Notebook tier: covariate_hidden_factor.ipynb — hidden factor analysis.

Workers:
  code/script/data_preprocessing/covariate/covariate_hidden_factor.R
    (compute_residual / Marchenko_PC, the latter also providing the Buja_Eyuboglu
    permutation route via --choose-k-method; PEER_extract)
  code/script/data_preprocessing/covariate/covariate_hidden_factor_peer.py
    (mofapy2 PEER fit + extraction of factors/weights/variance to TSV)

The PEER path fits mofapy2 in a standalone Python script (no reticulate / no MOFA2
Bioconductor package) and writes factor/weight/variance TSV sidecars; the R
PEER_extract step reads those to build the .PEER.gz covariate matrix and a ggplot
diagnostic PDF. Fixtures: the committed 200-gene rnaseq phenotype BED and a
#id-headed covariate matrix derived from qtl_mini (49 shared samples).
"""
from __future__ import annotations

import gzip
import subprocess
import sys

import pytest

from helpers.expected import assert_matches_expected

R = "code/script/data_preprocessing/covariate/covariate_hidden_factor.R"
PY = "code/script/data_preprocessing/covariate/covariate_hidden_factor_peer.py"
NB = "pipeline/covariate_hidden_factor.ipynb"
PHENO = "tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz"
COV = "tests/fixtures/covariate_hidden_factor/covariates.tsv"
MSD = "code/script"
EXPECTED_RESIDUAL = "tests/fixtures/covariate_hidden_factor/expected/residual.bed.gz"
EXPECTED_MARCHENKO = "tests/fixtures/covariate_hidden_factor/expected/Marchenko_PC.gz"
EXPECTED_BUJA = "tests/fixtures/covariate_hidden_factor/expected/Buja_Eyuboglu_PC.gz"
EXPECTED_PEER = "tests/fixtures/covariate_hidden_factor/expected/PEER.gz"
EXPECTED_PEER_SIDECARS = {
    "PEER.factors.tsv": "tests/fixtures/covariate_hidden_factor/expected/PEER.factors.tsv",
    "PEER.weights.tsv": "tests/fixtures/covariate_hidden_factor/expected/PEER.weights.tsv",
    "PEER.variance.tsv": "tests/fixtures/covariate_hidden_factor/expected/PEER.variance.tsv",
}
PEER_SEED = "1"


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


def _peer_fit(repo_root, cwd, resid, stem, seed=PEER_SEED, threads="1"):
    """mofapy2 fit -> HDF5 model + factors/weights/variance TSV sidecars."""
    return subprocess.run(
        [sys.executable, str(repo_root / PY),
         "--resid-file", str(resid), "--model-file", str(cwd / f"{stem}.PEER_MODEL.hd5"),
         "--factors-out", str(cwd / f"{stem}.PEER.factors.tsv"),
         "--weights-out", str(cwd / f"{stem}.PEER.weights.tsv"),
         "--variance-out", str(cwd / f"{stem}.PEER.variance.tsv"),
         "--num-factor", "5", "--iteration", "100", "--convergence-mode", "fast",
         "--num-threads", threads, "--tol", "0.001", "--r2-tol", "False", "--seed", seed],
        capture_output=True, text=True)


def test_peer_fit_extract(run_r, repo_root, tmp_path):
    resid = _residual(run_r, repo_root, tmp_path)
    cwd = tmp_path / "peer"; cwd.mkdir(parents=True, exist_ok=True)
    stem = "protocol_example.rnaseq.bed.covariates.residual"
    model = cwd / f"{stem}.PEER_MODEL.hd5"
    # PEER_fit: the notebook drives the standalone mofapy2 worker directly (no R shell-out);
    # exercise the same worker here via the interpreter running the tests (has mofapy2).
    p = _peer_fit(repo_root, cwd, resid, stem)
    assert p.returncode == 0, p.stdout + p.stderr
    assert model.exists()                                            # HDF5 model still saved
    for name, expected in EXPECTED_PEER_SIDECARS.items():
        produced = cwd / f"{stem}.{name}"
        assert produced.exists()
        # regression: mofapy2's variational fit is seeded through --seed, so the
        # factor/weight/variance tables reproduce the committed fixtures.
        assert_matches_expected(produced, repo_root / expected, mode="tolerant",
                                rtol=1e-6, atol=1e-8)
    # variance.tsv: per-factor rows + a Total row
    var_rows = _col1(cwd / f"{stem}.PEER.variance.tsv")
    assert var_rows[0] == "factor" and "Total" in var_rows and "Factor1" in var_rows

    # PEER_extract: TSV sidecars -> .PEER.gz (covariates + factors) + ggplot diag PDF
    p = run_r(repo_root / R, ["--step", "PEER_extract", "--cwd", cwd, "--modelFile", model,
                              "--covFile", repo_root / COV, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    peer_gz = cwd / f"{stem}.PEER.gz"
    diag = cwd / f"{stem}.PEER.diag.pdf"
    assert peer_gz.exists() and diag.exists()
    assert diag.stat().st_size > 0                                   # binary PDF: existence only
    rows = _col1(peer_gz)
    assert rows[0] == "#id"
    assert {"sex", "age", "PC1"}.issubset(set(rows))                 # known covariates
    assert [r for r in rows if r.startswith("Factor")] == [f"Factor{i}" for i in range(1, 6)]
    assert_matches_expected(peer_gz, repo_root / EXPECTED_PEER, mode="tolerant",
                            rtol=1e-6, atol=1e-8)


def test_peer_fit_is_seed_reproducible(run_r, repo_root, tmp_path):
    """Same --seed -> identical fit; the seed is what makes mofapy2 reproducible.

    Without it mofapy2 seeds itself off the clock (entry_point.set_train_options:
    ``seed = int(round(time() * 1000) % 1e6)``), so the variational initialisation —
    and every downstream factor — would drift run-to-run.
    """
    resid = _residual(run_r, repo_root, tmp_path)
    stem = "protocol_example.rnaseq.bed.covariates.residual"
    runs = []
    for tag, seed, threads in (("a", PEER_SEED, "1"), ("b", PEER_SEED, "4"), ("c", "1234", "1")):
        cwd = tmp_path / tag; cwd.mkdir(parents=True, exist_ok=True)
        p = _peer_fit(repo_root, cwd, resid, stem, seed=seed, threads=threads)
        assert p.returncode == 0, p.stdout + p.stderr
        runs.append((cwd / f"{stem}.PEER.factors.tsv").read_bytes())
    # same seed reproduces byte-for-byte, and is not perturbed by the thread count
    assert runs[0] == runs[1]
    # a different seed is a different initialisation: the run is genuinely seed-driven
    assert runs[0] != runs[2]


def test_peer_workflow_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run covariate_hidden_factor.ipynb PEER` chains PEER_1
    # (compute_residual) -> PEER_2 (mofapy2 fit) -> PEER_3 (extract) end to end.
    p = run_sos(repo_root / NB, "PEER", {
        "phenoFile": repo_root / PHENO, "covFile": repo_root / COV, "cwd": tmp_path,
        "N": 5, "iteration": 100, "numThreads": 1, "seed": PEER_SEED,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    peer_gz = list(tmp_path.glob("*.PEER.gz"))
    assert len(peer_gz) == 1
    assert len(list(tmp_path.glob("*.PEER.diag.pdf"))) == 1
    assert_matches_expected(peer_gz[0], repo_root / EXPECTED_PEER, mode="tolerant",
                            rtol=1e-6, atol=1e-8)


def test_marchenko_workflow_via_sos(run_sos, repo_root, tmp_path):
    p = run_sos(repo_root / NB, "Marchenko_PC", {
        "phenoFile": repo_root / PHENO, "covFile": repo_root / COV, "cwd": tmp_path,
        "N": 0, "numThreads": 1, "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(list(tmp_path.glob("*.Marchenko_PC.gz"))) == 1
