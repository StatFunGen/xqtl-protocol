"""Notebook tier: phenotype_imputation.ipynb — molecular-phenotype NA imputation.

Worker: code/script/data_preprocessing/phenotype/phenotype_imputation.R (9 methods),
plus phenotype_imputation.sh (a thin bash dispatcher; the notebook uses it for the
missxgboost step). Each method fills the NAs of a molecular-phenotype BED and writes a
bgzip+tabix'd ``<stem><suffix>.imputed.bed.gz``.

Fixture: a 250-feature x 60-sample subset of the MWE proteomics matrix
(protocol_example.protein.missing.bed.gz, ~10% NA). Every method must leave no NA in
the value columns. Backing packages: flashier/ebnm (EBMF/gEBMF), missForest, impute
(knn), softImpute (soft / bed_filter_na), xgboost via xgb_imp.R (missxgboost).
"""
from __future__ import annotations

import gzip

import pytest

R = "code/script/data_preprocessing/phenotype/phenotype_imputation.R"
SH = "code/script/data_preprocessing/phenotype/phenotype_imputation.sh"
NB = "pipeline/phenotype_imputation.ipynb"
FIX = "tests/fixtures/phenotype_imputation/protocol_example.protein.missing.bed.gz"
MSD = "code/script"
STEM = "protocol_example.protein.missing"   # get_outpath strips .bed.gz from the phenoFile

# step -> (get_outpath suffix when --output is omitted, extra CLI args)
METHODS = {
    "EBMF":          (".EBMF.imputed.bed.gz",      ["--num-factor", "5"]),
    "gEBMF":         (".gEBMF.imputed.bed.gz",     ["--num-factor", "5"]),
    "missforest":    (".missForest.imputed.bed.gz", []),
    "knn":           (".knn.imputed.bed.gz",       []),
    "soft":          (".soft.imputed.bed.gz",      []),
    "mean":          (".mean.imputed.bed.gz",      []),
    "lod":           (".lod.imputed.bed.gz",       []),
    "bed_filter_na": (".filtered.imputed.bed.gz",  ["--tol-missing", "0.5"]),
}


def _read(path):
    with gzip.open(path, "rt") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    return rows[0], rows[1:]


def _assert_imputed(path):
    header, body = _read(path)
    assert header[:4] == ["#chr", "start", "end", "ID"]
    assert len(header) - 4 == 60                         # all samples preserved
    assert body, "no features written"
    for r in body:
        assert "NA" not in r[4:], "NA remains after imputation"
    return len(body)


@pytest.mark.parametrize("method", list(METHODS))
def test_impute_method(run_r, repo_root, tmp_path, method):
    suffix, extra = METHODS[method]
    p = run_r(repo_root / R,
              ["--step", method, "--cwd", tmp_path, "--phenoFile", repo_root / FIX,
               "--numThreads", "1", *extra])
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / f"{STEM}{suffix}"
    assert out.exists() and (tmp_path / (out.name + ".tbi")).exists()
    assert 0 < _assert_imputed(out) <= 250


def test_missxgboost_via_sh(run_sh, repo_root, tmp_path):
    # the notebook drives missxgboost through the .sh dispatcher -> phenotype_imputation.R
    out = tmp_path / "xgb.imputed.bed.gz"
    p = run_sh(repo_root / SH,
               ["missxgboost", "--cwd", tmp_path, "--phenoFile", repo_root / FIX,
                "--output", out, "--qc-prior-to-impute", "TRUE", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and (tmp_path / (out.name + ".tbi")).exists()
    _assert_imputed(out)


def test_impute_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run phenotype_imputation.ipynb soft`
    p = run_sos(repo_root / NB, "soft", {
        "phenoFile": repo_root / FIX, "cwd": tmp_path, "numThreads": 1,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    outs = list(tmp_path.glob("*.imputed.bed.gz"))
    assert len(outs) == 1
    _assert_imputed(outs[0])
