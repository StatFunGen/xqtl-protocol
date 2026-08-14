"""Notebook tier: phenotype_imputation.ipynb — molecular-phenotype NA imputation.

Worker: code/script/data_preprocessing/phenotype/phenotype_imputation.R (8 methods).
Each method fills the NAs of a molecular-phenotype BED and writes a bgzip+tabix'd
``<stem><suffix>.imputed.bed.gz``.

Fixture: a 250-feature x 60-sample subset of the MWE proteomics matrix
(protocol_example.protein.missing.bed.gz, ~10% NA). Every method must leave no NA in
the value columns. Backing packages: flashier/ebnm (EBMF/gEBMF), missForest, impute
(knn), softImpute (soft / bed_filter_na).
"""
from __future__ import annotations

import gzip

import pytest

from helpers.expected import assert_matches_expected

R = "code/script/data_preprocessing/phenotype/phenotype_imputation.R"
NB = "pipeline/phenotype_imputation.ipynb"
FIX = "tests/fixtures/phenotype_imputation/protocol_example.protein.missing.bed.gz"
MSD = "code/script"
STEM = "protocol_example.protein.missing"   # get_outpath strips .bed.gz from the phenoFile
EXPECTED = "tests/fixtures/phenotype_imputation/expected"

# Every method is reproducible under a fixed RNG seed, so all eight are value-compared
# against a committed snapshot of the .imputed.bed.gz (the comparator decompresses it):
# mean/lod are closed form; EBMF/gEBMF (flashier self-seeds 666) and knn (impute.knn
# self-seeds) are deterministic regardless of --seed; missForest and softImpute (the
# `soft` step, and the soft path of `bed_filter_na`) draw fresh randomness that
# phenotype_imputation.R's --seed now governs. All runs below pass --seed 1.

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

# missForest (random-forest imputation) is NOT cross-platform reproducible: the RF split
# points hinge on floating-point comparisons that diverge across macOS/Linux BLAS, giving
# materially different (even sign-flipped) imputed values. So it is checked structure-only
# (existence + no-NA via _assert_imputed), not value-compared. (Method is a candidate for
# removal — pending collaborator decision.)
NON_REPRODUCIBLE = {"missforest"}


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
               "--numThreads", "1", "--seed", "1", *extra])
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / f"{STEM}{suffix}"
    assert out.exists() and (tmp_path / (out.name + ".tbi")).exists()
    assert 0 < _assert_imputed(out) <= 250
    # regression: value-compare the imputed matrix within tolerance (decompressed cell-wise;
    # header/IDs exact), except for methods that are not cross-platform reproducible.
    if method not in NON_REPRODUCIBLE:
        assert_matches_expected(out, repo_root / EXPECTED / out.name,
                                mode="tolerant", rtol=1e-6, atol=1e-8)


def test_impute_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run phenotype_imputation.ipynb soft`
    p = run_sos(repo_root / NB, "soft", {
        "phenoFile": repo_root / FIX, "cwd": tmp_path, "numThreads": 1,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    outs = list(tmp_path.glob("*.imputed.bed.gz"))
    assert len(outs) == 1
    _assert_imputed(outs[0])
