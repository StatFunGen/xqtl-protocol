"""Notebook tier: covariate_formatting.ipynb — merge genotype PCs into covariates.

Worker: code/script/data_preprocessing/covariate/covariate_formatting.R
Step ``merge_genotype_pc``: stacks the top-k genotype-PCA scores beneath the fixed
covariate matrix, intersecting covariates and PCA on their shared samples.

Fixtures:
  tests/fixtures/pca/protocol_example.unrelated.prune.pca.rds  — committed flashPCA
      model whose ``$pc_scores`` carries IID = SAMPLE_001.. (59 samples) + PC1..PCn
  tests/fixtures/covariate_formatting/covariates.base.tsv       — MWE fixed covariates
      (#id header; sex/age rows x 60 sample columns)
59 samples overlap, so the merge keeps that intersection.

NB: the merge is exercised with the notebook's default ``tol_cov=-1`` — the "do nothing"
sentinel that keeps every sample. This doubles as a regression guard for a fixed bug
where ``filter_mtx`` dropped *all* samples for a negative threshold (a 0 missing-rate is
> -1), leaving a covariate matrix with no samples.
"""
from __future__ import annotations

import gzip

R = "code/script/data_preprocessing/covariate/covariate_formatting.R"
NB = "pipeline/covariate_formatting.ipynb"
PCA = "tests/fixtures/pca/protocol_example.unrelated.prune.pca.rds"
COV = "tests/fixtures/covariate_formatting/covariates.base.tsv"
MSD = "code/script"


def _read_matrix(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    header = rows[0]
    ids = [r[0] for r in rows[1:]]
    return header, ids


def test_merge_genotype_pc(run_r, repo_root, tmp_path):
    out = tmp_path / "merged.gz"
    p = run_r(repo_root / R,
              ["--step", "merge_genotype_pc", "--cwd", tmp_path,
               "--pcaFile", repo_root / PCA, "--covFile", repo_root / COV,
               "--name", "merged", "--k", "10", "--tol-cov", "-1", "--mean-impute"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()
    header, ids = _read_matrix(out)
    assert header[0] == "#id"
    assert len(header) - 1 == 59                       # cov(60) ∩ pca(59) samples
    assert {"sex", "age"}.issubset(set(ids))           # fixed covariates on top
    assert [i for i in ids if i.startswith("PC")] == [f"PC{i}" for i in range(1, 11)]


def test_merge_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run covariate_formatting.ipynb merge_genotype_pc`.
    p = run_sos(repo_root / NB, "merge_genotype_pc", {
        "pcaFile": repo_root / PCA, "covFile": repo_root / COV,
        "cwd": tmp_path, "k": 10, "name": "merged",
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "merged.gz"
    assert out.exists()
    header, ids = _read_matrix(out)
    assert header[0] == "#id" and len(header) - 1 == 59
    assert {"sex", "age"}.issubset(set(ids))
