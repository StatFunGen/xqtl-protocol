"""Notebook tier: covariate_hidden_factor.ipynb — hidden factor analysis.

Workers:
  code/script/data_preprocessing/covariate/covariate_hidden_factor.R   (residual / Marchenko_PC / PEER_extract / BiCV)
  code/script/data_preprocessing/covariate/covariate_hidden_factor_peer.py  (mofapy2 PEER fit + extract to TSV)

The PEER path fits mofapy2 in a standalone Python script (no reticulate / no MOFA2
Bioconductor package) and writes factor/weight/variance TSV sidecars; the R
PEER_extract step reads those to build the .PEER.gz covariate matrix and a ggplot
diagnostic PDF. Fixtures: the committed 200-gene rnaseq phenotype BED and a
#id-headed covariate matrix derived from qtl_mini (49 shared samples).
"""
from __future__ import annotations

import gzip
import subprocess

import pytest

R = "code/script/data_preprocessing/covariate/covariate_hidden_factor.R"
PY = "code/script/data_preprocessing/covariate/covariate_hidden_factor_peer.py"
NB = "pipeline/covariate_hidden_factor.ipynb"
PHENO = "tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz"
COV = "tests/fixtures/covariate_hidden_factor/covariates.tsv"
MSD = "code/script"


def _col1(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [ln.split("\t", 1)[0] for ln in fh if ln.strip()]


def _has_r_pkg(repo_root, pkg):
    p = subprocess.run(
        ["pixi", "run", "--frozen", "Rscript", "-e",
         f'quit(status = !requireNamespace("{pkg}", quietly=TRUE))'],
        cwd=repo_root, capture_output=True, text=True)
    return p.returncode == 0


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


# each choose-k method needs pca() (PCAtools/BiocSingular) plus one method-specific package
_CHOOSE_K_DEPS = {
    "Marchenko": ("PCAtools", "BiocSingular", "RMTstat"),      # chooseMarchenkoPastur -> RMTstat::qmp
    "Buja_Eyuboglu": ("PCAtools", "BiocSingular", "jackstraw"),  # jackstraw::permutationPA
}


@pytest.mark.parametrize("method", ["Marchenko", "Buja_Eyuboglu"])
def test_marchenko_pc(run_r, repo_root, tmp_path, method):
    if not all(_has_r_pkg(repo_root, p) for p in _CHOOSE_K_DEPS[method]):
        pytest.skip(f"{'/'.join(_CHOOSE_K_DEPS[method])} not installed")
    resid = _residual(run_r, repo_root, tmp_path)
    out = tmp_path / f"out.{method}_PC.gz"
    p = run_r(repo_root / R, ["--step", "Marchenko_PC", "--cwd", tmp_path, "--residFile", resid,
                              "--covFile", repo_root / COV, "--choose-k-method", method,
                              "--output", out, "--N", "0"])
    assert p.returncode == 0, p.stdout + p.stderr
    rows = _col1(out)
    assert rows[0] == "#id"
    # known covariates are stacked above the inferred hidden factors
    assert {"sex", "age", "PC1"}.issubset(set(rows))
    assert any(r.startswith("Hidden_Factor_PC") for r in rows)


def test_peer_fit_extract(run_r, repo_root, tmp_path):
    resid = _residual(run_r, repo_root, tmp_path)
    cwd = tmp_path / "peer"
    # PEER_fit: mofapy2 python worker -> model .hd5 + TSV sidecars
    p = run_r(repo_root / R, ["--step", "PEER_fit", "--cwd", cwd, "--residFile", resid,
                              "--N", "5", "--iteration", "100", "--convergence-mode", "fast",
                              "--tol", "0.001", "--r2-tol", "False", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    stem = "protocol_example.rnaseq.bed.covariates.residual"
    model = cwd / f"{stem}.PEER_MODEL.hd5"
    assert model.exists()                                            # HDF5 model still saved
    for ext in ("PEER.factors.tsv", "PEER.weights.tsv", "PEER.variance.tsv"):
        assert (cwd / f"{stem}.{ext}").exists()
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
    assert diag.stat().st_size > 0
    rows = _col1(peer_gz)
    assert rows[0] == "#id"
    assert {"sex", "age", "PC1"}.issubset(set(rows))                 # known covariates
    assert [r for r in rows if r.startswith("Factor")] == [f"Factor{i}" for i in range(1, 6)]


def test_bicv(run_r, repo_root, tmp_path):
    resid = _residual(run_r, repo_root, tmp_path)
    cwd = tmp_path / "bicv"
    # BiCV_2: fake single-site VCF from the residual BED
    p = run_r(repo_root / R, ["--step", "BiCV_2", "--cwd", cwd, "--residFile", resid, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    vcf = cwd / "protocol_example.rnaseq.bed.covariates.residual.bed.fake.vcf.gz"
    assert vcf.exists() and (cwd / (vcf.name + ".tbi")).exists()
    with gzip.open(vcf, "rt") as fh:
        assert fh.readline().startswith("##fileformat=VCFv4.2")

    # BiCV_3: apex factor -> .BiCV.gz (known covariates + apex factors)
    out = cwd / "out.residual.BiCV.gz"
    p = run_r(repo_root / R, ["--step", "BiCV_3", "--cwd", cwd, "--residFile", resid,
                              "--vcfFile", vcf, "--covFile", repo_root / COV,
                              "--N", "3", "--iteration", "5", "--output", out, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()
    rows = _col1(out)
    assert rows[0] == "#id"
    assert {"sex", "age", "PC1"}.issubset(set(rows))
    assert any(r.startswith("factor_") for r in rows)                # apex-inferred factors


def test_peer_workflow_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run covariate_hidden_factor.ipynb PEER` chains PEER_1
    # (compute_residual) -> PEER_2 (mofapy2 fit) -> PEER_3 (extract) end to end.
    p = run_sos(repo_root / NB, "PEER", {
        "phenoFile": repo_root / PHENO, "covFile": repo_root / COV, "cwd": tmp_path,
        "N": 5, "iteration": 100, "numThreads": 1,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(list(tmp_path.glob("*.PEER.gz"))) == 1
    assert len(list(tmp_path.glob("*.PEER.diag.pdf"))) == 1


def test_bicv_workflow_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run covariate_hidden_factor.ipynb BiCV` chains BiCV_1
    # (compute_residual) -> BiCV_2 (fake VCF) -> BiCV_3 (apex factor).
    p = run_sos(repo_root / NB, "BiCV", {
        "phenoFile": repo_root / PHENO, "covFile": repo_root / COV, "cwd": tmp_path,
        "N": 3, "iteration": 5, "numThreads": 1,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(list(tmp_path.glob("*.BiCV.gz"))) == 1


def test_marchenko_workflow_via_sos(run_sos, repo_root, tmp_path):
    if not all(_has_r_pkg(repo_root, p) for p in ("PCAtools", "BiocSingular", "RMTstat")):
        pytest.skip("PCAtools/BiocSingular/RMTstat not installed")
    p = run_sos(repo_root / NB, "Marchenko_PC", {
        "phenoFile": repo_root / PHENO, "covFile": repo_root / COV, "cwd": tmp_path,
        "N": 0, "numThreads": 1, "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(list(tmp_path.glob("*.Marchenko_PC.gz"))) == 1
