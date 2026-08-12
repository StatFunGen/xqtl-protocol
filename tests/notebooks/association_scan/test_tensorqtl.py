"""Notebook tier: TensorQTL.ipynb — cis-QTL scan via the tensorqtl package.

The Python worker (TensorQTL.py) does ONLY the tensorqtl scan and writes raw p-value
tables; the q-value / FDR / regional-significance math lives in TensorQTL.R and is
invoked as separate steps by the notebook cells (python scan, then Rscript). So the
`cis` workflow is: cis_1 = python scan + `Rscript TensorQTL.R nominal_qvalues`;
cis_2 = `Rscript TensorQTL.R regional_postprocess`. There is no python->R shell-out.

Fixtures (qtl_mini, 49-sample chr22 MWE): example.chr22.{bed,bim,fam} genotype,
example_geneexpr.bed.gz phenotype (16 genes), example_covariates.tsv covariates
(transposed by the worker). Direct-input mode (a .bed genotype + a .bed.gz phenotype).

The Python worker is driven with the interpreter running the suite (which carries the
tensorqtl package); the R step via run_r; the full pipeline via run_sos.
"""
from __future__ import annotations

import gzip
import subprocess
import sys

WORKER = "code/script/association_scan/TensorQTL/TensorQTL.py"
RSCRIPT = "code/script/association_scan/TensorQTL/TensorQTL.R"
NB = "pipeline/TensorQTL.ipynb"
FIX = "tests/fixtures/qtl_mini"
MSD = "code/script"

NOMINAL = "example_geneexpr_chr22.cis_qtl.pairs.tsv.gz"
REGIONAL = "example_geneexpr_chr22.cis_qtl.regional.tsv.gz"
PARQUET = "example_geneexpr.cis_qtl_pairs.22.parquet"


def _scan_args(fix, cwd):
    return ["--step", "cis", "--cwd", str(cwd),
            "--genotype-file", str(fix / "example.chr22.bed"),
            "--phenotype-file", str(fix / "example_geneexpr.bed.gz"),
            "--covariate-file", str(fix / "example_covariates.tsv"),
            "--chromosome", "22", "--window", "1000000",
            "--MAC", "0", "--maf-threshold", "0",
            "--permutation", "True", "--numThreads", "1"]


def _header(path):
    with gzip.open(path, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")


def _run_scan(repo_root, cwd):
    p = subprocess.run([sys.executable, str(repo_root / WORKER), *_scan_args(repo_root / FIX, cwd)],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr


def test_cis_scan_worker(repo_root, tmp_path):
    # The Python scan writes raw p-value tables only — no q-values, no R shell-out.
    _run_scan(repo_root, tmp_path)
    assert (tmp_path / PARQUET).exists()
    nominal = tmp_path / NOMINAL
    assert nominal.exists() and (tmp_path / (nominal.name + ".tbi")).exists()
    assert (tmp_path / REGIONAL).exists()
    header = _header(nominal)
    assert {"molecular_trait_id", "variant_id", "pvalue"}.issubset(header)
    assert "qvalue" not in header                       # q-values are NOT added by the scan


def test_nominal_qvalues_r(run_r, repo_root, tmp_path):
    # TensorQTL.R adds per-gene Storey q-values in place (reads the raw .gz, re-bgzip+tabix).
    _run_scan(repo_root, tmp_path)
    nominal = tmp_path / NOMINAL
    r = run_r(repo_root / RSCRIPT, ["nominal_qvalues", nominal])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / (nominal.name + ".tbi")).exists()
    assert "qvalue" in _header(nominal)                 # now present


def test_cis_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run TensorQTL.ipynb cis` = cis_1 (scan + Rscript nominal_qvalues)
    # -> cis_2 (Rscript regional_postprocess). No shim needed: python3 and Rscript are
    # sibling bash commands, never nested.
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "cis", {
        "genotype_file": fix / "example.chr22.bed",
        "phenotype_file": fix / "example_geneexpr.bed.gz",
        "covariate_file": fix / "example_covariates.tsv",
        "name": "protocol_example", "chromosome": 22, "MAC": 0,
        "cwd": tmp_path, "numThreads": 1,
        "modular_script_dir": repo_root / MSD}, timeout=1200)
    assert p.returncode == 0, p.stdout + p.stderr
    nominal = tmp_path / NOMINAL
    sig = tmp_path / "example_geneexpr_chr22.cis_qtl_regional_significance.tsv.gz"
    summ = tmp_path / "example_geneexpr_chr22.cis_qtl_regional_significance.summary.txt"
    assert (tmp_path / PARQUET).exists()
    assert nominal.exists() and sig.exists() and summ.exists() and summ.stat().st_size > 0
    assert "qvalue" in _header(nominal)                 # q-values added by the R step in cis_1
