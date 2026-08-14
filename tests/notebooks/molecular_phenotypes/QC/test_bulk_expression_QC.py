"""Notebook tier: bulk_expression_QC.ipynb — GTEx-style bulk RNA-seq QC.

Worker: code/script/molecular_phenotypes/QC/bulk_expression_QC.R
Steps (workflow ``qc`` chains them):
  qc_1 — low-expression gene filter on the TPM GCT
  qc_2 — outlier-sample removal (RLE IQR + distance-sum + robust Mahalanobis)
  qc_3 — subset the raw-count GCT to the QC-filtered genes/samples

Fixtures are reused from the bulk_expression_normalization tier: the 60-sample MWE
RNA-seq TPM/count GCTs (300 genes). Outputs are gzip'd GCTs (``#1.2`` + ``# nrow ncol``
comment header, then a ``gene_ID`` + samples matrix) — this asserts on that structure,
the monotonic gene/sample reductions along the chain, and the raw-count subset matching
the QC'd TPM.
"""
from __future__ import annotations

import gzip

from helpers.expected import assert_matches_expected

WORKER = "code/script/molecular_phenotypes/QC/bulk_expression_QC.R"
NB = "pipeline/bulk_expression_QC.ipynb"
FIX = "tests/fixtures/bulk_expression_normalization"
EXP = "tests/fixtures/bulk_expression_normalization/expected"
MSD = "code/script"


def _read_gct(path):
    """Return (colnames, n_data_rows) for a GCT.gz, skipping ``#`` comment lines."""
    with gzip.open(path, "rt") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    data = [ln for ln in lines if not ln.startswith("#")]
    header = data[0].split("\t")
    return header, len(data) - 1


def _qc_1(run_r, repo_root, cwd):
    fix = repo_root / FIX
    p = run_r(repo_root / WORKER,
              ["--step", "qc_1", "--cwd", cwd,
               "--tpm-gct", fix / "input.tpm.gct.gz", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    out = cwd / "input.low_expression_filtered.tpm.gct.gz"
    assert out.exists(), "qc_1 output missing"
    return out


def test_qc_1_low_expression_filter(run_r, repo_root, tmp_path):
    out = _qc_1(run_r, repo_root, tmp_path)
    header, n_genes = _read_gct(out)
    assert header[0] == "gene_ID"
    assert len(header) - 1 == 60                      # all 60 samples preserved
    assert 0 < n_genes <= 300                         # low-expression genes dropped
    # the GCT `# nrow ncol` comment line agrees with the actual matrix
    with gzip.open(out, "rt") as fh:
        dims = [ln for ln in fh if ln.startswith("#") and ln[1:2] == " "][0].split()
    assert int(dims[1]) == n_genes and int(dims[2]) == 60
    # regression: the low-expression filter is deterministic (rowMeans threshold, no RNG)
    # -> the decompressed GCT reproduces the committed snapshot byte-for-byte.
    assert_matches_expected(
        out, repo_root / EXP / "expected.qc_1.low_expression_filtered.tpm.gct.gz", mode="exact")


def test_qc_2_outlier_removal(run_r, repo_root, tmp_path):
    tpm1 = _qc_1(run_r, repo_root, tmp_path)
    _, g1 = _read_gct(tpm1)
    p = run_r(repo_root / WORKER,
              ["--step", "qc_2", "--cwd", tmp_path, "--tpm-gct", tpm1, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "input.low_expression_filtered.outlier_removed.tpm.gct.gz"
    assert out.exists(), "qc_2 output missing"
    header, g2 = _read_gct(out)
    assert header[0] == "gene_ID"
    n_samples = len(header) - 1
    assert 0 < n_samples <= 60                        # only samples are dropped here
    assert g2 == g1                                   # qc_2 leaves the gene set intact
    # regression: the outlier-removal step is deterministic (RLE IQR + hclust +
    # prcomp + robust Mahalanobis, no RNG) -> the GCT reproduces the snapshot.
    assert_matches_expected(
        out, repo_root / EXP / "expected.qc_2.outlier_removed.tpm.gct.gz", mode="exact")


def test_qc_3_subset_counts(run_r, repo_root, tmp_path):
    tpm1 = _qc_1(run_r, repo_root, tmp_path)
    p = run_r(repo_root / WORKER,
              ["--step", "qc_2", "--cwd", tmp_path, "--tpm-gct", tpm1, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    tpm2 = tmp_path / "input.low_expression_filtered.outlier_removed.tpm.gct.gz"
    tpm_header, tpm_genes = _read_gct(tpm2)

    fix = repo_root / FIX
    p = run_r(repo_root / WORKER,
              ["--step", "qc_3", "--cwd", tmp_path, "--tpm-gct", tpm2,
               "--counts-gct", fix / "input.geneCount.gct.gz", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "input.low_expression_filtered.outlier_removed.geneCount.gct.gz"
    assert out.exists(), "qc_3 output missing"
    cnt_header, cnt_genes = _read_gct(out)
    # raw counts subset to exactly the QC'd TPM genes and samples
    assert cnt_genes == tpm_genes
    assert cnt_header == tpm_header
    # regression: the count subset (filter genes to QC'd TPM, select QC'd samples)
    # is deterministic -> the GCT reproduces the snapshot byte-for-byte.
    assert_matches_expected(
        out, repo_root / EXP / "expected.qc_3.outlier_removed.geneCount.gct.gz", mode="exact")


def test_qc_workflow_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run bulk_expression_QC.ipynb qc` chains qc_1 -> qc_2 -> qc_3.
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "qc", {
        "tpm_gct": fix / "input.tpm.gct.gz",
        "counts_gct": fix / "input.geneCount.gct.gz",
        "cwd": tmp_path, "numThreads": 1,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "input.low_expression_filtered.tpm.gct.gz").exists()
    assert (tmp_path / "input.low_expression_filtered.outlier_removed.tpm.gct.gz").exists()
    assert (tmp_path / "input.low_expression_filtered.outlier_removed.geneCount.gct.gz").exists()
