"""Notebook tier: drive mnm_regression.ipynb through `sos run` on qtl_mini.

This is the website-published entrypoint for the QTL fine-mapping / TWAS-weight
chain. Running the `qtl_dataset_construct+susie_twas` step exercises the SoS cell
orchestration end-to-end (param wiring, path resolution, step chaining) that the
direct script tests can't reach, and asserts the same output S4 shapes.
"""
from __future__ import annotations

import pytest

GENE = "ENSG00000283047"


def test_susie_twas(run_sos, read_rds, repo_root, qtl_mini, tmp_path):
    cwd = tmp_path / "mnm"
    p = run_sos(
        repo_root / "pipeline/mnm_regression.ipynb",
        "qtl_dataset_construct+susie_twas",
        {
            "name": "test_study",
            "cwd": cwd,
            "genoFile": qtl_mini / "example.chr22.bed",
            "phenoFile": qtl_mini / "pheno_manifest_multicontext.tsv",
            "covFile": qtl_mini / "example_covariates.tsv",
            "customized-association-windows": qtl_mini / "association_windows.bed",
            "region-name": GENE,
            "transpose-covariates": True,          # QTLtools-format covariates
            "modular_script_dir": repo_root / "code/script",
        },
        cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr

    fmr = cwd / f"fine_mapping/test_study.{GENE}.univariate_bvsr.rds"
    tw = cwd / f"twas_weights/test_study.{GENE}.univariate_twas_weights.rds"
    assert fmr.exists(), f"missing fine-mapping output:\n{p.stdout}"
    assert tw.exists(), f"missing twas-weights output:\n{p.stdout}"
    assert read_rds(fmr)["class"] == "QtlFineMappingResult"
    assert read_rds(tw)["class"] == "TwasWeights"
