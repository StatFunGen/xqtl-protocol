"""Notebook tier: drive mnm_regression.ipynb through `sos run` on qtl_mini.

This is the website-published entrypoint for the QTL fine-mapping / TWAS-weight
chain. Running the `qtl_dataset_construct+susie_twas` step exercises the SoS cell
orchestration end-to-end (param wiring, path resolution, step chaining) that the
direct script tests can't reach, and asserts the same output S4 shapes.
"""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

GENE = "ENSG00000283047"


def test_susie_twas(run_sos, read_rds, repo_root, qtl_mini, tmp_path):
    cwd = tmp_path / "mnm"
    p = run_sos(
        repo_root / "pipeline/mnm_regression.ipynb",
        "qtl_dataset_construct+susie_twas",
        {
            "name": "test_study",
            "cwd": cwd,
            "genoFile": qtl_mini / "protocol_example.genotype.chr22.bed",
            "phenoFile": qtl_mini / "protocol_example.pheno_manifest_context.tsv",
            "covFile": qtl_mini / "example_covariates.tsv",
            "customized-association-windows": qtl_mini / "association_windows.bed",
            "region-name": GENE,
            "transpose-covariates": True,          # QTLtools-format covariates
            "seed": 1,                              # reproducible susie/twas fit
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

    # regression: the FMR (univariate SuSiE fine-mapping) is cross-platform-stable and is
    # value-compared. The TwasWeights (univariate_twas_weights.rds) is NON-REPRODUCIBLE
    # cross-platform — value-compare DISABLED pending a collaborator decision. Its
    # cross-validation susie/mr.mash refits diverge across macOS vs Linux BLAS (3-19% on CI;
    # the underlying fits are under-converged at default). We confirmed the convergence IS
    # tunable from the wrapper (twas_method_args -> --method-args), but whether tightening
    # closes the gap is unresolved. Checked for existence + class (above) only.
    # See memory: cross-platform-numeric-divergence.
    exp = repo_root / "tests/fixtures/mnm_regression/expected"
    assert_matches_expected(fmr, exp / "univariate_bvsr.rds", mode="tolerant",
                            rtol=1e-6, atol=1e-8)
