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


def test_mnm(run_sos, read_rds, repo_root, qtl_mini, tmp_path):
    """`mnm` fits every context of one gene jointly with mvSuSiE over the QtlDataset that
    qtl_dataset_construct writes, so the two steps run together. Value-compared: the fit is
    byte-identical across seeded reruns on one machine."""
    cwd = tmp_path / "mnm"
    p = run_sos(
        repo_root / "pipeline/mnm_regression.ipynb",
        "qtl_dataset_construct+mnm",
        {
            "name": "test_study",
            "cwd": cwd,
            "genoFile": qtl_mini / "protocol_example.genotype.chr22.bed",
            "phenoFile": qtl_mini / "protocol_example.pheno_manifest_context.tsv",
            "covFile": qtl_mini / "example_covariates.tsv",
            "customized-association-windows": qtl_mini / "association_windows.bed",
            "region-name": GENE,
            "transpose-covariates": True,          # QTLtools-format covariates
            "seed": 1,                              # reproducible mvsusie fit
            "modular_script_dir": repo_root / "code/script",
        },
        cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr

    fmr = cwd / f"multivariate_fine_mapping/test_study.{GENE}.multicontext_bvsr.rds"
    assert fmr.exists(), f"missing multi-context output:\n{p.stdout}"
    assert read_rds(fmr)["class"] == "QtlFineMappingResult"
    exp = repo_root / "tests/fixtures/mnm_regression/expected"
    assert_matches_expected(fmr, exp / "multicontext_bvsr.rds", mode="tolerant",
                            rtol=1e-6, atol=1e-8)


def test_fsusie_ti_export(run_sos, read_rds, repo_root, qtl_mini, tmp_path):
    """The native fSuSiE workflow writes one indexed table with curves and bands."""
    import csv
    import gzip
    import json
    import subprocess

    cwd = tmp_path / "fsusie"
    windows = tmp_path / "functional_region.bed"
    windows.write_text("#chr\tstart\tend\tID\nchr22\t10000000\t18000000\tfunctional_region\n")
    p = run_sos(
        repo_root / "pipeline/mnm_regression.ipynb",
        "qtl_dataset_construct+fsusie",
        {
            "name": "test_study", "cwd": cwd,
            "genoFile": qtl_mini / "protocol_example.genotype.chr22.bed",
            "phenoFile": qtl_mini / "protocol_example.pheno_manifest_context.tsv",
            "covFile": qtl_mini / "example_covariates.tsv",
            "customized-association-windows": windows,
            "transpose-covariates": True, "seed": 1,
            "susie-top-pc": 1, "mem": "40G",
            "fsusie-method-args": json.dumps({"fsusie": {
                "post_processing": "TI", "max_scale": 4, "L": 2,
                "max_SNP_EM": 20, "verbose": False}}),
            "modular_script_dir": repo_root / "code/script",
        }, cwd=repo_root, timeout=1800)
    assert p.returncode == 0, p.stdout + p.stderr
    fits = list((cwd / "fsusie").glob("*.fsusie.rds"))
    assert len(fits) == 1
    info = read_rds(fits[0])
    assert info["class"] == "QtlFineMappingResult"
    assert info["nrow"] == 4  # one joint fit + one PC per context
    table = cwd / "fsusie/test_study.exported.bed.gz"
    assert table.with_suffix(table.suffix + ".tbi").exists()
    with gzip.open(table, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert len(reader.fieldnames) == 20
        rows = list(reader)
    assert rows, "Fixture must exercise a nonempty credible-set export"
    for row in rows:
        n = int(row["grid_resolution"])
        assert len(row["grid_effects"].split(";")) == n
        widths = [float(v) for v in row["grid_band_halfwidth"].split(";")]
        assert len(widths) == n and min(widths) >= 0 and max(widths) > 0
        assert len(row["epi_mark_names"].split(";")) == len(row["epi_mark_effects"].split(";"))
    p = subprocess.run(["tabix", str(table), "22:10000000-18000000"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip()
