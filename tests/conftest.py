"""Shared pytest fixtures for the xqtl-protocol wrapper tests.

pytest is the single orchestrator: it runs the R/py CLI wrappers as subprocesses
against the committed fixtures and asserts on their outputs. These fixtures give
tests a `run_r` subprocess runner (which also records which scripts were
exercised, for the session-end inventory report) and a `read_rds` structural
probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent          # tests/ -> repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))    # tests/ on sys.path

from helpers import r_runner                                 # noqa: E402
from helpers import sos_runner                                # noqa: E402
from helpers.inventory import report_untested                # noqa: E402

# Basenames of every wrapper script a test drove this session (for inventory).
_TOUCHED: set[str] = set()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def qtl_mini() -> Path:
    """The committed chr22 mini fixture bundle."""
    return REPO_ROOT / "tests" / "fixtures" / "qtl_mini"


@pytest.fixture(scope="session")
def run_r():
    """Run an R wrapper (records the script for the inventory report)."""
    def _run(script, args=(), timeout: int = 600):
        _TOUCHED.add(Path(script).name)
        return r_runner.run_r(script, args, timeout=timeout)
    return _run


@pytest.fixture(scope="session")
def read_rds():
    """Structural summary (class / nrow / accessors) of an RDS output."""
    return r_runner.read_rds


@pytest.fixture(scope="session")
def run_sos():
    """Run a SoS notebook step (the production entrypoint) as a subprocess."""
    return sos_runner.run_sos


def pytest_sessionfinish(session, exitstatus):
    """Session-end: report which pecotmr_integration wrappers went untested.

    This is a SCRIPT-tier metric (wrappers driven via run_r). Notebook-tier tests
    drive their scripts through `sos run` (bash), not run_r, so the inventory is
    meaningless for a notebook-only run (it would read ~0/N and mislead). Only
    emit it when the session actually collected script-tier tests.
    """
    ran_scripts = any(it.nodeid.startswith("scripts/")
                      for it in getattr(session, "items", []))
    if not ran_scripts:
        return
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    report_untested(REPO_ROOT, _TOUCHED, tr)

@pytest.fixture(scope="session")
def qtl_dataset(run_r, repo_root, qtl_mini, tmp_path_factory):
    """Build one QtlDataset from the qtl_mini fixtures (once per session) and
    return its RDS path. The covariates are QTLtools-format (covariate rows,
    sample columns), so --transpose-covariates is required."""
    out = tmp_path_factory.mktemp("qtl") / "qtl_dataset.rds"
    script = repo_root / "code/script/pecotmr_integration/qtl_dataset_construct.R"
    p = run_r(script, [
        "--study", "test_study",
        "--genotype-prefix", qtl_mini / "example.chr22",
        "--phenotype-manifest", qtl_mini / "pheno_manifest_multicontext.tsv",
        "--transpose-covariates",
        "--output", out,
    ], timeout=600)
    assert p.returncode == 0, f"qtl_dataset_construct failed:\n{p.stdout}\n{p.stderr}"
    return out


@pytest.fixture(scope="session")
def fmr(run_r, repo_root, qtl_dataset, tmp_path_factory):
    """A QtlFineMappingResult for the fixture gene (built once); the input for
    the plot / upset / export wrappers."""
    out = tmp_path_factory.mktemp("fmr") / "fmr.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping.R",
              ["--qtl-dataset", qtl_dataset,
               "--gene-id", "ENSG00000283047", "--output", out], timeout=600)
    assert p.returncode == 0, f"fine_mapping failed:\n{p.stdout}\n{p.stderr}"
    return out


@pytest.fixture(scope="session")
def mash_model_chain(run_r, repo_root, tmp_path_factory):
    """Build the MASH fit chain once from the committed MWE-derived mash input:
    Vhat (mash_vhat.R) -> prior U (mash_prior.R) -> fitted model (mash_fit.R).
    Returns the intermediate RDS paths; the inputs for the mash_fit /
    mash_posterior per-script tests."""
    S = repo_root / "code/script/pecotmr_integration"
    data = repo_root / "tests/fixtures/mash/mashr_input.rds"
    d = tmp_path_factory.mktemp("mash_fit")

    def ok(script, args, t=400):
        p = run_r(S / script, args, timeout=t)
        assert p.returncode == 0, f"{script} failed:\n{p.stdout}\n{p.stderr}"

    vhat = d / "vhat.rds"
    ok("mash_vhat.R", ["--data", data, "--method", "simple",
                       "--effect-model", "EE", "--output", vhat])
    prior = d / "prior.rds"
    ok("mash_prior.R", ["--data", data, "--engine", "cov_ed",
                        "--components", "canonical,pca", "--npc", "3",
                        "--vhat-data", vhat, "--effect-model", "EE",
                        "--output", prior])
    model = d / "model.rds"
    ok("mash_fit.R", ["--data", data, "--vhat-data", vhat, "--prior-data", prior,
                      "--effect-model", "EE", "--output", model])
    return {"data": data, "vhat": vhat, "prior": prior, "model": model}


@pytest.fixture(scope="session")
def ctwas_chain(run_r, repo_root, tmp_path_factory):
    """Run the cTWAS chain once over the blessed 20-block fixture and return the
    three intermediate RDS ({inputs, est, finemap}); the inputs for the
    ctwas_assemble / ctwas_est / ctwas_finemap per-script tests."""
    import csv
    r = repo_root
    S = r / "code/script/pecotmr_integration"
    fx = r / "tests/fixtures"
    ldmeta = fx / "ld_reference/ld_meta_file.ctwas.tsv"
    d = tmp_path_factory.mktemp("ctwas")

    def ok(script, args, t=300):
        p = run_r(S / script, args, timeout=t)
        assert p.returncode == 0, f"{script} failed:\n{p.stdout}\n{p.stderr}"

    # S4 TwasWeights (region-provenance stamped) — the committed modern fixture
    # (the legacy_ctwas_weights_to_s4 conversion step has been retired).
    s4 = fx / "twas/protocol_example.ctwas_weights.s4.chr22.rds"
    manifest = d / "manifest.tsv"
    ok("ctwas_manifest.R",
       ["--ld-meta", ldmeta, "--chrom", "chr22", "--gwas-sumstats-dir", d, "--output", manifest],
       t=120)
    for row in csv.DictReader(open(manifest), delimiter="\t"):
        ok("gwas_sumstats_construct.R",
           ["--study", "protocol_example_twas_chr22",
            "--gwas-tsv", fx / "twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
            "--ld-block", row["region"], "--ld-meta", ldmeta,
            "--output", row["gwas_sumstats_rds"]])
    inputs = d / "ctwas_inputs.rds"
    ok("ctwas_assemble.R",
       ["--manifest", manifest, "--twas-weights", s4, "--twas-weight-cutoff", "0",
        "--cs-min-cor", "0", "--min-pip-cutoff", "0", "--max-num-variants", "Inf",
        "--output", inputs])
    est = d / "ctwas_est.rds"
    ok("ctwas_est.R",
       ["--inputs", inputs, "--thin", "1", "--niter", "50",
        "--group-prior-var-structure", "shared_all", "--min-group-size", "1",
        "--fallback-to-prefit", "--ncore", "1", "--output", est])
    finemap = d / "ctwas_finemap.rds"
    ok("ctwas_finemap.R",
       ["--est", est, "--L", "5", "--min-nonsnp-pip", "0.5", "--keep-snps",
        "--ncore", "1", "--output", finemap])
    return {"inputs": inputs, "est": est, "finemap": finemap}
