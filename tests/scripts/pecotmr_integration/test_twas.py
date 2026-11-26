"""Tier B: twas.R -> TWAS-Z GRanges from a committed S4 TwasWeights + a
GwasSumStats built here (the legacy twas-weights conversion has been retired)."""
from __future__ import annotations

import subprocess

import pytest

from helpers.expected import assert_matches_expected
from helpers.r_runner import rscript_bin

FX = "tests/fixtures"
TW = FX + "/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.twas_weights.s4.rds"
EXPECTED = FX + "/twas/expected/twas.chr22.rds"


@pytest.mark.parametrize("via_table", [False, True], ids=["inline", "gene_table"])
def test_twas(via_table, run_r, read_rds, repo_root, tmp_path):
    r = repo_root
    gss = tmp_path / "gss.rds"
    assert run_r(r / "code/script/pecotmr_integration/gwas_sumstats_construct.R",
        ["--study", "protocol_example_twas_chr22",
         "--gwas-tsv", r / f"{FX}/twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
         "--ld-block", "chr22:10000000-19000000",
         "--ld-meta", r / f"{FX}/ld_reference/ld_meta_file.tsv", "--output", gss], timeout=300).returncode == 0
    out = tmp_path / "twas.rds"
    if via_table:
        # the notebook's form: a chromosome-wide region's per-gene lists are too
        # long for the command line. An empty fine_mapping_result means no MR.
        table = tmp_path / "genes.tsv"
        table.write_text(f"twas_weights\tfine_mapping_result\toutput\n{r / TW}\t\t{out}\n")
        genes = ["--gene-table", table]
    else:
        genes = ["--twas-weights", r / TW, "--output", out]
    p = run_r(r / "code/script/pecotmr_integration/twas.R",
              [*genes, "--gwas-sumstats", gss], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert read_rds(out)["class"] == "GRanges"
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)


def test_twas_empty_gwas_window(run_r, repo_root, tmp_path):
    """A window with no GWAS variants: gwas_sumstats_construct.R writes an empty
    GwasSumStats, and the gene gets an empty result with the columns of a scored
    one, and a message saying why."""
    r = repo_root
    gss = tmp_path / "gss.rds"
    # chr22:10.8-11.2 Mb lies inside the LD panel, between two GWAS variants.
    assert run_r(r / "code/script/pecotmr_integration/gwas_sumstats_construct.R",
        ["--study", "protocol_example_twas_chr22",
         "--gwas-tsv", r / f"{FX}/twas/protocol_example.twas.gwas_sumstats.chr22.tsv.gz",
         "--ld-block", "chr22:10800000-11200000",
         "--ld-meta", r / f"{FX}/ld_reference/ld_meta_file.tsv", "--output", gss], timeout=300).returncode == 0
    out = tmp_path / "twas.rds"
    p = run_r(r / "code/script/pecotmr_integration/twas.R",
              ["--twas-weights", r / TW, "--gwas-sumstats", gss, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "no (qtl, gwas) tuple produced a result" in p.stdout + p.stderr
    check = subprocess.run(
        [rscript_bin(), "-e",
         "suppressPackageStartupMessages(library(GenomicRanges)); "
         f"x <- readRDS('{out}'); scored <- readRDS('{r / EXPECTED}'); "
         "stopifnot(is(x, 'GRanges'), length(x) == 0L, "
         "identical(lapply(mcols(x), class), lapply(mcols(scored), class)))"],
        capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_twas_missing_gwas_sumstats(run_r, repo_root, tmp_path):
    """gwas_sumstats_construct.R always writes its GwasSumStats, so a missing file
    is a wrong path, and an error."""
    r = repo_root
    p = run_r(r / "code/script/pecotmr_integration/twas.R",
              ["--twas-weights", r / TW, "--output", tmp_path / "twas.rds",
               "--gwas-sumstats", tmp_path / "none.gwas_sumstats.rds"], timeout=300)
    assert p.returncode != 0
    assert "cannot open" in p.stdout + p.stderr
