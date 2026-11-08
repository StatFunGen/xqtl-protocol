"""Tier B: twas.R -> TWAS-Z GRanges from a committed S4 TwasWeights + a
GwasSumStats built here (the legacy twas-weights conversion has been retired)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

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
