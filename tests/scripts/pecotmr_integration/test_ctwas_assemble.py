"""Tier B: ctwas_assemble.R -> assembled cTWAS inputs (from the ctwas_chain)."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/ctwas/expected/ctwas_assemble.rds"
WEIGHTS = "tests/fixtures/twas/protocol_example.ctwas_weights.s4.chr22.rds"


def test_ctwas_assemble(ctwas_chain, read_rds, repo_root):
    assert read_rds(ctwas_chain["inputs"])["class"] == "list"

    # embeds absolute LD/snp-map reference paths -> normalize_paths (basename compare).
    assert_matches_expected(ctwas_chain["inputs"], repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8, normalize_paths=True)


def test_ctwas_assemble_list_file(ctwas_chain, run_r, repo_root, tmp_path):
    """--twas-weights-list, the notebook's form (a genome-scale list is too long for
    the command line), assembles the same inputs as --twas-weights; the arguments
    otherwise match the ctwas_chain call."""
    weights = tmp_path / "weights.txt"
    weights.write_text(f"{repo_root / WEIGHTS}\n")
    out = tmp_path / "ctwas_inputs.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/ctwas_assemble.R",
              ["--manifest", ctwas_chain["manifest"], "--twas-weights-list", weights,
               "--method", "susie", "--twas-weight-cutoff", "0",
               "--cs-min-cor", "0", "--min-pip-cutoff", "0", "--max-num-variants", "Inf",
               "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert_matches_expected(out, repo_root / EXPECTED,
                            mode="tolerant", rtol=1e-6, atol=1e-8, normalize_paths=True)
