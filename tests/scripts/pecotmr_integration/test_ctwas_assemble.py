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


def _assemble(run_r, repo_root, manifest, out, extra=()):
    """ctwas_assemble.R with the ctwas_chain arguments plus `extra`."""
    return run_r(repo_root / "code/script/pecotmr_integration/ctwas_assemble.R",
                 ["--manifest", manifest, "--twas-weights", repo_root / WEIGHTS,
                  "--method", "susie", "--twas-weight-cutoff", "0",
                  "--cs-min-cor", "0", "--min-pip-cutoff", "0", "--max-num-variants", "Inf",
                  *extra, "--output", out], timeout=300)


GROUP_CHECK = r"""
a <- commandArgs(TRUE); x <- readRDS(a[1]); g <- readRDS(a[2])
stopifnot(identical(g$z_gene$id, x$z_gene$id),
          isTRUE(all.equal(g$z_gene$z, x$z_gene$z)))
ctx <- substr(x$z_gene$context, 1, 1)
stopifnot(identical(g$z_gene$context, ctx),
          identical(g$z_gene$group, paste(ctx, x$z_gene$type, sep = "|")))
wctx <- function(w) unname(vapply(w, `[[`, "", "context"))
stopifnot(identical(wctx(g$weights), substr(wctx(x$weights), 1, 1)),
          isTRUE(all.equal(lapply(g$weights, `[[`, "wgt"), lapply(x$weights, `[[`, "wgt"))))
cat("groups", length(unique(x$z_gene$group)), "->", length(unique(g$z_gene$group)), "\n")
"""


def test_ctwas_assemble_group_context_pattern(ctwas_chain, run_r, repo_root, tmp_path):
    """--group-context-pattern removes the matched part of each context, so the
    gene's cTWAS group becomes <trimmed context>|<type>; gene IDs, z-scores and
    weights are unchanged. The pattern here keeps the first character."""
    out = tmp_path / "grouped.rds"
    p = _assemble(run_r, repo_root, ctwas_chain["manifest"], out,
                  ["--group-context-pattern", "(?<=^.).*$"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert "--group-context-pattern:" in p.stdout
    check = tmp_path / "check_groups.R"
    check.write_text(GROUP_CHECK)
    c = run_r(check, [ctwas_chain["inputs"], out])
    assert c.returncode == 0, c.stdout + c.stderr


def test_ctwas_assemble_group_context_pattern_empty_context(ctwas_chain, run_r, repo_root, tmp_path):
    """A pattern that removes a whole context stops instead of writing an empty group."""
    p = _assemble(run_r, repo_root, ctwas_chain["manifest"], tmp_path / "x.rds",
                  ["--group-context-pattern", ".*"])
    assert p.returncode != 0
    assert "leaves an empty context" in p.stdout + p.stderr
