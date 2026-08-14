"""Tier B: mash_feature_score.R (meta / nsig / pval_pair / finemap) +
mash_feature_score_merge.R, over an MWE-derived posterior contrast."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

FX = "tests/fixtures/mash_posterior"
CELLS = "ALL,Ast,End,Exc,Inh,Mic,OPC,Oli"
EXPECTED_MERGE = "tests/fixtures/mash/expected/feature_score_merge.nsig.csv"


def _contrast(run_r, repo_root, tmp_path):
    out = tmp_path / "contrast.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_posterior_contrast.R",
              ["--posterior", repo_root / FX / "posterior.rds",
               "--orig-data", repo_root / FX / "orig.rds", "--orig-key", "bhat",
               "--cells", CELLS, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    return out


@pytest.mark.parametrize("method,score_type", [
    ("meta", "meta_z"),
    ("nsig", "nsig_ratio"),
    ("pval_pair", "pval_pair"),
    ("finemap", "finemap"),
])
def test_mash_feature_score(method, score_type, run_r, read_rds, repo_root, tmp_path):
    contrast = _contrast(run_r, repo_root, tmp_path)
    out = tmp_path / f"fs_{method}.rds"
    args = ["--method", method, "--contrast", contrast, "--gene-ids", "GENE1",
            "--output", out]
    if method in ("meta", "pval_pair"):
        args += ["--meta-method", "REML"]
    if method == "nsig":
        args += ["--p-cutoff", "0.5"]
    if method == "finemap":
        args += ["--fine-mapping", repo_root / FX / "fine_mapping.rds",
                 "--conditions", CELLS]
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_feature_score.R",
              args, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "data.frame"
    assert {"gene", "condition", "contrast", "score", "scoreType"}.issubset(probe["names"])
    # NOTE: no value-compare on this RDS. mash_feature_score.R names the score
    # rows off the (tmp) contrast-file path via Map()+rbind, so the data.frame's
    # row.names attribute embeds a machine-specific path that the frozen
    # rds_compare.R canon() cannot basename-normalize (it normalizes columns /
    # leaves, not the row.names attribute). The reproducible value-compare lives
    # on the merge step's CSV output (test_mash_feature_score_merge), whose
    # write.csv(row.names=FALSE) drops those path-derived names.


def test_mash_feature_score_merge(run_r, repo_root, tmp_path):
    contrast = _contrast(run_r, repo_root, tmp_path)
    fs = tmp_path / "fs.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_feature_score.R",
              ["--method", "nsig", "--contrast", contrast, "--gene-ids", "GENE1",
               "--p-cutoff", "0.5", "--output", fs], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    out = tmp_path / "merged.csv"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_feature_score_merge.R",
              ["--scores", fs, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()
    assert "scoreType" in out.read_text().splitlines()[0]
    assert_matches_expected(out, repo_root / EXPECTED_MERGE, mode="tolerant",
                            rtol=1e-6, atol=1e-8)
