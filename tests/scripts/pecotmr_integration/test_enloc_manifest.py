"""Tier B: enloc_manifest.R pairs xQTL x GWAS units in both modes (4 units each)."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("mode,sep", [("enrichment", None), ("coloc", "@")])
def test_enloc_manifest(run_r, repo_root, tmp_path, mode, sep):
    fx = repo_root / "tests/fixtures/susie_enloc"
    out = tmp_path / f"{mode}.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/enloc_manifest.R",
              ["--mode", mode,
               "--xqtl-meta", fx / "protocol_example.enloc.xqtl_meta.tsv",
               "--gwas-meta", fx / "protocol_example.enloc.gwas_meta.tsv",
               "--context-meta", fx / "protocol_example.enloc.context_meta.tsv",
               "--qtl-path", fx, "--gwas-path", fx, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert rows[0] == ["unit_id", "qtl_files", "gwas_files"]
    assert len(rows) == 5           # header + 4 units
    if sep:
        assert all(sep in r[0] for r in rows[1:])
