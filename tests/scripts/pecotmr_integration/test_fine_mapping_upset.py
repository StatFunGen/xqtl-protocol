"""Tier B: fine_mapping_upset.R renders a valid PNG even for an FMR with no
credible sets (the empty-upset path)."""
from __future__ import annotations

import pytest


def test_fine_mapping_upset(fmr, run_r, repo_root, tmp_path):
    out = tmp_path / "upset.png"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_upset.R",
              ["--input", fmr, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.stat().st_size > 0
