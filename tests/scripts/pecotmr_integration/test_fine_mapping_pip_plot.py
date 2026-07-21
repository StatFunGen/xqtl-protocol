"""Tier B: fine_mapping_pip_plot.R renders a non-empty PIP PNG from an FMR."""
from __future__ import annotations

import pytest


def test_fine_mapping_pip_plot(fmr, run_r, repo_root, tmp_path):
    out = tmp_path / "pip.png"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_pip_plot.R",
              ["--input", fmr, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.stat().st_size > 0
