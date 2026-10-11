"""Tier B: fine_mapping_pip_plot.R renders a REAL PIP PNG from an FMR.

Asserting on the wrapper's own summary line, not just a non-zero file size: the
script's fallback path writes a valid PNG carrying only the words "No usable PIP
vectors", so a size check passes on a blank plot. That is exactly what happened
when pecotmr 0.9.0 renamed getTopLoci() -> topLoci() -- the accessor error was
swallowed, every panel vanished, and this test stayed green. The "Wrote PIP
landscape" line is printed only on the real path.
"""
from __future__ import annotations

import re

import pytest


def test_fine_mapping_pip_plot(fmr, run_r, repo_root, tmp_path):
    out = tmp_path / "pip.png"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_pip_plot.R",
              ["--input", fmr, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "Wrote PIP landscape" in p.stdout, (
        f"stub plot instead of a real one:\n{p.stdout}\n{p.stderr}")
    panels = int(re.search(r"Wrote PIP landscape \((\d+) panel", p.stdout).group(1))
    assert panels > 0, f"rendered {panels} panels"
