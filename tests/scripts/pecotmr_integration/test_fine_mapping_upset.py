"""Tier B: fine_mapping_upset.R renders a valid PNG even for an FMR with no
credible sets (the empty-upset path).

The toy FMR genuinely has zero credible sets, so the empty plot IS the expected
result -- but the test asserts the wrapper's reason for it rather than just the
file's existence. Without that, "empty because the input has no CS" is
indistinguishable from "empty because the accessor blew up and was swallowed",
which is how the pecotmr 0.9.0 getCs() -> credibleSets() rename slipped through.
"""
from __future__ import annotations

import pytest


def test_fine_mapping_upset(fmr, run_r, repo_root, tmp_path):
    out = tmp_path / "upset.png"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_upset.R",
              ["--input", fmr, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "No credible sets found across any input" in p.stderr, (
        f"expected the no-CS path for this fixture:\n{p.stdout}\n{p.stderr}")
