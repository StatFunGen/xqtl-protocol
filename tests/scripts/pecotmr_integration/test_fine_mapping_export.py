"""Tier B: fine_mapping_export projects an FMR into a flat TSV view."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

# The export views are a deterministic reshaping of the FMR (no RNG of their own);
# the fixtures were generated from a --seed 1 fine_mapping.R fit saved as `fmr.rds`
# (matching the `fmr` fixture's filename) so the `source` basename column lines up.
# The QtlFineMappingResult content for this gene is seed-invariant on one machine,
# so the un-seeded `fmr` fixture reproduces the same values; `tolerant` absorbs any
# x86/ARM float drift across the CI matrix.
EXPECTED = "tests/fixtures/fine_mapping/expected"


@pytest.mark.parametrize("view", ["pip", "topLoci", "marginals"])
def test_fine_mapping_export_populated_views(fmr, run_r, repo_root, tmp_path, view):
    out = tmp_path / f"export_{view}.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_export.R",
              ["--input", fmr, "--view", view, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = out.read_text().splitlines()
    assert len(lines) > 1, f"{view} view has no data rows"    # header + >=1 row
    assert_matches_expected(out, repo_root / EXPECTED / f"export_{view}.tsv",
                            mode="tolerant", rtol=1e-6, atol=1e-8)


def test_fine_mapping_export_cs_view(fmr, run_r, repo_root, tmp_path):
    # The toy FMR has no credible sets: the cs view is a valid, header-only TSV.
    out = tmp_path / "export_cs.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_export.R",
              ["--input", fmr, "--view", "cs", "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_text().splitlines(), "cs view is empty (no header)"
    assert_matches_expected(out, repo_root / EXPECTED / "export_cs.tsv",
                            mode="tolerant", rtol=1e-6, atol=1e-8)
