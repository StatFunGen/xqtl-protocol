"""Script tier: fine_mapping_top_loci_bed.R — top-loci BED export from a
QtlFineMappingResult (getTopLoci joined to getMarginalEffects)."""
from __future__ import annotations

import gzip

from helpers.expected import assert_matches_expected

FMR = "tests/fixtures/mnm_postprocessing/protocol_example.mvsusie.fine_mapping.rds"
# Deterministic reshaping of a committed FMR (no RNG, basenames only, no embedded
# absolute paths); the .gz is decompressed before comparison so gzip mtime bytes
# don't cause false diffs.
EXPECTED = "tests/fixtures/fine_mapping/expected/top_loci.bed.gz"


def test_fine_mapping_top_loci_bed(run_r, repo_root, tmp_path):
    out = tmp_path / "top_loci.bed.gz"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_top_loci_bed.R",
              ["--input", repo_root / FMR, "--signal-cutoff", "0", "--output", out],
              timeout=200)
    assert p.returncode == 0, p.stdout + p.stderr
    header = gzip.open(out, "rt").readline().rstrip("\n").split("\t")
    assert "variant_ID" in header and "PIP" in header
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant",
                            rtol=1e-6, atol=1e-8)
