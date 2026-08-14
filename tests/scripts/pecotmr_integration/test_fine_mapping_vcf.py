"""Script tier: fine_mapping_vcf.R — writeSumstatsVcf over a FineMappingResult.
The mvSuSiE FMR carries two contexts, so --split-by-context emits one VCF per
context (writeSumstatsVcf derives the per-context paths from --output).

Regression: the full VCF (records + schema header) reproduces a committed fixture,
masking only the single volatile line — a ``##fileDate=YYYYMMDD`` stamp with the
current date. The FORMAT values are formatted from stored FMR slots (ES =
conditional_effect / marginal beta, PIP, LBF), not recomputed with drifting
arithmetic, so the data rows are byte-deterministic (verified: two same-day runs
are byte-identical) and portable across the CI arches. ``ignore_lines`` drops the
daily header line so the committed fixture stays valid on any day.
"""
from __future__ import annotations

import gzip

from helpers.expected import assert_matches_expected

FMR = "tests/fixtures/mnm_postprocessing/protocol_example.mvsusie.fine_mapping.rds"
EXPECTED = "tests/fixtures/mnm_postprocessing/expected"
CONTEXTS = ("context1", "context2")


def test_fine_mapping_vcf(run_r, repo_root, tmp_path):
    out = tmp_path / "fm.vcf.bgz"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_vcf.R",
              ["--input", repo_root / FMR, "--output", out,
               "--sample-name", "protocol_example", "--split-by-context"], timeout=200)
    assert p.returncode == 0, p.stdout + p.stderr
    for ctx in CONTEXTS:
        produced = tmp_path / f"fm.{ctx}.vcf.bgz"
        assert produced.exists(), p.stdout
        head = gzip.open(produced, "rt").read(4000)
        assert head.startswith("##fileformat=VCF") and "#CHROM" in head
        # regression: records + schema match the committed VCF, masking only the daily
        # ##fileDate line (numeric coords compared with tolerance; the colon-packed
        # FORMAT sample cell is a byte-deterministic string formatted from stored slots).
        assert_matches_expected(produced, repo_root / EXPECTED / f"fm.{ctx}.vcf.bgz",
                                mode="tolerant", rtol=1e-6, atol=1e-8,
                                ignore_lines=[r"^##fileDate"])
