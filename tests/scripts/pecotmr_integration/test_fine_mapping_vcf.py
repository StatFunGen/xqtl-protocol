"""Script tier: fine_mapping_vcf.R — writeSumstatsVcf over a FineMappingResult.
The mvSuSiE FMR carries two contexts, so --split-by-context emits one VCF per
context (writeSumstatsVcf derives the per-context paths from --output)."""
from __future__ import annotations

import glob
import gzip

FMR = "tests/fixtures/mnm_postprocessing/protocol_example.mvsusie.fine_mapping.rds"


def test_fine_mapping_vcf(run_r, repo_root, tmp_path):
    out = tmp_path / "fm.vcf.bgz"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_vcf.R",
              ["--input", repo_root / FMR, "--output", out,
               "--sample-name", "protocol_example", "--split-by-context"], timeout=200)
    assert p.returncode == 0, p.stdout + p.stderr
    vcfs = glob.glob(str(tmp_path / "**" / "*.vcf.bgz"), recursive=True)
    assert vcfs, p.stdout
    head = gzip.open(vcfs[0], "rt").read(4000)
    assert head.startswith("##fileformat=VCF") and "#CHROM" in head
