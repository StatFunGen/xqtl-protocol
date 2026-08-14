"""Script tier: fine_mapping_overlap.R — allele-aware QTL x GWAS top-loci overlap
(overlapTopLoci). Uses the susie_enloc chr1 fixtures: the chr1 gene's top loci
overlap the two committed chr1 GWAS blocks."""
from __future__ import annotations

from helpers.expected import assert_matches_expected

QTL = "tests/fixtures/susie_enloc/protocol_example.enloc.MiGA_eQTL.ENSG00000142798.fine_mapping.rds"
GWAS1 = "tests/fixtures/susie_enloc/protocol_example.enloc.gwas.chr1_20110062_22020160.fine_mapping.rds"
GWAS2 = "tests/fixtures/susie_enloc/protocol_example.enloc.gwas.chr1_22020160_24199848.fine_mapping.rds"
# Deterministic allele-aware overlap of committed FMRs (no RNG). The only source
# column is `gwas_source`, a basename — no embedded absolute paths.
EXPECTED = "tests/fixtures/fine_mapping/expected/overlapped.gwas.tsv"


def test_fine_mapping_overlap(run_r, repo_root, tmp_path):
    out = tmp_path / "overlapped.gwas.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_overlap.R",
              ["--qtl", repo_root / QTL,
               "--gwas", repo_root / GWAS1, repo_root / GWAS2,
               "--signal-cutoff", "0", "--output", out], timeout=200)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()
    assert out.read_text().splitlines()                      # header (+ overlap rows)
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant",
                            rtol=1e-6, atol=1e-8)
