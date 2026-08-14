"""Notebook tier: gregor.ipynb (GREGOR regulatory-annotation enrichment).

GREGOR is an external Perl tool (conda `gregor`); it is not ported. The `gregor`
workflow chains three steps: `gregor_1` writes the `.conf`, `gregor_2` runs
`GREGOR` (on PATH) against a reference database, and `gregor_3` parses GREGOR's
per-annotation overlap output into a 2x2 table and runs Fisher's exact test.

The real EUR reference is ~20 GB; a self-consistent chr22 slice is committed under
`tests/fixtures/gregor/EUR/` (see that dir's README for how it's carved). To match
that reference, index SNPs are `chr:pos` (hg19) and `ld_window_size` must be the 1 Mb
window the r2>=0.7 databases were built with; `min_neighbor=2` fits the fixture's
small matched-control cubes. `gregor_fisher_plot` is self-contained R and is tested
from a committed enrichment-results table alone (no GREGOR needed).
"""
from __future__ import annotations

from helpers.expected import assert_matches_expected

NB = "pipeline/gregor.ipynb"
FX = "tests/fixtures/gregor"


def test_gregor_enrichment(run_sos, repo_root, tmp_path):
    """Full `gregor` workflow: write conf -> run GREGOR against the committed chr22
    EUR reference slice -> parse overlaps + Fisher test. All 4 chr22 index SNPs annotate
    and the enrichment table carries the per-annotation odds ratio / Fisher p-value."""
    fx = repo_root / FX
    cwd = tmp_path / "gregor"
    # bed-file index lists absolute paths to the annotation bed(s)
    bed_index = tmp_path / "bed.file.index"
    bed_index.write_text(f"{fx / 'test_peaks.bed'}\n")

    p = run_sos(repo_root / NB, "gregor",
                {"gregor_db": fx,
                 "index_snp_file": fx / "index.snps.txt",
                 "bed_file_index": bed_index,
                 "pop": "EUR",
                 "ld_window_size": 1000000,   # must match the reference's build window
                 "min_neighbor": 2,
                 "cwd": cwd},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr

    # GREGOR's raw overlap statistics (gregor_2)
    stat = cwd / "index_gregor_output" / "StatisticSummaryFile.txt"
    assert stat.exists(), p.stdout
    # all 4 index SNPs were annotatable against the slice
    annotated = (cwd / "index_gregor_output" / "index_SNP" / "annotated.index.snp.txt").read_text().splitlines()
    assert len(annotated) - 1 == 4, annotated                 # header + 4 SNPs

    # parsed enrichment table with Fisher test (gregor_3)
    res = cwd / "index_gregor_output_enrichment_results.txt"
    assert res.exists(), p.stdout
    rows = [ln.split() for ln in res.read_text().splitlines() if ln.strip()]
    header, data = rows[0], rows[1:]
    for col in ("Bed_File", "InBed_Index_SNP", "odds", "p_fisher"):
        assert col in header, header
    assert any(r[header.index("Bed_File")] == "test_peaks.bed" for r in data), data
    # odds ratio and Fisher p are finite numbers in range
    oi, pi = header.index("odds"), header.index("p_fisher")
    for r in data:
        assert float(r[oi]) >= 0
        assert 0 <= float(r[pi]) <= 1
    # regression: the whole parsed table (counts + Fisher odds/CI/p) is deterministic
    # against the fixed chr22 reference, so it reproduces the committed snapshot within
    # tol (whitespace-delimited; Bed_File is a basename, no abs paths).
    assert_matches_expected(res, repo_root / FX / "expected" / "enrichment_results.txt",
                            mode="tolerant", rtol=1e-6, atol=1e-8)


def test_gregor_fisher_plot(run_sos, repo_root, tmp_path):
    """`gregor_fisher_plot`: self-contained R odds-ratio comparison plot from two
    enrichment-results tables -> a PDF. No GREGOR needed."""
    fx = repo_root / FX
    cwd = tmp_path / "plot"
    p = run_sos(repo_root / NB, "gregor_fisher_plot",
                {"fisher1": fx / "example_enrichment_results.txt",
                 "fisher2": fx / "example_enrichment_results.txt",
                 "cwd": cwd},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    pdfs = list(cwd.glob("*_enrichment.pdf"))
    assert pdfs and pdfs[0].stat().st_size > 0, p.stdout
