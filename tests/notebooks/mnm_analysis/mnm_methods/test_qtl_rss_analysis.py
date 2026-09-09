"""Notebook tier: qtl_rss_analysis.ipynb — RSS fine-mapping and TWAS weights from
cis-QTL summary statistics.

Drives the whole chain on committed chr22 toy data: the 16-gene TensorQTL nominal
table (tests/fixtures/tensorqtl/expected) restricted to one gene, with the qtl_mini
genotypes doubling as the LD reference panel. That is IN-SAMPLE LD — the same 49
samples the scan itself used — which is fine for exercising the code path but
optimistic statistically; a real run supplies a separate reference panel.

The variant ids here are PLINK .bim order (chr:pos:A1:A2), so the notebook is given
`--variant-id-alleles A1A2`. Declaring that wrong does not error: summaryStatsQc
would "correct" the apparent mismatch by sign- and strand-flipping every variant, so
the QC line asserted below (0 flips) is the check that the orientation is right.
"""
from __future__ import annotations

from helpers.expected import assert_matches_expected

GENE = "ENSG00000283047"
STUB = f"test_study.context1.{GENE}"
EXP = "tests/fixtures/qtl_rss_analysis/expected"


def test_qtl_rss(run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "qtl_rss"
    fx = repo_root / "tests" / "fixtures"
    p = run_sos(
        repo_root / "pipeline/qtl_rss_analysis.ipynb",
        "qtl_rss",
        {
            "cwd": cwd,
            "modular_script_dir": repo_root / "code/script",
            "sumstats": fx / "tensorqtl/expected/cis_qtl.pairs.tsv.gz",
            "ld-sketch": fx / "qtl_mini/protocol_example.genotype.chr22.bed",
            "study": "test_study",
            "context": "context1",
            "trait": GENE,
            "variant-id-alleles": "A1A2",   # these ids are .bim order
            "methods": "susie",
            "twas-methods": "lasso",
            "seed": 999,
        },
        cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr

    ss = cwd / f"sumstats/{STUB}.qtl_sumstats.rds"
    fm = cwd / f"fine_mapping/{STUB}.qtl_rss_finemap.rds"
    tw = cwd / f"twas_weights/{STUB}.qtl_rss_twas_weights.rds"
    assert ss.exists() and fm.exists() and tw.exists(), p.stdout

    assert read_rds(ss)["class"] == "QtlSumStats"
    info = read_rds(fm)
    assert info["class"] == "QtlFineMappingResult"
    assert info["MethodNames"] == ["susie"]
    assert read_rds(tw)["class"] == "TwasWeights"

    # The gene's cis window is 200 of the table's 22,742 rows: the trait filter ran.
    # Zero flips means the declared allele order agrees with the LD panel — if it
    # did not, QC would silently flip all 200 and every downstream number would be
    # wrong while the run still "succeeded".
    out = (cwd / f"sumstats/{STUB}.qtl_sumstats.stdout").read_text()
    assert "200 variant(s)" in out, out
    qc = (cwd / f"sumstats/{STUB}.qtl_sumstats.stderr").read_text()   # message(), not stdout
    assert "sign-flip 0, strand-flip 0" in qc, qc

    # regression: all three are deterministic (QC/reshaping has no RNG; the susie_rss
    # and lassosum fits are seeded) and byte-identical run-to-run on one machine.
    # normalize_paths: the ldSketch embeds the genotype-panel path.
    #
    # The QtlSumStats is deterministic reshaping + QC with no linear algebra, so it
    # holds at rtol=1e-6. The two FITS do not: the LD here is in-sample from 49
    # samples, so a 200-variant panel has rank <= 48 and the solves sit close to
    # singular, where log-Bayes-factor sums amplify last-bit BLAS differences. The
    # macOS-vs-Linux drift measured on CI is 3.5e-6 (logBF) and 5.3e-6 (susieFit
    # lbf_variable) -- bounded rounding, not a behaviour change, so tolerance is the
    # right tool (cf. the TensorQTL fixtures, which use rtol=1e-4 for the same
    # reason). rtol=1e-4 leaves ~20x headroom over the observed drift while staying
    # four orders of magnitude tighter than a real regression: when the SuSiE L
    # default changed, the same comparison moved by 0.49 relative.
    exp = repo_root / EXP
    assert_matches_expected(ss, exp / "qtl_sumstats.rds", mode="tolerant",
                            rtol=1e-6, atol=1e-8, normalize_paths=True)
    assert_matches_expected(fm, exp / "qtl_rss_finemap.rds", mode="tolerant",
                            rtol=1e-4, atol=1e-8, normalize_paths=True)
    # Same LD, same rank deficiency: loosened on the same basis rather than waiting
    # for a second CI round to measure it (the fit was never reached on the run that
    # surfaced the fine-mapping drift).
    assert_matches_expected(tw, exp / "qtl_rss_twas_weights.rds", mode="tolerant",
                            rtol=1e-4, atol=1e-8, normalize_paths=True)
