"""Script tier: qtl_association_postprocessing.R — TensorQTL cis-QTL hierarchical
multiple-testing correction via pecotmr::qtlAssociationPostprocess. Runs on the
committed 10-gene fixture and checks the enriched QtlSumStats RDS + the per-method
regional export, including a value cross-check against the legacy inst-engine
(ENSG00000283047 -> p_bonferroni_min 0.18795830866889446).
"""
from __future__ import annotations

import csv
import gzip

FX = "tests/fixtures/qtl_association_postprocessing"


def test_qtl_association_postprocessing(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / FX
    out = tmp_path / "qap.rds"
    exports = tmp_path / "exports"
    p = run_r(
        repo_root / "code/script/pecotmr_integration/qtl_association_postprocessing.R",
        ["--regional", fx / "protocol_example.cis_qtl.regional.tsv.gz",
         "--pairs", fx / "protocol_example.cis_qtl.pairs.tsv.gz",
         "--n-variants-stats",
         fx / "protocol_example.maf_0.01_window_1000000_cis_n_variants_stats.tsv.gz",
         "--maf-cutoff", "0.01", "--cis-window", "1000000", "--pvalue-cutoff", "0.05",
         "--study", "protocol_example", "--context", "bulk_rnaseq", "--genome", "hg38",
         "--output", out, "--output-dir", exports], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists(), p.stdout

    # The enriched regional export carries the correction columns.
    reg = exports / "qap.cis_regional.fdr.tsv.gz"
    assert reg.exists()
    with gzip.open(reg, "rt") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    header = rows[0].keys()
    for col in ("p_bonferroni_min_original", "fdr_bonferroni_min_original",
                "q_bonferroni_min_original", "p_bonferroni_min_filtered",
                "fdr_beta", "p_nominal_threshold"):
        assert col in header, f"missing correction column {col!r}"

    # Value cross-check against the legacy engine on the same gene.
    row = next(r for r in rows if r["molecular_trait_object_id"] == "ENSG00000283047")
    assert abs(float(row["p_bonferroni_min_original"]) - 0.18795830866889446) < 1e-9

    assert (exports / "qap.summary.tsv").exists()


def _wrapper(run_r, repo_root, out, exports, fdr):
    fx = repo_root / FX
    return run_r(
        repo_root / "code/script/pecotmr_integration/qtl_association_postprocessing.R",
        ["--regional", fx / "protocol_example.cis_qtl.regional.tsv.gz",
         "--pairs", fx / "protocol_example.cis_qtl.pairs.tsv.gz",
         "--n-variants-stats",
         fx / "protocol_example.maf_0.01_window_1000000_cis_n_variants_stats.tsv.gz",
         "--maf-cutoff", "0.01", "--cis-window", "1000000", "--pvalue-cutoff", "0.05",
         "--fdr-threshold", str(fdr),
         "--study", "protocol_example", "--context", "bulk_rnaseq", "--genome", "hg38",
         "--output", out, "--output-dir", exports], timeout=300)


def test_significant_export_branch(run_r, repo_root, tmp_path):
    # The toy has no genome-wide-significant QTLs at 0.05 (verified against the
    # legacy summary), so exercise the significant-export branch at a permissive
    # 0.90 threshold and assert it is INTERNALLY consistent (not the legacy's
    # quirky filtered count): permutation events == the genes with q_beta < 0.90,
    # and the variant-level QTL export is non-empty.
    exports = tmp_path / "exp90"
    p = _wrapper(run_r, repo_root, tmp_path / "qap90.rds", exports, 0.90)
    assert p.returncode == 0, p.stdout + p.stderr

    with gzip.open(exports / "qap90.cis_regional.fdr.tsv.gz", "rt") as fh:
        reg = list(csv.DictReader(fh, delimiter="\t"))
    q_sig = [r for r in reg
             if r["q_beta"] not in ("", "NA") and float(r["q_beta"]) < 0.90]
    assert len(q_sig) > 0

    with gzip.open(exports / "qap90.significant_events.permutation.tsv.gz", "rt") as fh:
        ev = list(csv.DictReader(fh, delimiter="\t"))
    assert len(ev) == len(q_sig)                       # events == q_beta<0.90 genes
    assert (exports / "qap90.significant_qtl.permutation.tsv.gz").exists()

    import csv as _csv
    with open(exports / "qap90.summary.tsv") as fh:
        summ = {r["method"]: int(r["significant_qtl"]) for r in _csv.DictReader(fh, delimiter="\t")}
    assert summ["permutation"] > 0                     # variant-level export non-empty
