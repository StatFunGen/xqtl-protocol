"""Notebook tier: qtl_association_postprocessing.ipynb `default` workflow. Stages
the committed fixture as the work dir (the notebook globs the regional / pairs /
n_variants files by pattern) and runs the sos step end-to-end, asserting the
consolidated enriched-QtlSumStats RDS is produced.
"""
from __future__ import annotations

import shutil

FX = "tests/fixtures/qtl_association_postprocessing"


def test_qtl_association_postprocessing(run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "tensorqtl_cis"
    cwd.mkdir()
    for f in (repo_root / FX).glob("*.gz"):
        shutil.copy(f, cwd / f.name)
    out_dir = tmp_path / "out"

    p = run_sos(
        repo_root / "pipeline/qtl_association_postprocessing.ipynb", "default",
        dict(cwd=cwd, modular_script_dir=repo_root / "code/script", output_dir=out_dir,
             maf_cutoff="0.01", cis_window="1000000", pvalue_cutoff="0.05",
             study="protocol_example", context="bulk_rnaseq", genome="hg38"),
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr

    rds = out_dir / "tensorqtl_cis.qtl_association_postprocessing.rds"
    assert rds.exists(), p.stdout
    assert (out_dir / "tensorqtl_cis.qtl_association_postprocessing.cis_regional.fdr.tsv.gz").exists()
