"""Notebook tier: bam_to_bw.ipynb — BAM -> bigWig coverage track.

The single step ``reformat_vcf_wasp`` is inline bash (no worker script): it runs
``samtools index`` then deepTools ``bamCoverage`` on each input BAM, emitting
``<cwd>/<bam-stem>.bw``. Tested end-to-end through SoS.

Fixture: the committed coordinate-sorted chr22:16M-17M BAM from phenotype_formatting.
It is copied into the tmp dir first so ``samtools index`` writes the ``.bai`` there
rather than polluting tests/fixtures.
"""
from __future__ import annotations

import shutil

NB = "pipeline/bam_to_bw.ipynb"
BAM = "tests/fixtures/phenotype_formatting/protocol_example.chr22_16M_17M.bam"


def test_bam_to_bw_via_sos(run_sos, repo_root, tmp_path):
    bam = tmp_path / "protocol_example.chr22_16M_17M.bam"
    shutil.copy(repo_root / BAM, bam)
    p = run_sos(repo_root / NB, "reformat_vcf_wasp",
                {"bam_files": bam, "cwd": tmp_path})
    assert p.returncode == 0, p.stdout + p.stderr
    bw = tmp_path / "protocol_example.chr22_16M_17M.bw"
    assert bw.exists() and bw.stat().st_size > 0        # bigWig written
    assert (tmp_path / "protocol_example.chr22_16M_17M.bam.bai").exists()   # samtools index ran
