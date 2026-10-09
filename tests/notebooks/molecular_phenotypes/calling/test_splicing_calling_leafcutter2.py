"""Notebook tier: splicing_calling_leafcutter2.ipynb workflows.

LeafCutter2 2.x is not in any conda channel yet (only a broken 0.0.1 on dnachun),
so the tests that need the binary skip until it is installable. The argument-contract
test needs no binary and always runs.

Reference GTF/FASTA are not in the repository, so the full clustering test is gated
on XQTL_REFERENCE_DATA pointing at a directory holding the protocol's pair.
"""
from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

import pytest

NB = "code/SoS/molecular_phenotypes/calling/splicing_calling_leafcutter2.ipynb"
FIX = "tests/fixtures/splicing_calling/leafcutter"
GTF = "Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.gtf"
FASTA = "GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta"

# 2.x exposes leafcutter2-star2junc; pre-2.0 builds do not. Probe by command rather than by
# version string: packaged builds do not always carry the upstream version (the dnachun conda
# build is labelled 1.0.1 but ships 2.0.1), so a version comparison would wrongly reject it.
needs_lc2 = pytest.mark.skipif(
    shutil.which("leafcutter2-star2junc") is None,
    reason="LeafCutter2 2.x not installed (no leafcutter2-star2junc on PATH)",
)
_ref = os.environ.get("XQTL_REFERENCE_DATA")
needs_ref = pytest.mark.skipif(
    not (_ref and (Path(_ref) / GTF).exists() and (Path(_ref) / FASTA).exists()),
    reason="XQTL_REFERENCE_DATA not set to a directory holding the protocol GTF and FASTA",
)


def test_cluster_requires_explicit_offset(run_sos, repo_root, tmp_path):
    """--offset has no default on leafcutter2_cluster by design: it depends on the junction
    file format and the wrong value corrupts classification silently. Omitting it must fail
    at argument parsing, before any work is done, so this needs no binary and no reference data."""
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "leafcutter2_cluster",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     juncfiles=fix / "protocol_example.junction_files.txt",
                     annotation_gtf=tmp_path / "absent.gtf",
                     genome_fasta=tmp_path / "absent.fa"),
                cwd=repo_root, timeout=300)
    assert p.returncode != 0
    assert "--offset" in (p.stdout + p.stderr)


@needs_lc2
def test_star2junc(run_sos, repo_root, tmp_path):
    """Converts the shipped SJ.out.tab fixtures to BED6. Needs no reference data."""
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "star2junc",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     data_dir=fix),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr

    for sid in ("SAMPLE_001", "SAMPLE_002"):
        out = tmp_path / f"{sid}.SJ.out.junc.gz"
        assert out.exists(), f"missing {out}"
        rows = [l for l in gzip.open(out, "rt").read().splitlines() if l.strip()]
        assert rows, "no junctions written"
        # BED6: chrom, start, end, name, score, strand
        assert len(rows[0].split("\t")) == 6, rows[0]


@needs_lc2
@needs_ref
def test_cluster_and_classify(run_sos, repo_root, tmp_path):
    """leafcutter2_cluster on the shipped regtools junctions (hence --offset 1).

    --min-clu-reads 3 because the chr22:16M-17M fixtures yield no cluster at the
    default 30, and LeafCutter2 reports clustering success before failing later.
    """
    fix = repo_root / FIX
    ref = Path(os.environ["XQTL_REFERENCE_DATA"])
    p = run_sos(repo_root / NB, "leafcutter2_cluster",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     juncfiles=fix / "protocol_example.junction_files.txt",
                     annotation_gtf=ref / GTF, genome_fasta=ref / FASTA,
                     offset=1, min_clu_reads=3),
                cwd=repo_root, timeout=1800)
    assert p.returncode == 0, p.stdout + p.stderr

    ratios = tmp_path / "protocol_example.cluster_ratios.gz"
    counts = tmp_path / "protocol_example.junction_counts.gz"
    assert ratios.exists() and counts.exists()

    rows = [l for l in gzip.open(ratios, "rt").read().splitlines() if l.strip()]
    assert len(rows) > 1, "no clusters survived filtering"
    assert rows[0].split()[1:] == ["SAMPLE_001", "SAMPLE_002"], rows[0]

    chrom, *cells = rows[1].split()
    # intron id carries the 5th :CLASS field that LeafCutter1 ids lack
    parts = chrom.split(":")
    assert len(parts) == 5, chrom
    assert parts[4] in {"PR", "UP", "NE", "IN"}, chrom
    # cluster_ratios is numerator/denominator -- this is what splicing_normalization parses
    assert all("/" in c for c in cells), cells
    # junction_counts is plain integers, and is NOT a valid normalization input
    crows = [l for l in gzip.open(counts, "rt").read().splitlines() if l.strip()]
    assert "/" not in crows[1].split()[1], crows[1]
