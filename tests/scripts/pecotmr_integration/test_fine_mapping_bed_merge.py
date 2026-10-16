"""Script tier: fine_mapping_bed_merge.R — fold the per-region published BED
tables written by fine_mapping.R into one sorted, bgzipped, tabix-indexed
table."""
from __future__ import annotations

import gzip
import subprocess

# The published vocabulary (pecotmr::exportToBed): the VCF's own field
# names, shared by both schemas, not the retired variant_ID / event_ID /
# region_ID / cs_coverage_0.95 spellings.
HEADER = "#chr\tstart\tend\tID\tcontext\tregion_id\tPIP\n"


def _write(path, rows):
    _write_with(path, HEADER, rows)


def _write_with(path, header, rows):
    with gzip.open(path, "wt") as handle:
        handle.write(header)
        for row in rows:
            handle.write("\t".join(str(v) for v in row) + "\n")


def test_merge_sorts_numerically_and_indexes(run_r, repo_root, tmp_path):
    # Two regions, deliberately out of order and spanning chr2 vs chr10 so a
    # plain character sort would put 10 before 2.
    a = tmp_path / "r1.exported.bed.gz"
    b = tmp_path / "r2.exported.bed.gz"
    _write(a, [("2", 300, 301, "2:301:A:T", "c1", "chr2_1_1000", 0.9)])
    _write(b, [("2", 100, 101, "2:101:A:T", "c1", "chr2_1_1000", 0.5),
               ("10", 50, 51, "10:51:A:T", "c1", "chr10_1_1000", 0.7)])
    out = tmp_path / "merged.bed.gz"

    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_bed_merge.R",
              ["--input", a, b, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    lines = gzip.open(out, "rt").read().rstrip("\n").split("\n")
    assert lines[0] == HEADER.rstrip("\n"), "the published header must survive"
    # one header + three data rows, with every region's rows kept
    assert len(lines) == 4
    starts = [int(line.split("\t")[1]) for line in lines[1:]]
    chroms = [line.split("\t")[0] for line in lines[1:]]
    # natural chromosome order, as `sort -k1,1V` gave: 2 before 10, not "10" < "2"
    assert chroms == ["2", "2", "10"], "chromosomes sort naturally"
    assert starts == [100, 300, 50], "positions sort numerically within a chromosome"

    # the index is usable
    assert (tmp_path / "merged.bed.gz.tbi").exists()
    q = subprocess.run(["tabix", str(out), "2:1-1000"],
                       capture_output=True, text=True)
    assert q.returncode == 0 and q.stdout.strip()


def test_merge_of_a_single_region_is_a_passthrough(run_r, repo_root, tmp_path):
    a = tmp_path / "only.exported.bed.gz"
    _write(a, [("22", 10, 11, "22:11:A:T", "c1", "chr22_1_99", 0.4)])
    out = tmp_path / "one.bed.gz"
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_bed_merge.R",
              ["--input", a, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert gzip.open(out, "rt").read() == gzip.open(a, "rt").read()


def test_merge_requires_an_input(run_r, repo_root, tmp_path):
    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_bed_merge.R",
              ["--output", tmp_path / "x.bed.gz"], timeout=300)
    assert p.returncode != 0


# The top-loci schema, which is a half-open BED like the fSuSiE one: the two
# share a vocabulary, so #chr / start / end / ID / REF / ALT here as well.
TOPLOCI_HEADER = "#chr\tstart\tend\tID\tREF\tALT\tPIP\n"

# A position table, which only reaches this step from outside the pipeline --
# no pecotmr writer produces one. Its third column is an ALLELE.
POSITION_HEADER = "chr\tpos\ta1\ta2\tvariant_id\tPIP\n"


def test_merge_indexes_the_top_loci_bed(run_r, repo_root, tmp_path):
    """The top-loci table is a BED now -- #chr/start/end -- so it sorts on its
    coordinates and is tabix-indexed like the fSuSiE one."""
    a = tmp_path / "r1.top_loci.bed.gz"
    b = tmp_path / "r2.top_loci.bed.gz"
    _write_with(a, TOPLOCI_HEADER,
                [("chr2", 299, 300, "chr2:300:A:T", "A", "T", 0.9)])
    _write_with(b, TOPLOCI_HEADER,
                [("chr10", 49, 50, "chr10:50:C:G", "C", "G", 0.7),
                 ("chr2", 99, 100, "chr2:100:G:A", "G", "A", 0.5)])
    out = tmp_path / "merged.top_loci.bed.gz"

    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_bed_merge.R",
              ["--input", a, b, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    lines = gzip.open(out, "rt").read().rstrip("\n").split("\n")
    assert lines[0] == TOPLOCI_HEADER.rstrip("\n")
    rows = [l.split("\t") for l in lines[1:]]
    assert [r[0] for r in rows] == ["chr2", "chr2", "chr10"], "natural chromosome order"
    assert [int(r[1]) for r in rows] == [99, 299, 49], "numeric position order"
    assert (tmp_path / "merged.top_loci.bed.gz.tbi").exists()


def test_merge_does_not_index_a_position_table(run_r, repo_root, tmp_path):
    """A chr/pos/a1 table is not a BED: its third column is an allele, so the
    tabix "bed" preset would read it as an end coordinate."""
    a = tmp_path / "r1.positions.tsv.gz"
    b = tmp_path / "r2.positions.tsv.gz"
    _write_with(a, POSITION_HEADER,
                [("chr2", 300, "A", "T", "chr2:300:A:T", 0.9)])
    _write_with(b, POSITION_HEADER,
                [("chr10", 50, "C", "G", "chr10:50:C:G", 0.7),
                 ("chr2", 100, "G", "A", "chr2:100:G:A", 0.5)])
    out = tmp_path / "merged.positions.tsv.gz"

    p = run_r(repo_root / "code/script/pecotmr_integration/fine_mapping_bed_merge.R",
              ["--input", a, b, "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    lines = gzip.open(out, "rt").read().rstrip("\n").split("\n")
    assert lines[0] == POSITION_HEADER.rstrip("\n")
    rows = [l.split("\t") for l in lines[1:]]
    assert [r[0] for r in rows] == ["chr2", "chr2", "chr10"], "natural chromosome order"
    assert [int(r[1]) for r in rows] == [100, 300, 50], "numeric position order"
    # alleles survive untouched in the allele column
    assert [r[2] for r in rows] == ["G", "A", "C"]
    assert not (tmp_path / "merged.positions.tsv.gz.tbi").exists()
