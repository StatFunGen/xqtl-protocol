"""Notebook tier: splicing_calling_leafcutter2.ipynb workflows.

The clustering test uses the chr22 GTF + FASTA pair in FIX/reference, built by
build_reference.R to cover the loci the junction fixtures cluster into.
"""
from __future__ import annotations

import gzip
import shutil

import pytest

NB = "code/SoS/molecular_phenotypes/calling/splicing_calling_leafcutter2.ipynb"
FIX = "tests/fixtures/splicing_calling/leafcutter"
GTF = "reference/Homo_sapiens.GRCh38.103.chr22.leafcutter2.reformatted.gtf"
FASTA = "reference/chr22.leafcutter2.fa.gz"


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


def test_cluster_and_classify(run_sos, repo_root, tmp_path):
    """leafcutter2_cluster on the shipped regtools junctions (hence --offset 1).

    --min-clu-reads 3 because the fixtures yield no cluster at the default 30, and
    LeafCutter2 reports clustering success before failing later. The FASTA is copied
    into tmp_path because LeafCutter2 (pyfastx) writes its index next to it.
    """
    fix = repo_root / FIX
    fasta = tmp_path / "ref" / "chr22.leafcutter2.fa.gz"
    fasta.parent.mkdir()
    for ext in ("", ".fai", ".gzi"):
        shutil.copy(fix / (FASTA + ext), str(fasta) + ext)
    p = run_sos(repo_root / NB, "leafcutter2_cluster",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     juncfiles=fix / "protocol_example.junction_files.txt",
                     annotation_gtf=fix / GTF, genome_fasta=fasta,
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
    # the reference fixture covers TTC28 so that some junction is classified against
    # the annotation rather than only labelled intergenic (IN)
    classes = {r.split()[0].split(":")[4] for r in rows[1:]}
    assert classes - {"IN"}, classes
    # cluster_ratios is numerator/denominator -- this is what splicing_normalization parses
    assert all("/" in c for c in cells), cells
    # junction_counts is plain integers, and is NOT a valid normalization input
    crows = [l for l in gzip.open(counts, "rt").read().splitlines() if l.strip()]
    assert "/" not in crows[1].split()[1], crows[1]


# ---------------------------------------------------------------- differential_splicing
# The split and join scripts need only base R, argparser and vroom; the per-part
# leafcutter_ds.R step needs the R leafcutter package and is not exercised here.
import subprocess

from helpers.r_runner import rscript_bin

SCRIPTS = "code/script/molecular_phenotypes/calling"
PARTS = [str(i) for i in range(1, 23)] + ["X", "other"]
SIG_HDR = "status\tloglr\tdf\tp\tcluster\tp.adjust"
EFF_HDR = "intron\tlogef\tA\tB\tdeltapsi"


def _script(repo_root, name, *args):
    return subprocess.run([rscript_bin(), str(repo_root / SCRIPTS / name), *map(str, args)],
                          capture_output=True, text=True)


def _counts(path):
    with gzip.open(path, "wt") as f:
        f.write("S1 S2 S3\n")
        f.write("chr1:10:20:clu_1_+ 1 2 3\n")
        f.write("1:30:40:clu_2_+ 1 2 3\n")
        f.write("chrX:10:20:clu_3_- 4 5 6\n")
        f.write("chrY:10:20:clu_4_- 4 5 6\n")
        f.write("GL000220.1:5:9:clu_5_+ 1 1 1\n")


def test_ds_split_by_chromosome(repo_root, tmp_path):
    """Every row lands in exactly one part, 'chr' or not; all 24 parts exist with the header."""
    counts, groups = tmp_path / "c.counts.gz", tmp_path / "g.txt"
    _counts(counts)
    groups.write_text("S1 A\nS2 B\nS3 B\n")
    p = _script(repo_root, "leafcutter_ds_split.R", "--counts", counts, "--groups", groups,
                "--output-prefix", tmp_path / "parts" / "g")
    assert p.returncode == 0, p.stderr
    got = {}
    for part in PARTS:
        rows = gzip.open(tmp_path / "parts" / f"g.part_{part}.counts.gz", "rt").read().splitlines()
        assert rows[0] == "S1 S2 S3", part
        got[part] = [r.split()[0] for r in rows[1:]]
    assert got["1"] == ["chr1:10:20:clu_1_+", "1:30:40:clu_2_+"]
    assert got["X"] == ["chrX:10:20:clu_3_-"]
    assert got["other"] == ["chrY:10:20:clu_4_-", "GL000220.1:5:9:clu_5_+"]
    assert sum(len(v) for v in got.values()) == 5


@pytest.mark.parametrize("groups, message", [
    ("SAMPLE_X A\nSAMPLE_Y B\n", "no sample"),
    ("S1 A\nS2 B\nS3 C\n", "3 groups"),
])
def test_ds_split_rejects_bad_groups(repo_root, tmp_path, groups, message):
    """A groups file leafcutter_ds.R cannot use stops the run before any part is tested."""
    counts, g = tmp_path / "c.counts.gz", tmp_path / "g.txt"
    _counts(counts)
    g.write_text(groups)
    p = _script(repo_root, "leafcutter_ds_split.R", "--counts", counts, "--groups", g,
                "--output-prefix", tmp_path / "parts" / "g")
    assert p.returncode != 0
    assert message in p.stderr, p.stderr


def test_ds_merge_recomputes_fdr(repo_root, tmp_path):
    """The joined p.adjust is Benjamini-Hochberg over every tested cluster, as R's
    p.adjust(method = "fdr") gives in one unsplit run; empty parts are skipped."""
    def part(name, sig_rows, eff_rows):
        (tmp_path / f"{name}_cluster_significance.txt").write_text(
            "".join(r + "\n" for r in ([SIG_HDR] + sig_rows if sig_rows else [])))
        (tmp_path / f"{name}_effect_sizes.txt").write_text(
            "".join(r + "\n" for r in ([EFF_HDR] + eff_rows if eff_rows else [])))
        return tmp_path / f"{name}_cluster_significance.txt"
    a = part("a", ["Success\t5\t1\t0.01\t1:clu_1_+\t0.02", "Success\t2\t1\t0.04\t1:clu_2_+\t0.04"],
             ["1:1:2:clu_1_+\t0.1\t0.5\t0.6\t0.1"])
    b = part("b", ["Success\t3\t1\t0.03\t2:clu_3_-\t0.03",
                   "<=1 sample with coverage>min_coverage\tNA\tNA\tNA\t2:clu_4_-\tNA"],
             ["2:1:2:clu_3_-\t-0.1\t0.6\t0.5\t-0.1"])
    c = part("c", [], [])
    p = _script(repo_root, "leafcutter_ds_merge.R", "--output-prefix", tmp_path / "all", "--files", a, b, c)
    assert p.returncode == 0, p.stderr
    rows = (tmp_path / "all_cluster_significance.txt").read_text().splitlines()
    assert rows[0] == SIG_HDR
    q = {r.split("\t")[4]: r.split("\t")[5] for r in rows[1:]}
    # m = 3 tested: 0.01 -> 0.03, 0.03 -> min(0.045, 0.04) = 0.04, 0.04 -> 0.04
    assert float(q["1:clu_1_+"]) == pytest.approx(0.03)
    assert float(q["2:clu_3_-"]) == pytest.approx(0.04)
    assert float(q["1:clu_2_+"]) == pytest.approx(0.04)
    assert q["2:clu_4_-"] == "NA"
    eff = (tmp_path / "all_effect_sizes.txt").read_text().splitlines()
    assert eff[0] == EFF_HDR and len(eff) == 3


def test_regtools_driver_rejects_bad_min_mapq(repo_root, tmp_path):
    """--min-mapq is checked before samtools or regtools is called, so this needs neither."""
    p = subprocess.run(["bash", str(repo_root / SCRIPTS / "regtools_junctions.sh"),
                        "--bam", tmp_path / "x.bam", "--output", tmp_path / "x.junc",
                        "--strandness", "XS", "--min-mapq", "unique"],
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert "--min-mapq must be a whole number" in p.stderr, p.stderr
