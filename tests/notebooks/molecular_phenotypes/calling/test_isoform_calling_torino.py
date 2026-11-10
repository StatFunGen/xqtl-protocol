"""Notebook tier: isoform_calling_torino.ipynb workflows and the scripts behind them.

The fixture is 24 single-end BAMs over AP000547.3 (chr22), simulated by
build_fixture.R from three of its transcripts in proportions recorded in
simulated_isoform_proportions.tsv. The coverage, depth-matrix and usage steps
need only bedtools, samtools, htslib and R (argparser, data.table); the
factorization step also needs the torino R package.
"""
from __future__ import annotations

import csv
import gzip
import subprocess

import pytest

from helpers.r_runner import rscript_bin

NB = "code/SoS/molecular_phenotypes/calling/isoform_calling_torino.ipynb"
FIX = "tests/fixtures/splicing_calling/torino"
SCRIPTS = "code/script/molecular_phenotypes/calling"
GENE = "ENSG00000283633"            # AP000547.3, carries the reads
EMPTY_GENE = "ENSG00000287285"      # AP000547.4, no reads
READ_LEN = 100
# retained intron of the simulated intron-retention transcript, 1-based inclusive
RETAINED = (16614179, 16614698)
SKIPPED_INTRON = (16602216, 16611657)


def _truth(repo_root):
    with open(repo_root / FIX / "simulated_isoform_proportions.tsv") as fh:
        return {r["sample_id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def _matrix(path):
    """Depth matrix CSV -> (positions, {sample: [depth, ...]})."""
    with gzip.open(path, "rt") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0] == "Sample_ID", rows[0][:3]
    pos = [int(c.rsplit(":", 1)[1]) for c in rows[0][1:]]
    return pos, {r[0]: [int(x) for x in r[1:]] for r in rows[1:]}


def _bed(path):
    with gzip.open(path, "rt") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]


def _sh(repo_root, name, *args):
    return subprocess.run(["bash", str(repo_root / SCRIPTS / name), *map(str, args)],
                          capture_output=True, text=True)


def _r(repo_root, name, *args):
    return subprocess.run([rscript_bin(), str(repo_root / SCRIPTS / name), *map(str, args)],
                          capture_output=True, text=True)


# ------------------------------------------------------------------ torino_preprocessing
def test_preprocessing_writes_depth_matrices(run_sos, repo_root, tmp_path):
    """Coverage and depth matrices on the fixture: the gene with reads is written, the one
    without is listed as skipped, and the matrix holds exactly the simulated reads."""
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "torino_preprocessing",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     genes=fix / "protocol_example.genes.bed"),
                cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr

    listed = (tmp_path / "count_matrix" / "protocol_example.chunk_1.list").read_text().split()
    assert [s.rsplit("/", 1)[-1] for s in listed] == [f"{GENE}.csv.gz"]
    skipped = (tmp_path / "count_matrix" / "protocol_example.chunk_1.skipped.tsv").read_text().splitlines()
    assert skipped[0] == "gene_id\tgene_name\treason"
    assert [r.split("\t")[0] for r in skipped[1:]] == [EMPTY_GENE]

    pos, depth = _matrix(listed[0])
    # gene window: BED start 16601910 (0-based) and end 16615111, 50 bases of flank each side
    assert pos[0] == 16601910 - 50 + 1 and pos[-1] == 16615111 + 50
    assert pos == list(range(pos[0], pos[-1] + 1))
    truth = _truth(repo_root)
    assert list(depth) == list(truth), "rows must follow the manifest order"
    for sid, d in depth.items():
        # every simulated read is 100 aligned bases, so summed depth is reads x 100
        assert sum(d) == int(truth[sid]["reads"]) * READ_LEN, sid
        # spliced reads add no depth to the first intron, which no transcript retains
        assert not any(d[i] for i, x in enumerate(pos) if SKIPPED_INTRON[0] <= x <= SKIPPED_INTRON[1]), sid

    # depth in the retained intron tracks the simulated intron-retention fraction
    ir_depth = {s: sum(d[i] for i, x in enumerate(pos) if RETAINED[0] <= x <= RETAINED[1]) / sum(d)
                for s, d in depth.items()}
    a = [ir_depth[s] for s in truth if truth[s]["group"] == "A"]
    b = [ir_depth[s] for s in truth if truth[s]["group"] == "B"]
    assert max(a) < min(b), (a, b)


def test_gene_chunks_split_the_gene_list(run_sos, repo_root, tmp_path):
    """--gene-chunks 2 runs one depth-matrix job per slice of the gene list."""
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "torino_preprocessing",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     genes=fix / "protocol_example.genes.bed", gene_chunks=2),
                cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    cm = tmp_path / "count_matrix"
    assert (cm / "protocol_example.chunk_1.list").read_text().split() == []
    assert (cm / "protocol_example.chunk_1.skipped.tsv").read_text().splitlines()[1].startswith(EMPTY_GENE)
    assert (cm / "protocol_example.chunk_2.list").read_text().split()[0].endswith(f"{GENE}.csv.gz")


# ------------------------------------------------------------------ torino_coverage.sh
def test_stranded_coverage_partitions_reads(repo_root, tmp_path):
    """rf writes plus- and minus-strand files whose depths add up to the unstranded depth."""
    bam = repo_root / FIX / "bam" / "SAMPLE_001.bam"
    for mode in ("strand_missing", "rf"):
        p = _sh(repo_root, "torino_coverage.sh", "--bam", bam, "--output-prefix", tmp_path / mode / "S",
                "--strandness", mode)
        assert p.returncode == 0, p.stderr

    def total(path):
        return sum((int(r[2]) - int(r[1])) * int(r[3]) for r in _bed(path))
    unstranded = total(tmp_path / "strand_missing" / "S.coverage.bedgraph.gz")
    plus = total(tmp_path / "rf" / "S.plus.coverage.bedgraph.gz")
    minus = total(tmp_path / "rf" / "S.minus.coverage.bedgraph.gz")
    assert unstranded == int(_truth(repo_root)["SAMPLE_001"]["reads"]) * READ_LEN
    assert plus + minus == unstranded and plus > 0 and minus > 0
    assert (tmp_path / "rf" / "S.plus.coverage.bedgraph.gz.tbi").exists()


@pytest.mark.parametrize("args, message", [
    (["--strandness", "XS"], "--strandness must be strand_missing, rf or fr"),
    (["--strandness", "rf", "--min-mapq", "unique"], "--min-mapq must be a whole number"),
])
def test_coverage_rejects_bad_arguments(repo_root, tmp_path, args, message):
    """Bad options stop the script before samtools or bedtools is called."""
    p = _sh(repo_root, "torino_coverage.sh", "--bam", tmp_path / "x.bam",
            "--output-prefix", tmp_path / "x", *args)
    assert p.returncode != 0
    assert message in p.stderr, p.stderr


# ------------------------------------------------------------------ torino_usage.R
def _fake_factorizations(repo_root, tmp_path):
    """RDS files in the layout torino::run_torino_gene writes: one result for the gene with
    reads (SAMPLE_021 dropped, as Torino drops samples below min_counts) and one NULL,
    as Torino returns when the factorization fails."""
    script = f"""
        set.seed(1)
        s <- sprintf("SAMPLE_%03d", c(1:20, 22:24))
        el <- matrix(runif(length(s) * 3), length(s), 3, dimnames = list(s, paste0("factor_", 1:3)))
        el <- el / rowSums(el)
        saveRDS(list(gene = "{GENE}", torino_K3 = list(EL = el, samples = s), coords_all = "chr22:1"),
                "{tmp_path}/ok.rds")
        saveRDS(list(gene = "{EMPTY_GENE}", torino_K3 = NULL, coords_all = NULL), "{tmp_path}/null.rds")
    """
    p = subprocess.run([rscript_bin(), "-e", script], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return tmp_path / "ok.rds", tmp_path / "null.rds"


def test_usage_tables(repo_root, tmp_path):
    ok, null = _fake_factorizations(repo_root, tmp_path)
    fix = repo_root / FIX
    p = _r(repo_root, "torino_usage.R", "--factorizations", ok, null,
           "--genes", fix / "protocol_example.genes.bed",
           "--samples", fix / "protocol_example.sample_list.txt",
           "--output-prefix", tmp_path / "out" / "cohort")
    assert p.returncode == 0, p.stderr
    assert EMPTY_GENE in p.stderr                      # the NULL factorization is reported

    raw = _bed(tmp_path / "out" / "cohort.torino_usage.bed.gz")
    samples = list(_truth(repo_root))
    assert raw[0] == ["#chr", "start", "end", "ID"] + samples
    assert [r[3] for r in raw[1:]] == [f"{GENE}:factor_{k}" for k in (1, 2, 3)]
    assert all(r[:3] == ["chr22", "16601910", "16615111"] for r in raw[1:])
    col = raw[0].index("SAMPLE_021")
    assert all(r[col] == "NA" for r in raw[1:])
    # usages of one sample sum to 1 across a gene's components in the fake input
    assert sum(float(r[4]) for r in raw[1:]) == pytest.approx(1.0)

    norm = _bed(tmp_path / "out" / "cohort.torino_usage.qqnorm.bed.gz")
    assert norm[0] == raw[0] and len(norm) == len(raw)
    assert "NA" not in {v for r in norm[1:] for v in r[4:]}
    groups = (tmp_path / "out" / "cohort.torino_usage.phenotype_group.txt").read_text().splitlines()
    assert groups == [f"{GENE}:factor_{k}\t{GENE}" for k in (1, 2, 3)]
    for f in ("cohort.torino_usage.bed.gz.tbi", "cohort.torino_usage.qqnorm.bed.gz.tbi"):
        assert (tmp_path / "out" / f).exists()


def test_usage_filter_can_drop_everything(repo_root, tmp_path):
    """A threshold no component meets is an error, not an empty phenotype table."""
    ok, _ = _fake_factorizations(repo_root, tmp_path)
    fix = repo_root / FIX
    p = _r(repo_root, "torino_usage.R", "--factorizations", ok,
           "--genes", fix / "protocol_example.genes.bed",
           "--samples", fix / "protocol_example.sample_list.txt",
           "--output-prefix", tmp_path / "out" / "cohort", "--min-usage-q90", 2)
    assert p.returncode != 0
    assert "no component passes" in p.stderr, p.stderr


# ------------------------------------------------------------------ torino (needs torino)
def test_torino_recovers_intron_retention(run_sos, repo_root, tmp_path):
    """End to end on the fixture. One component's usage must follow the simulated
    intron-retention fraction, the variable that separates the two sample groups."""
    fix = repo_root / FIX
    p = run_sos(repo_root / NB, "torino",
                dict(cwd=tmp_path, samples=fix / "protocol_example.sample_list.txt",
                     genes=fix / "protocol_example.genes.bed", K=3),
                cwd=repo_root, timeout=1800)
    assert p.returncode == 0, p.stdout + p.stderr
    raw = _bed(tmp_path / "protocol_example.torino_usage.bed.gz")
    assert [r[3] for r in raw[1:]] == [f"{GENE}:factor_{k}" for k in (1, 2, 3)]

    truth = _truth(repo_root)
    samples = raw[0][4:]
    ir = [float(truth[s]["intron_retention"]) for s in samples]

    def cor(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        return sxy / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    best = max(abs(cor([float(v) for v in r[4:]], ir)) for r in raw[1:])
    assert best > 0.8, best
