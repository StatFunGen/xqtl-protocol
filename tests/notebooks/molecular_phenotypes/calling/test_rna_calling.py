"""Notebook tier: RNA_calling.ipynb R workers (the pandas merges ported from RNA_calling.py).

RNA_calling.R now covers rnaseqc_merge / star_align_3 / rsem_call_2 (previously
RNA_calling.py) alongside aggregate_picard_qc. Validation target is the python it
replaces (the committed MWE merged references predate it and are stale):
  * star_align_3   — pure filename derivation -> BYTE-IDENTICAL (also matches the
    committed MWE bam_file_list manifest).
  * rnaseqc_merge  — GCT outer-merge + metrics; values carried as text ->
    BYTE-IDENTICAL to pandas (fixtures are synthetic; RNA-SeQC is not run in the
    MWE toy example).
  * rsem_call_2    — 7 merged matrices; numeric columns are numerically identical
    to pandas (float repr is cosmetic), labels/row-order exact. The .cnt metrics
    match except the machine-dependent File path column.
Fixtures are the MWE per-sample RSEM results (downsampled to 500 rows) + synthetic
RNA-SeQC GCTs.
"""
from __future__ import annotations

import gzip
import subprocess

WORKER = "code/script/molecular_phenotypes/calling/RNA_calling.R"
SH = "code/script/molecular_phenotypes/calling/RNA_calling.sh"
FIX = "tests/fixtures/rna_calling"


def _rows(text, sep="\t"):
    return [ln.split(sep) for ln in text.splitlines() if ln.strip()]


def _sam_body(bam, extra=()):
    """SAM records (no header) of a BAM, as a list of lines — for deterministic
    BAM-content comparison via an independent samtools call."""
    out = subprocess.run(["samtools", "view", *extra, str(bam)],
                         capture_output=True, text=True, check=True)
    return [ln for ln in out.stdout.splitlines() if ln]


def _assert_num_match(got_text, exp_text, label_cols, sep="\t", tol=1e-9):
    a, b = _rows(got_text, sep), _rows(exp_text, sep)
    assert len(a) == len(b), f"row count {len(a)} != {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert len(ra) == len(rb), f"col count differs at row {i}"
        for j, (x, y) in enumerate(zip(ra, rb)):
            if i == 0 or j < label_cols or x == "NA" or y == "NA":
                assert x == y, f"mismatch row {i} col {j}: {x!r} != {y!r}"
            else:
                assert abs(float(x) - float(y)) < tol, f"num diff row {i} col {j}: {x} vs {y}"


def test_rsem_call_2(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    inp = []
    for s in ("SAMPLE_001", "SAMPLE_002"):
        inp += [fix / f"{s}.rsem.isoforms.results", fix / f"{s}.rsem.genes.results", fix / f"{s}.rsem.cnt"]
    p = run_r(repo_root / WORKER, ["--step", "rsem_call_2", "--cwd", tmp_path, "--name", "rsem", "--input", *inp])
    assert p.returncode == 0, p.stdout + p.stderr
    for f in ("rsem_transcripts_expected_count", "rsem_transcripts_tpm", "rsem_transcripts_fpkm",
              "rsem_transcripts_isopct", "rsem_genes_expected_count", "rsem_genes_tpm", "rsem_genes_fpkm"):
        _assert_num_match(gzip.open(tmp_path / f"rsem.{f}.txt.gz", "rt").read(),
                          gzip.open(fix / "expected" / f"rsem.{f}.txt.gz", "rt").read(), label_cols=2)
    # .cnt metrics: compare every column except the machine-dependent File path
    got = _rows((tmp_path / "rsem.rsem.aggregated_quality.metrics.tsv").read_text())
    exp = _rows((fix / "expected" / "rsem.rsem.aggregated_quality.metrics.tsv").read_text())
    drop = exp[0].index("File")
    keep = lambda row: [c for i, c in enumerate(row) if i != drop]
    assert keep(got[0]) == keep(exp[0])                      # header exact
    for rg, re_ in zip(got[1:], exp[1:]):
        assert rg[0] == re_[0]                               # Sample exact
        assert [float(c) for c in keep(rg)[1:]] == [float(c) for c in keep(re_)[1:]]


def test_star_align_3(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    inp = []
    for s in ("SAMPLE_001", "SAMPLE_002"):
        coord = f"{s}.Aligned.sortedByCoord.out_nowasp_noqc.md.bam"
        bw = f"{s}.Aligned.sortedByCoord.out_nowasp_noqc.md.bw"
        inp += ["d0", "d1", coord, "d3", bw, "d5"]
    out = tmp_path / "bam_file_list"
    p = run_r(repo_root / WORKER, ["--step", "star_align_3", "--output", out, "--input", *inp,
                                   "--sample-id", "SAMPLE_001", "SAMPLE_002", "--strand", "rf", "rf",
                                   "--var-vcf-file", ""])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_text() == (fix / "expected" / "star.bam_file_list").read_text()


def test_detect_strand(run_r, repo_root):
    fix = repo_root / FIX
    p = run_r(repo_root / WORKER, ["--step", "detect_strand", "--input", fix / "SAMPLE_001.ReadsPerGene.out.tab"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.strip() == (fix / "expected" / "SAMPLE_001.strand.txt").read_text().strip()


def test_fastp_manifest(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    trimmed = [f"out/{s}_{r}.fastq.trimmed.fq.gz" for s in ("SAMPLE_001", "SAMPLE_002") for r in ("R1", "R2")]
    out = tmp_path / "fastq.list.trimmed.txt"
    p = run_r(repo_root / WORKER, ["--step", "fastp_manifest", "--sample-list", fix / "fastq.list.txt",
                                   "--is-paired-end", 1, "--output", out, "--input", *trimmed])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_text() == (fix / "expected" / "fastq.list.trimmed.txt").read_text()


def test_aggregate_picard_qc(run_r, repo_root, tmp_path):
    # inputs are the MWE's per-sample Picard metrics (post-STAR); readPicard drops
    # the File column so the aggregate is path-independent. Row order follows the
    # filesystem find() order, so compare header exact + data rows as a set.
    fix = repo_root / FIX
    out = tmp_path / "agg.tsv"
    p = run_r(repo_root / WORKER, ["--step", "aggregate_picard_qc",
                                   "--input-dir", fix / "picard_metrics",
                                   "--output", out, "--is-paired-end", 1])
    assert p.returncode == 0, p.stdout + p.stderr
    got = _rows(out.read_text())
    exp = _rows((fix / "expected_picard_aggregated.metrics.tsv").read_text())
    assert got[0] == exp[0]                       # header exact
    assert sorted(got[1:]) == sorted(exp[1:])     # per-sample rows, order-independent


def test_ribosomal_intervals(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    # the R step writes the BAM header (via Rsamtools) then appends the rRNA intervals
    out = tmp_path / "ri.txt"
    p = run_r(repo_root / WORKER, ["--step", "ribosomal_intervals", "--bam", fix / "header.bam",
                                   "--gtf", fix / "rrna_subset.gtf", "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_text() == (fix / "expected" / "rrna_intervals_full.txt").read_text()


def test_filter_reads(run_sh, repo_root, tmp_path):
    # The `unique` branch is samtools' MAPQ filter: `samtools view -h -q MAPQ |
    # samtools view -b`. Assert the wrapper's filtered output == an independent
    # `samtools view -q MAPQ` of the same input (deterministic, tool-for-tool).
    fix = repo_root / FIX / "bam"
    out_cord = tmp_path / "cord.filtered.bam"
    out_trans = tmp_path / "trans.filtered.bam"
    p = run_sh(repo_root / SH, [
        "filter_reads", "--cwd", tmp_path,
        "--input-cord-bam", fix / "SAMPLE_001.cord.bam",
        "--input-trans-bam", fix / "SAMPLE_001.trans.bam",
        "--output-cord-bam", out_cord, "--output-trans-bam", out_trans,
        "--unique", "1", "--wasp", "", "--mapping-quality", "255"])
    assert p.returncode == 0, p.stdout + p.stderr
    for inp, out in ((fix / "SAMPLE_001.cord.bam", out_cord),
                     (fix / "SAMPLE_001.trans.bam", out_trans)):
        assert _sam_body(out) == _sam_body(inp, extra=("-q", "255"))
    # every surviving read is MAPQ>=255, and filtering actually dropped something
    assert len(_sam_body(out_cord)) < len(_sam_body(fix / "SAMPLE_001.cord.bam"))
    assert all(int(r.split("\t")[4]) >= 255 for r in _sam_body(out_cord))


def test_filter_reads_no_filter_copies(run_sh, repo_root, tmp_path):
    # default branch (unique="" wasp="") is a passthrough `cp` — output BAM must
    # carry every input read unchanged.
    fix = repo_root / FIX / "bam"
    out_cord = tmp_path / "cord.bam"
    out_trans = tmp_path / "trans.bam"
    p = run_sh(repo_root / SH, [
        "filter_reads", "--cwd", tmp_path,
        "--input-cord-bam", fix / "SAMPLE_001.cord.bam",
        "--input-trans-bam", fix / "SAMPLE_001.trans.bam",
        "--output-cord-bam", out_cord, "--output-trans-bam", out_trans,
        "--unique", "", "--wasp", "", "--mapping-quality", "255"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert _sam_body(out_cord) == _sam_body(fix / "SAMPLE_001.cord.bam")
    assert _sam_body(out_trans) == _sam_body(fix / "SAMPLE_001.trans.bam")


def test_bam_to_fastq(run_sh, repo_root, tmp_path):
    # name-sort (samtools) then convert to paired fastq (bedtools bamtofastq).
    # Deterministic: every input read becomes one fastq record, split evenly R1/R2.
    fix = repo_root / FIX / "bam"
    bam = fix / "SAMPLE_001.cord.bam"
    fq1, fq2 = tmp_path / "reads.1.fastq", tmp_path / "reads.2.fastq"
    p = run_sh(repo_root / SH, [
        "bam_to_fastq", "--cwd", tmp_path, "--input-bam", bam,
        "--sorted-bam", tmp_path / "sorted.bam",
        "--output-fastq1", fq1, "--output-fastq2", fq2])
    assert p.returncode == 0, p.stdout + p.stderr
    r1 = [ln[1:] for i, ln in enumerate(fq1.read_text().splitlines()) if i % 4 == 0]
    r2 = [ln[1:] for i, ln in enumerate(fq2.read_text().splitlines()) if i % 4 == 0]
    # bedtools emits balanced R1/R2 records (one complete pair each); nothing is
    # invented and nothing exceeds the input; every emitted read exists in the BAM.
    names = {ln.split("\t")[0] for ln in _sam_body(bam)}
    assert len(r1) == len(r2) > 0
    assert len(r1) + len(r2) <= len(_sam_body(bam))
    assert {n.split("/")[0] for n in r1 + r2} <= names


def test_fastqc(run_sh, repo_root, tmp_path):
    # --input-fastq mode: run FastQC on one fastq and stage its .html/.zip aliases.
    fix = repo_root / FIX
    fq = fix / "fastq" / "SAMPLE_001_R1.fastq.gz"
    p = run_sh(repo_root / SH, [
        "fastqc", "--cwd", tmp_path, "--sample-list", fix / "fastq.list.txt",
        "--data-dir", fix / "fastq", "--input-fastq", fq, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    # FastQC's real output + the CWD-level aliases the wrapper links
    assert (tmp_path / "fastqc_raw" / "SAMPLE_001_R1_fastqc.zip").exists()
    assert (tmp_path / "fastqc_raw" / "SAMPLE_001_R1_fastqc.html").exists()
    assert (tmp_path / "SAMPLE_001_R1_fastqc.zip").exists()   # aliased into cwd
    # the zip is a real FastQC archive (carries fastqc_data.txt)
    import zipfile
    with zipfile.ZipFile(tmp_path / "fastqc_raw" / "SAMPLE_001_R1_fastqc.zip") as z:
        assert any(n.endswith("fastqc_data.txt") for n in z.namelist())


def test_multiqc_report(run_sh, repo_root, tmp_path):
    # MultiQC aggregates a directory of tool outputs into one HTML report. Feed it
    # the committed Picard metrics (its picard module recognises them).
    fix = repo_root / FIX
    out = tmp_path / "SAMPLE.multiqc_report.html"
    p = run_sh(repo_root / SH, [
        "multiqc_report", "--cwd", tmp_path, "--input-dir", fix / "picard_metrics",
        "--output-report", out, "--multiqc-config", tmp_path / "cfg.yml"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()                                       # the report was written
    assert (tmp_path / "cfg.yml").exists()                    # config emitted by wrapper


def _fq_reads(path):
    return sum(1 for _ in gzip.open(path, "rt")) // 4


def test_fastp_trim_adaptor(run_sh, repo_root, tmp_path):
    # paired-end fastp adaptor trimming; adapter-fasta "." -> auto-detect mode.
    fix = repo_root / FIX / "fastq"
    o1, o2 = tmp_path / "S1_R1.trimmed.fq.gz", tmp_path / "S1_R2.trimmed.fq.gz"
    p = run_sh(repo_root / SH, [
        "fastp_trim_adaptor", "--cwd", tmp_path, "--paired-end", "1",
        "--input-fastq1", fix / "SAMPLE_001_R1.fastq.gz",
        "--input-fastq2", fix / "SAMPLE_001_R2.fastq.gz",
        "--output-fastq1", o1, "--output-fastq2", o2, "--adapter-fasta", ".",
        "--min-len", "15", "--window-size", "4", "--required-quality", "20",
        "--leading", "20", "--trailing", "20", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert _fq_reads(o1) == _fq_reads(o2) > 0            # balanced paired survivors
    assert (tmp_path / "S1_R1.trimmed.fq.html").exists()  # fastp report siblings
    assert (tmp_path / "S1_R1.trimmed.fq.json").exists()


def test_trimmomatic_trim_adaptor(run_sh, repo_root, tmp_path):
    # paired-end trimmomatic ILLUMINACLIP; the 4 paired/unpaired outputs are written
    # and the two paired streams stay balanced (one surviving pair each).
    fix = repo_root / FIX
    fq = fix / "fastq"
    p1, u1 = tmp_path / "p1.fq.gz", tmp_path / "u1.fq.gz"
    p2, u2 = tmp_path / "p2.fq.gz", tmp_path / "u2.fq.gz"
    p = run_sh(repo_root / SH, [
        "trimmomatic_trim_adaptor", "--cwd", tmp_path, "--paired-end", "1",
        "--java-mem", "4g", "--input-fastq1", fq / "SAMPLE_001_R1.fastq.gz",
        "--input-fastq2", fq / "SAMPLE_001_R2.fastq.gz",
        "--output-paired1", p1, "--output-unpaired1", u1,
        "--output-paired2", p2, "--output-unpaired2", u2,
        "--adapter-fasta", fix / "reference" / "adapters.fa",
        "--seed-mismatches", "2", "--palindrome-clip-threshold", "30",
        "--simple-clip-threshold", "10", "--leading", "3", "--trailing", "3",
        "--window-size", "4", "--required-quality", "20", "--min-len", "36"])
    assert p.returncode == 0, p.stdout + p.stderr
    for f in (p1, u1, p2, u2):
        assert f.exists()
    assert _fq_reads(p1) == _fq_reads(p2) > 0            # both-surviving pairs balanced


def test_picard_qc(run_sh, repo_root, tmp_path):
    # Picard CollectMultipleMetrics + MarkDuplicates + CollectRnaSeqMetrics +
    # bamCoverage over the committed chr22-window BAM/reference (STAR not involved).
    fix = repo_root / FIX
    ref = fix / "reference"
    bam = fix / "bam" / "SAMPLE_001.cord.bam"
    t = tmp_path
    p = run_sh(repo_root / SH, [
        "picard_qc_star_align_2", "--cwd", t,
        "--gtf", ref / "chr22.window.collapsed.gtf",
        "--ref-flat", ref / "chr22.window.refFlat.txt",
        "--reference-fasta", ref / "chr22.window.fa.gz",
        "--java-mem", "4g", "--optical-distance", "100",
        "--input-bam", bam, "--sample-id", "SAMPLE_001", "--strand", "rf",
        "--picard-metrics", t / "SAMPLE_001.alignment_summary_metrics",
        "--picard-rna-metrics", t / "SAMPLE_001.rna_metrics",
        "--md-bam", t / "SAMPLE_001.md.bam", "--md-metrics", t / "SAMPLE_001.md.metrics",
        "--bigwig", t / "SAMPLE_001.md.bw", "--output-summary", t / "SAMPLE_001.bam_file_meta",
        "--var-vcf-file", "", "--zap-raw-bam", "False"])
    assert p.returncode == 0, p.stdout + p.stderr
    # picard metrics files carry their METRICS CLASS provenance
    assert "picard.analysis.AlignmentSummaryMetrics" in \
        (t / "SAMPLE_001.alignment_summary_metrics").read_text()
    assert "picard.analysis.RnaSeqMetrics" in (t / "SAMPLE_001.rna_metrics").read_text()
    # MarkDuplicates flags but never drops reads: md.bam keeps every input read
    assert len(_sam_body(t / "SAMPLE_001.md.bam")) == len(_sam_body(bam))
    assert (t / "SAMPLE_001.md.bw").stat().st_size > 0            # bamCoverage bigwig
    # bam_file_meta is a deterministic text derivation of the sample/output names
    assert _rows((t / "SAMPLE_001.bam_file_meta").read_text()) == [
        ["sample_id", "strand", "coord_bam_list", "BW_list", "SJ_list", "trans_bam_list"],
        ["SAMPLE_001", "rf", "SAMPLE_001.md.bam", "SAMPLE_001.md.bw",
         "SAMPLE_001.SJ.out.tab", "SAMPLE_001.toTranscriptome.out.bam"]]


def test_rnaseqc_call(run_sh, repo_root, tmp_path):
    # RNA-SeQC over the committed chr22-window BAM + collapsed GTF, then the
    # wrapper renames the per-BAM outputs to the {sample}.rnaseqc.* contract.
    fix = repo_root / FIX
    ref = fix / "reference"
    p = run_sh(repo_root / SH, [
        "rnaseqc_call", "--cwd", tmp_path, "--sample-list", fix / "fastq.list.txt",
        "--data-dir", ref, "--gtf", ref / "chr22.window.collapsed.gtf",
        "--reference-fasta", ref / "chr22.window.fa.gz",
        "--input-bam", fix / "bam" / "SAMPLE_001.cord.bam", "--sample-id", "SAMPLE_001",
        "--strand", "rf", "--detection-threshold", "5", "--mapping-quality", "255",
        "--paired-end", "1", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    for suf in ("gene_tpm.gct.gz", "gene_reads.gct.gz", "exon_reads.gct.gz"):
        f = tmp_path / f"SAMPLE_001.rnaseqc.{suf}"
        assert f.exists()
        assert gzip.open(f, "rt").readline().startswith("#1.2")   # valid GCT header
    assert (tmp_path / "SAMPLE_001.rnaseqc.metrics.tsv").exists()


def test_rnaseqc_merge(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    inp = []
    for s in ("SAMPLE_001", "SAMPLE_002"):
        inp += [fix / f"{s}.rnaseqc.gene_tpm.gct.gz", fix / f"{s}.rnaseqc.gene_reads.gct.gz",
                fix / f"{s}.rnaseqc.exon_reads.gct.gz", fix / f"{s}.rnaseqc.metrics.tsv"]
    p = run_r(repo_root / WORKER, ["--step", "rnaseqc_merge", "--cwd", tmp_path, "--name", "rnaseqc", "--input", *inp])
    assert p.returncode == 0, p.stdout + p.stderr
    for f in ("gene_tpm.gct.gz", "gene_readsCount.gct.gz", "exon_readsCount.gct.gz"):
        assert gzip.open(tmp_path / f"rnaseqc.rnaseqc.{f}", "rt").read() == \
            gzip.open(fix / "expected" / f"rnaseqc.rnaseqc.{f}", "rt").read()
    assert (tmp_path / "rnaseqc.rnaseqc.metrics.tsv").read_text() == \
        (fix / "expected" / "rnaseqc.rnaseqc.metrics.tsv").read_text()
