"""Notebook tier: rss_ld_sketch.ipynb — RSS LD random-projection sketch.

Worker: code/script/reference_data/rss_ld_sketch.R (generate_w / process_block).

Fixtures (tests/fixtures/rss_ld_sketch), from the MWE:
  protocol_example.genotype.chr22.bgz(.tbi)  60-sample chr22 VCF
  protocol_example.ld_blocks.bed             3 blocks (16-20M / 30-34M / 44-48M)
  expected/afreq_deterministic.tsv           MWE reference .afreq, columns 1-6

W is drawn with R's RNG, so the dosage sketch is NOT byte-reproducible against the
numpy-generated MWE. The DETERMINISTIC .afreq columns (CHROM/ID/REF/ALT/ALT_FREQS/OBS_CT)
depend only on the VCF + filtering, so those are byte-validated here; the W-dependent
dosage is checked structurally (field count + [0, 2] range).
"""
from __future__ import annotations

import gzip

WORKER = "code/script/reference_data/rss_ld_sketch.R"
NB = "pipeline/rss_ld_sketch.ipynb"
FIX = "tests/fixtures/rss_ld_sketch"
BLOCKS = [(16000000, 20000000), (30000000, 34000000), (44000000, 48000000)]
B = 50


def _run_blocks(run_r, repo_root, tmp_path):
    """generate_w then process_block over the 3 fixture blocks; return tmp_path."""
    worker = repo_root / WORKER
    vcf = repo_root / FIX / "protocol_example.genotype.chr22.bgz"
    W = tmp_path / "W_B50.rds"
    p = run_r(worker, ["--step", "generate_w", "--n-samples", "60", "--B", B,
                       "--seed", "123", "--output", W])
    assert p.returncode == 0, p.stdout + p.stderr
    for bs, be in BLOCKS:
        p = run_r(worker, ["--step", "process_block", "--vcf", vcf, "--chrom", "chr22",
                           "--block-start", bs, "--block-end", be, "--w-matrix", W,
                           "--cohort-id", "protocol_example.", "--output-dir", tmp_path, "--B", B])
        assert p.returncode == 0, p.stdout + p.stderr
    return tmp_path


def _block_file(tmp_path, bs, be, ext):
    tag = f"chr22_{bs}_{be}"
    return tmp_path / "chr22" / tag / f"protocol_example..{tag}.{ext}"


def test_rss_process_block_afreq_deterministic(run_r, repo_root, tmp_path):
    _run_blocks(run_r, repo_root, tmp_path)
    # concatenate per-block .afreq (header once), keep the 6 deterministic columns
    rows = []
    for i, (bs, be) in enumerate(BLOCKS):
        lines = _block_file(tmp_path, bs, be, "afreq").read_text().splitlines()
        for ln in (lines if i == 0 else lines[1:]):
            rows.append("\t".join(ln.split("\t")[:6]))
    got = "\n".join(rows) + "\n"
    assert got == (repo_root / FIX / "expected" / "afreq_deterministic.tsv").read_text()


def test_rss_process_block_dosage_structural(run_r, repo_root, tmp_path):
    _run_blocks(run_r, repo_root, tmp_path)
    bs, be = BLOCKS[0]
    n_vars = 0
    with gzip.open(_block_file(tmp_path, bs, be, "dosage.gz"), "rt") as fh:
        for ln in fh:
            f = ln.split()
            assert len(f) == 3 + B                         # ID ALT REF + B sketch values
            assert all(0.0 <= float(x) <= 2.0 for x in f[3:])  # plink2-importable dosage range
            n_vars += 1
    assert n_vars > 0
    # .meta records the filter tally and the passing count matches the .map / .afreq
    meta = dict(l.split("=", 1) for l in _block_file(tmp_path, bs, be, "meta").read_text().splitlines())
    assert int(meta["n_passed"]) == n_vars
    assert int(meta["B"]) == B and int(meta["source_n_samples"]) == 60


def test_rss_process_block_via_sos(run_r, run_sos, repo_root, tmp_path):
    # Exercise the SoS wiring: generate W, then run the `process_block` workflow
    # (BED -> for_each blocks -> R worker) and byte-check the deterministic .afreq.
    W = tmp_path / "W_B50.rds"
    p = run_r(repo_root / WORKER, ["--step", "generate_w", "--n-samples", "60",
                                   "--B", B, "--seed", "123", "--output", W])
    assert p.returncode == 0, p.stdout + p.stderr
    p = run_sos(repo_root / NB, "process_block", {
        "ld_block_file": repo_root / FIX / "protocol_example.ld_blocks.bed",
        "chrom": 22, "vcf_base": repo_root / FIX, "vcf_prefix": "protocol_example.genotype.",
        "cohort_id": "protocol_example.", "output_dir": tmp_path, "W_matrix": W, "B": B,
        "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    rows = []
    for i, (bs, be) in enumerate(BLOCKS):
        lines = _block_file(tmp_path, bs, be, "afreq").read_text().splitlines()
        for ln in (lines if i == 0 else lines[1:]):
            rows.append("\t".join(ln.split("\t")[:6]))
    got = "\n".join(rows) + "\n"
    assert got == (repo_root / FIX / "expected" / "afreq_deterministic.tsv").read_text()


def test_rss_merge_chrom(run_r, run_sos, repo_root, tmp_path):
    # process the 3 blocks, then merge them into one per-chromosome pgen (plink2 pmerge)
    # + concatenated afreq + the base-R filter summary (ported off data.table).
    _run_blocks(run_r, repo_root, tmp_path)
    p = run_sos(repo_root / NB, "merge_chrom", {
        "chrom": 22, "output_dir": tmp_path, "cohort_id": "protocol_example.",
        "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    chrom_dir = tmp_path / "chr22"
    assert (chrom_dir / "protocol_example..chr22.pgen").exists()
    afreq = (chrom_dir / "protocol_example..chr22.afreq").read_text().splitlines()
    assert afreq[0].startswith("#CHROM") and len(afreq) - 1 == 673   # all variants merged
