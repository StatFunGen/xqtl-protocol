"""Notebook tier: VCF_QC.ipynb — the two steps rewired to VCF_QC.sh (rename_chrs,
dbsnp_annotate). The qc/qc_2/qc_3 steps already delegated to VCF_QC.sh.

Fixtures (tests/fixtures/vcf_qc), from the MWE:
  protocol_example.genotype.chr22.vcf.gz   chr-prefixed 60-sample chr22 VCF
  numeric_chr22.vcf.gz                      same, chrom renamed 'chr22' -> '22' (for rename_chrs)

dbsnp_annotate is a deterministic `bcftools query + awk` transform, so its output is
byte-checked against an independent bcftools computation. rename_chrs is checked for
the 1->chr1 renaming.
"""
from __future__ import annotations

import gzip
import shutil
import subprocess

NB = "pipeline/VCF_QC.ipynb"
SH = "code/script/data_preprocessing/genotype/VCF_QC.sh"
FIX = "tests/fixtures/vcf_qc"
REF = f"{FIX}/reference/chr22.win48.fa.gz"
SUB = f"{FIX}/genotype.chr22_48M.vcf.gz"          # variants in the reference's real window
DBSNP = f"{FIX}/genotype.chr22_48M.variants.gz"


def _vcf_records(path):
    out = subprocess.run(["bcftools", "view", "-H", str(path)],
                         capture_output=True, text=True, check=True)
    return sum(1 for ln in out.stdout.splitlines() if ln)


def _run_qc1(run_sh, repo_root, tmp_path):
    """qc (multiallelic split + left-normalization vs the reference) -> normalized VCF."""
    out = tmp_path / "qc1.vcf.gz"
    p = run_sh(repo_root / SH, ["qc", "--cwd", tmp_path, "--genoFile", repo_root / SUB,
                                "--output", out, "--dbsnp-variants", repo_root / DBSNP,
                                "--reference-genome", repo_root / REF, "--bi-allelic", "False",
                                "--snp-only", "False", "--gt-only-vcf-qc", "True", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def _expected_variants(vcf):
    """Replicate _dbsnp_annotate: CHROM, POS, END, ID with END = POS + max(|REF|,|ALT|) - 1."""
    q = subprocess.run(["bcftools", "query", "-f", "%CHROM\t%POS\t%ID\t%REF\t%ALT\n", str(vcf)],
                       capture_output=True, text=True, check=True)
    rows = []
    for ln in q.stdout.splitlines():
        chrom, pos, vid, ref, alt = ln.split("\t")
        end = int(pos) + max(len(ref), len(alt)) - 1
        rows.append(f"{chrom}\t{pos}\t{end}\t{vid}")
    return rows


def test_dbsnp_annotate(run_sos, repo_root, tmp_path):
    vcf = repo_root / FIX / "protocol_example.genotype.chr22.vcf.gz"
    p = run_sos(repo_root / NB, "dbsnp_annotate", {
        "genoFile": vcf, "cwd": tmp_path, "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "protocol_example.genotype.chr22.variants.gz"
    got = gzip.open(out, "rt").read().splitlines()
    assert got == _expected_variants(vcf)                    # byte-exact 4-column table


def test_rename_chrs(run_sos, repo_root, tmp_path):
    # rename_chrs writes next to its input, so copy the numeric-chrom VCF into tmp first
    src = repo_root / FIX / "numeric_chr22.vcf.gz"
    vcf = tmp_path / "numeric_chr22.vcf.gz"
    shutil.copy(src, vcf)
    p = run_sos(repo_root / NB, "rename_chrs", {
        "genoFile": vcf, "cwd": tmp_path, "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "numeric_chr22.add_chr.vcf.gz"
    assert out.exists()
    # every record's chrom is now chr-prefixed (was bare '22')
    chroms = subprocess.run(["bcftools", "query", "-f", "%CHROM\n", str(out)],
                            capture_output=True, text=True, check=True).stdout.split()
    assert chroms and all(c == "chr22" for c in chroms)


def test_qc_normalize(run_sh, repo_root, tmp_path):
    # left-normalization against the N-padded chr22:48-49M reference; a couple of
    # variants drop during multiallelic/dup handling, the rest survive.
    out = _run_qc1(run_sh, repo_root, tmp_path)
    assert out.exists()
    n_in = _vcf_records(repo_root / SUB)
    n_out = _vcf_records(out)
    assert 0 < n_out <= n_in and n_out > n_in - 20


def test_qc_2(run_sh, repo_root, tmp_path):
    # variant-level QC (GT-only mode) runs cleanly on the normalized VCF and writes a valid VCF.
    qc1 = _run_qc1(run_sh, repo_root, tmp_path)
    out = tmp_path / "qc2.vcf.gz"
    p = run_sh(repo_root / SH, ["qc_2", "--cwd", tmp_path, "--genoFile", qc1, "--output", out,
                                "--gt-only-vcf-qc", "True", "--geno-filter", "0.2", "--DP-snp", "10",
                                "--GQ", "20", "--DP-indel", "10", "--AB-snp", "0.15", "--AB-indel", "0.2",
                                "--hwe-filter", "0", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and _vcf_records(out) >= 0            # valid VCF (GT-only toy data may filter all)


def test_qc_3(run_sh, repo_root, tmp_path):
    # per-VCF summary statistics (novel/known variant counts + Ts/Tv) over the normalized VCF.
    qc1 = _run_qc1(run_sh, repo_root, tmp_path)
    outs = {k: tmp_path / f"{k}" for k in
            ("novel.sumstats", "known.sumstats", "novel.tstv", "known.tstv")}
    p = run_sh(repo_root / SH, ["qc_3", "--cwd", tmp_path, "--genoFile", qc1,
                                "--novel-sumstats", outs["novel.sumstats"],
                                "--known-sumstats", outs["known.sumstats"],
                                "--novel-tstv", outs["novel.tstv"], "--known-tstv", outs["known.tstv"],
                                "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    for f in outs.values():
        assert f.exists()
