"""Notebook tier: genotype_formatting.ipynb — the Python worker ported to R.

genotype_formatting.py -> genotype_formatting.R (steps ld_by_region_plink_1,
write_data_list, vcf_gz_summary, file_size_summary). The LD output switched from a
numpy .npz to an RDS (nothing in-repo read the .npz).

write_data_list is deterministic (byte-validated against the retired Python here as an
inline expectation); ld_by_region runs plink and is checked for a square LD matrix +
matching variant IDs (probed via a small Rscript).
"""
from __future__ import annotations

import os
import subprocess

WORKER = "code/script/data_preprocessing/genotype/genotype_formatting.R"
SH = "code/script/data_preprocessing/genotype/genotype_formatting.sh"
BED = "tests/fixtures/ld_prune_reference/genotype/protocol_example.ld_genotype.chr22"
VCF = "tests/fixtures/vcf_qc/protocol_example.genotype.chr22.vcf.gz"
CHR21 = "tests/fixtures/grm/geno/chr21"
CHR22 = "tests/fixtures/grm/geno/chr22"


def _lines(path):
    return sum(1 for _ in open(path))


def _vcf_records(path):
    out = subprocess.run(["bcftools", "view", "-H", str(path)],
                         capture_output=True, text=True, check=True)
    return sum(1 for ln in out.stdout.splitlines() if ln)


def _rscript(expr):
    r = subprocess.run([os.environ.get("XQTL_RSCRIPT", "Rscript"), "-e", expr],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def test_write_data_list(run_r, repo_root, tmp_path):
    f1 = tmp_path / "s.chr1.bed"; f1.write_text("x")
    f2 = tmp_path / "s.chr2.bed"; f2.write_text("y")
    f3 = tmp_path / "s.chr3.bed"; f3.write_text("")            # empty -> excluded
    out = tmp_path / "list.tsv"
    files = "::".join(str(x) for x in (f1, f2, f3))
    p = run_r(repo_root / WORKER, ["--step", "write_data_list", "--data-files", files,
                                   "--ext", "bed", "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    lines = out.read_text().splitlines()
    assert lines[0] == "#id\t#path"
    assert lines[1] == f"chr1\t{f1}"
    assert lines[2] == f"chr2\t{f2}"
    assert len(lines) == 3                                     # empty chr3 dropped


def test_ld_by_region(run_r, repo_root, tmp_path):
    bed = repo_root / f"{BED}.bed"
    out = tmp_path / "region.float16.rds"
    p = run_r(repo_root / WORKER, [
        "--step", "ld_by_region_plink_1", "--genoFile", bed, "--region-chrom", "22",
        "--region-start", "16050000", "--region-end", "16150000",
        "--output", out, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists()
    # RDS holds a square LD matrix whose dimension matches the variant-ID vector,
    # and the IDs are the region's .bim variant IDs.
    probe = _rscript(
        f'r <- readRDS("{out}"); '
        'stopifnot(is.matrix(r$ld), nrow(r$ld) == ncol(r$ld), '
        'length(r$variant_ids) == nrow(r$ld)); '
        'cat(nrow(r$ld), r$variant_ids[1])')
    n, first_id = probe.split()
    assert int(n) > 0 and first_id.startswith("chr22:")


# ── the .sh format-conversion / merge steps ──────────────────────────────────

def test_gf_plink_to_vcf(run_sh, repo_root, tmp_path):
    out = tmp_path / "out.vcf.gz"
    p = run_sh(repo_root / SH, ["plink_to_vcf_1", "--genoFile", repo_root / f"{BED}.bed",
                                "--output", out, "--mem", "4G", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert _vcf_records(out) == _lines(repo_root / f"{BED}.bim")     # every variant kept


def test_gf_vcf_to_plink(run_sh, repo_root, tmp_path):
    p = run_sh(repo_root / SH, ["vcf_to_plink", "--cwd", tmp_path,
                                "--genoFile", repo_root / VCF, "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    beds = list(tmp_path.glob("*.bed"))
    assert len(beds) == 1 and beds[0].with_suffix(".fam").exists()


def test_gf_genotype_by_region(run_sh, repo_root, tmp_path):
    out = tmp_path / "region.bed"
    p = run_sh(repo_root / SH, ["genotype_by_region_1", "--genoFile", repo_root / f"{BED}.bed",
                                "--output", out, "--region-chrom", "22", "--region-start", "16000000",
                                "--region-end", "17000000", "--window", "0", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    n_sub, n_all = _lines(tmp_path / "region.bim"), _lines(repo_root / f"{BED}.bim")
    assert 0 < n_sub < n_all                                        # region subset


def test_gf_genotype_by_chrom(run_sh, repo_root, tmp_path):
    out = tmp_path / "bychrom.bed"
    p = run_sh(repo_root / SH, ["genotype_by_chrom", "--cwd", tmp_path,
                                "--genoFile", repo_root / f"{BED}.bed", "--name", "test",
                                "--output", out, "--chrom", "22", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and out.with_suffix(".bim").exists()


def test_gf_merge_plink(run_sh, repo_root, tmp_path):
    plist = tmp_path / "plist.txt"
    plist.write_text(f"{repo_root / CHR21}\n{repo_root / CHR22}\n")
    p = run_sh(repo_root / SH, ["merge_plink", "--cwd", tmp_path, "--genoFile", plist,
                                "--name", "merged", "--mem", "4G", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "merged.bim").exists()
    assert _lines(tmp_path / "merged.bim") == _lines(repo_root / f"{CHR21}.bim") + _lines(repo_root / f"{CHR22}.bim")


def test_gf_merge_vcf(run_sh, repo_root, tmp_path):
    # build two distinct-chromosome VCFs from the committed beds, then concat them
    vcfs = []
    for src in (CHR21, CHR22):
        out = tmp_path / f"{src.split('/')[-1]}"
        subprocess.run(["plink2", "--bfile", str(repo_root / src), "--recode", "vcf-iid",
                        "--out", str(out), "--silent"], check=True, capture_output=True)
        subprocess.run(["bgzip", "-f", f"{out}.vcf"], check=True)
        vcfs.append(f"{out}.vcf.gz")
    merged = tmp_path / "merged.vcf.gz"
    p = run_sh(repo_root / SH, ["merge_vcf", "--genoFile", " ".join(vcfs), "--output", merged])
    assert p.returncode == 0, p.stdout + p.stderr
    assert _vcf_records(merged) == _vcf_records(vcfs[0]) + _vcf_records(vcfs[1])
