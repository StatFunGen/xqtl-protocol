"""Notebook tier: METAL.ipynb — GWAS meta-analysis post-processing.

The METAL binary runs in the notebook (METAL_1/METAL_2); the extracted worker
code/script/multivariate_genome/metal.R does the post-processing that used to be inline
Python + R:
  --step reformat : raw METAL output -> reformatted .METAL.txt + a GWAS-VCF (.vcf.bgz)
  --step recipe   : build {name}.METAL_list.txt (#chr -> per-chrom sumstat file)

Fixtures (tests/fixtures/metal, from the MWE): a raw METAL output (protocol_example.22.1
.METAL.txt) for the reformat worker test, a sumstat_list, and a small metal-format
sumstats table (variant_id/alt/ref/beta/se/pval/N) that drives the end-to-end SoS run
through the real `metal` binary.
"""
from __future__ import annotations

import gzip

R = "code/script/multivariate_genome/metal.R"
NB = "pipeline/METAL.ipynb"
FIX = "tests/fixtures/metal"
MSD = "code/script"
RAW = "protocol_example.22.1.METAL.txt"


def test_reformat(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    sumstat = tmp_path / "out.METAL.txt"
    vcf = tmp_path / "out.METAL.vcf.bgz"
    p = run_r(repo_root / R, ["--step", "reformat", "--input", fix / RAW,
                              "--output-sumstat", sumstat, "--output-vcf", vcf,
                              "--name", "protocol_example"])
    assert p.returncode == 0, p.stdout + p.stderr
    n_in = len((fix / RAW).read_text().splitlines()) - 1
    lines = sumstat.read_text().splitlines()
    assert lines[0].split("\t") == ["variant_id", "alt", "ref", "beta", "se",
                                    "pval", "Direction", "pos", "chrom"]
    assert len(lines) - 1 == n_in
    # pos/chrom parsed from variant_id (chr:pos_ref_alt) on the first data row
    first = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
    assert first["chrom"] == first["variant_id"].split(":")[0]
    assert first["pos"] == first["variant_id"].split(":")[1].split("_")[0]
    # GWAS-VCF: one record per input variant, VCFv4.2
    assert vcf.exists() and (tmp_path / (vcf.name + ".tbi")).exists()
    with gzip.open(vcf, "rt") as fh:
        vlines = fh.readlines()
    assert any(l.startswith("##fileformat=VCFv4.2") for l in vlines)
    assert sum(1 for l in vlines if not l.startswith("#")) == n_in


def test_recipe(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    out = tmp_path / "recipe.txt"
    p = run_r(repo_root / R, ["--step", "recipe",
                              "--sumstat-list", fix / "protocol_example.sumstat_list.tsv",
                              "--name", "protocol_example", "--output", out,
                              tmp_path / "protocol_example.22.METAL.txt"])
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [ln.split("\t") for ln in out.read_text().splitlines()]
    assert rows[0] == ["#chr", "protocol_example"]
    assert rows[1][0] == "22" and rows[1][1].endswith("protocol_example.22.METAL.txt")


def test_metal_via_sos(run_sos, repo_root, tmp_path):
    # End-to-end SoS: METAL_1 (script) -> METAL_2 (metal binary) -> METAL_3 (reformat +
    # VCF) -> METAL_4 (recipe). sumstat_list is generated pointing at the committed
    # metal-format sumstats (absolute path).
    fix = repo_root / FIX
    slist = tmp_path / "sumstat_list.tsv"
    slist.write_text("#chr\tprotocol_example\n22\t{}\n".format(
        fix / "protocol_example.gwas_sumstats.chr22.tsv"))
    p = run_sos(repo_root / NB, "METAL", {
        "sumstat_list_path": slist, "cwd": tmp_path,
        "modular_script_dir": repo_root / MSD}, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "protocol_example.22.METAL.vcf.bgz").exists()
    assert (tmp_path / "protocol_example.22.METAL.txt").exists()
    assert (tmp_path / "protocol_example.METAL_list.txt").exists()
