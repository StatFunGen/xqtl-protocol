"""Notebook tier: phenotype_formatting.ipynb steps driven end-to-end via SoS on the
committed chr22 fixtures (tests/fixtures/phenotype_formatting). The workers are the
ported R script code/script/data_preprocessing/phenotype/phenotype_formatting.R.
"""
from __future__ import annotations

import shutil

NB = "pipeline/phenotype_formatting.ipynb"
FX = "tests/fixtures/phenotype_formatting"
PHENO = "protocol_example.rnaseq.bed.bed.gz"


def _base(repo_root, out):
    return dict(cwd=out, modular_script_dir=repo_root / "code/script")


def test_phenotype_by_chrom(run_sos, repo_root, tmp_path):
    # The MWE-demonstrated step: runs phenotype_by_chrom_1 then _2.
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "phenotype_by_chrom",
                {**_base(repo_root, out),
                 "phenoFile": repo_root / FX / PHENO,
                 "name": "protocol_example", "chrom": "chr22"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.chr22.bed.gz").exists()
    assert (out / "protocol_example.chr22.bed.gz.tbi").exists()
    assert (out / "protocol_example.phenotype_by_chrom_files.txt").exists()
    assert (out / "protocol_example.phenotype_by_chrom_files.region_list.txt").exists()


def test_phenotype_by_region(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "phenotype_by_region",
                {**_base(repo_root, out),
                 "phenoFile": repo_root / FX / PHENO,
                 "name": "protocol_example",
                 "region_list": repo_root / FX / "regions.txt"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    per_region = list(out.glob("regions_phenotype_by_region/protocol_example.*.bed.gz"))
    assert per_region, p.stdout
    assert (out / "protocol_example.phenotype_by_region_files.txt").exists()


def test_gct_extract_samples(run_sos, repo_root, tmp_path):
    # Output is written next to the input GCT, so stage the fixture in a temp dir.
    work = tmp_path / "work"; work.mkdir()
    gct = work / "protocol_example.tpm.gct.gz"
    shutil.copy(repo_root / FX / "protocol_example.tpm.gct.gz", gct)
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "gct_extract_samples",
                {**_base(repo_root, out),
                 "phenoFile": gct,
                 "keep_samples": repo_root / FX / "keep_samples.txt"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (work / "protocol_example.tpm.sample_matched.gct.gz").exists()


def test_phenotype_annotate_by_tad(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "phenotype_annotate_by_tad",
                {**_base(repo_root, out),
                 "phenoFile": repo_root / FX / PHENO,
                 "TAD_list": repo_root / FX / "tad_list.txt",
                 "phenotype_per_tad": 2},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert list(out.glob("*_pheno_per_region.region_list")), p.stdout


def test_bam_subsetting(run_sos, repo_root, tmp_path):
    # The committed BAM fixture is chr22:16-17M (44 KB); its .bai sits alongside it.
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "bam_subsetting",
                {**_base(repo_root, out),
                 "phenoFile": repo_root / FX / "protocol_example.chr22_16M_17M.bam",
                 "region": "chr22:16200000-16800000"},
                cwd=repo_root, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.chr22_16M_17M.subsetted.bam").exists()


def test_phenotype_by_chrom_gct(run_sos, repo_root, tmp_path):
    # GCT-style by-chrom extraction: pull one chromosome from the tabix BED and
    # blank the 3 coordinate columns (legacy awk $1=$2=$3="").
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "phenotype_by_chrom_gct",
                {**_base(repo_root, out),
                 "phenoFile": repo_root / FX / PHENO,
                 "name": "protocol_example", "chrom": "chr22"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    gct = out / "protocol_example.chr22.gct"
    assert gct.exists() and gct.stat().st_size > 0
