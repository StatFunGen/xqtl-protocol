"""Notebook tier: sldsc_enrichment.ipynb.

The R analysis blocks of make_annotation_files_ldscore are ported to
code/script/enrichment/make_annotation.R (Step A: write .annot.gz; Step D: write
.l2.M). Only the polyfun LD-score computation (Step C: ldsc.py / compute_ldscores.py)
stays external. Those two worker steps are tested here at worker level against the
MWE references. munge_sumstats_polyfun / get_heritability remain external-tool
orchestration (polyfun over the ~350M panel), OUT of CI scope. `postprocess` /
`meta_subset` (pecotmr wrappers) consume polyfun's small per-trait OUTPUTS.
"""
from __future__ import annotations

import gzip

import pytest

MK = "code/script/enrichment/make_annotation.R"
FX = "tests/fixtures/sldsc_enrichment"


def test_make_annotation_annot(run_r, repo_root, tmp_path):
    """Step A: reference .annot + target variant list -> per-SNP ANNOT (binary), byte-exact."""
    p = run_r(repo_root / MK,
              ["--step", "annot", "--targets", repo_root / FX / "target.tsv",
               "--reference-anno", repo_root / FX / "reference.2.annot.gz",
               "--emit-single", "--annotation-name", "protocol_example",
               "--cwd", tmp_path, "--chrom", 2])
    assert p.returncode == 0, p.stdout + p.stderr
    got = gzip.open(tmp_path / "protocol_example_single_1/protocol_example_single_1.2.annot.gz", "rt").read()
    exp = gzip.open(repo_root / FX / "expected_single_1.2.annot.gz", "rt").read()
    assert got == exp


def test_make_annotation_mfiles(run_r, repo_root, tmp_path):
    """Step D: .annot + polyfun ldscore parquet -> .l2.M (sum of ANNOT over ldscore SNPs)."""
    d = tmp_path / "protocol_example_single_1"; d.mkdir()
    (d / "protocol_example_single_1.2.annot.gz").write_bytes((repo_root / FX / "expected_single_1.2.annot.gz").read_bytes())
    (d / "protocol_example_single_1.2.l2.ldscore.parquet").write_bytes((repo_root / FX / "single_1.2.l2.ldscore.parquet").read_bytes())
    p = run_r(repo_root / MK,
              ["--step", "mfiles", "--annotation-name", "protocol_example", "--cwd", tmp_path,
               "--chrom", 2, "--emit-single", "--n-targets", 1, "--ldscore-ext", "l2.ldscore.parquet"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert (d / "protocol_example_single_1.2.l2.M").read_text() == \
        (repo_root / FX / "expected_single_1.2.l2.M").read_text()


def test_postprocess_and_meta_subset(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures" / "sldsc_enrichment"
    cwd = tmp_path / "sldsc"

    # (1) postprocess: bundle polyfun's per-trait outputs into a SldscData and run
    # the DerSimonian-Laird random-effects meta -> per_trait / meta / params.
    p = run_sos(
        repo_root / "pipeline/sldsc_enrichment.ipynb", "postprocess",
        {
            "cwd": cwd,
            "annotation_name": "protocol_example",
            "traits_file": fx / "sumstats_test_all.txt",
            "heritability_cwd": fx / "sldsc_heritability",
            "target_categories": "ANNOT_0",
            "target_categories_label": "protocol_example_annotation",
            "target_anno_dir": fx / "target_anno",
            "maf_cutoff": 0,
            "polyfun_path": ".",       # unused by postprocess; global param must be set
            "python_exec": "python",
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    pp = cwd / "protocol_example.sldsc_postprocess.rds"
    assert pp.exists(), p.stdout
    info = read_rds(pp)
    assert info["class"] == "list"
    assert {"per_trait", "meta", "params"}.issubset(set(info["names"])), info

    # (2) meta_subset: re-meta a trait subset off the postprocess RDS (no rerun).
    p = run_sos(
        repo_root / "pipeline/sldsc_enrichment.ipynb", "meta_subset",
        {
            "cwd": cwd,
            "annotation_name": "protocol_example",
            "postprocess_rds": pp,
            "subset_traits_file": fx / "sumstats_test_category1.txt",
            "subset_name": "category1",
            "target_categories": "ANNOT_0",
            "polyfun_path": ".",
            "python_exec": "python",
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    meta = cwd / "protocol_example.category1.meta.rds"
    assert meta.exists(), p.stdout
    minfo = read_rds(meta)
    assert minfo["class"] == "list"
    assert "enrichment" in minfo["names"], minfo
