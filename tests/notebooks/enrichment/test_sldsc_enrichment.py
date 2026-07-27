"""Notebook tier: sldsc_enrichment.ipynb.

The R analysis blocks of make_annotation_files_ldscore are ported to
code/script/enrichment/make_annotation.R (Step A: write .annot.gz; Step D: write
.l2.M). `postprocess` / `meta_subset` (pecotmr wrappers) consume polyfun's small
per-trait OUTPUTS.

polyfun CLIs are the packaged **bin** entry points (`munge_polyfun_sumstats.py`,
`compute_ldscores.py`, `ldsc.py`), invoked by bare name on PATH — the notebook no
longer prefixes `python`/`polyfun_path` (those bin scripts are /bin/sh wrappers
that `exec python3 <libexec>/...`, so feeding them to `python` breaks).
`munge_sumstats_polyfun` and the full `make_annotation_files_ldscore` chain (Step A
annot -> Step C polyfun `compute_ldscores.py` -> Step D .l2.M) both run here.
`get_heritability` (`ldsc.py --h2`) runs too but needs the full multi-chromosome LD
panel (baseline + weights + per-chrom target LD, tens of MB) — not committed, so
left untested.
"""
from __future__ import annotations

import gzip
import shutil

import pytest


def _have_polyfun():
    """Whether polyfun's CLIs are on PATH (skip guard)."""
    return shutil.which("munge_polyfun_sumstats.py") is not None

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
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    meta = cwd / "protocol_example.category1.meta.rds"
    assert meta.exists(), p.stdout
    minfo = read_rds(meta)
    assert minfo["class"] == "list"
    assert "enrichment" in minfo["names"], minfo


def test_munge_sumstats_polyfun(run_sos, repo_root, tmp_path):
    """munge_sumstats_polyfun: polyfun munge_polyfun_sumstats.py on a small BOLT-LMM
    subset -> munged .parquet (SNP/CHR/BP/A1/A2/MAF/N/Z). BOLT input needs --n."""
    if not _have_polyfun():
        pytest.skip("polyfun not installed")
    sumstats = tmp_path / "boltlmm_small.sumstats.gz"
    sumstats.write_bytes((repo_root / FX / "boltlmm_small.sumstats.gz").read_bytes())
    p = run_sos(repo_root / "pipeline/sldsc_enrichment.ipynb", "munge_sumstats_polyfun",
                {"sumstats": sumstats, "n": 100000, "cwd": tmp_path, "annotation_name": "test"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "boltlmm_small.sumstats.munged.parquet"
    assert out.exists()
    import pandas as pd
    d = pd.read_parquet(out)
    assert d.shape[0] > 0
    assert {"SNP", "CHR", "BP", "A1", "A2", "Z", "N"}.issubset(set(d.columns))


def test_make_annotation_files_ldscore(run_sos, repo_root, tmp_path):
    """Full make_annotation_files_ldscore chain: Step A make_annotation.R annot ->
    Step C polyfun compute_ldscores.py (via the bin wrapper) -> Step D .l2.M.
    The chr2 LD-score parquet must reproduce the committed reference (compute_ldscores
    is deterministic); the annot.gz + .l2.M must byte-match their references."""
    if not _have_polyfun():
        pytest.skip("polyfun not installed")
    out = tmp_path / "out"
    p = run_sos(repo_root / "pipeline/sldsc_enrichment.ipynb", "make_annotation_files_ldscore",
                {"cwd": out, "annotation_name": "protocol_example",
                 "annotation_file": repo_root / FX / "target.tsv",
                 "reference_anno_file": repo_root / FX / "reference.2.annot.gz",
                 "genome_ref_file": repo_root / FX / "reference.2.bed",
                 "chromosome": 2, "ld_wind_cm": 1.0,
                 "modular_script_dir": repo_root / "code/script"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    d = out / "protocol_example_single_1"
    stem = "protocol_example_single_1.2"
    # Step C: polyfun LD scores reproduce the committed reference (machine precision)
    import numpy as np
    import pandas as pd
    got = pd.read_parquet(d / f"{stem}.l2.ldscore.parquet")
    exp = pd.read_parquet(repo_root / FX / "single_1.2.l2.ldscore.parquet")
    assert list(got.columns) == list(exp.columns) and got.shape == exp.shape
    num = [c for c in got.columns if got[c].dtype.kind in "fi"]
    assert float(np.abs(got[num].values - exp[num].values).max()) < 1e-9
    # Step A / Step D: annot + .l2.M byte-exact
    assert gzip.open(d / f"{stem}.annot.gz", "rt").read() == \
        gzip.open(repo_root / FX / "expected_single_1.2.annot.gz", "rt").read()
    assert (d / f"{stem}.l2.M").read_text() == \
        (repo_root / FX / "expected_single_1.2.l2.M").read_text()


def test_get_heritability(run_sos, repo_root, tmp_path):
    """get_heritability: polyfun ldsc.py --h2 (via bin) for one trait against the
    committed multi-chromosome LD panel (target LD + 97-annot baseline + weights),
    maf_cutoff=0 so no .frq. The .results table must byte-match the committed
    reference (deterministic sLDSC on a fixed panel)."""
    if not _have_polyfun():
        pytest.skip("polyfun not installed")
    gh = repo_root / FX / "get_heritability"
    panel = gh / "panel"
    out = tmp_path / "out"
    p = run_sos(repo_root / "pipeline/sldsc_enrichment.ipynb", "get_heritability",
                {"cwd": out, "annotation_name": "protocol_example",
                 "all_traits_file": gh / "traits.txt", "sumstat_dir": panel,
                 "target_anno_dirs": gh / "target" / "protocol_example_single_1",
                 "baseline_ld_dir": panel, "baseline_name": "annotations.",
                 "weights_dir": panel, "weight_name": "weights.",
                 "maf_cutoff": 0, "n_blocks": 200},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    got = out / "protocol_example_single_1" / "sumstats2.parquet.results"
    ref = (repo_root / FX / "sldsc_heritability" / "protocol_example_single_1"
           / "sumstats2.parquet.results")
    assert got.read_text() == ref.read_text()          # byte-exact sLDSC enrichment table
