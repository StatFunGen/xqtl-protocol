"""Notebook tier: mnm_postprocessing.ipynb (thin FMR wrappers).

Drives the migrated cis-results-export / top-loci / plot / VCF workflows on a
committed single-gene susie FineMappingResult fixture (derived once from the
qtl_mini bundle). The heavy legacy logic now lives in pecotmr + the
pecotmr_integration wrappers, so these cells are thin bash calls.
"""
from __future__ import annotations

import glob

import pytest

NB = "code/SoS/mnm_analysis/mnm_postprocessing.ipynb"
FMR = "protocol_example.ENSG00000283047.fine_mapping.rds"


def test_cis_results_export_and_top_loci(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures" / "mnm_postprocessing"
    cwd = tmp_path / "out"
    region_file = tmp_path / "regions.tsv"
    region_file.write_text("chr22\t9939388\t11961338\tENSG00000283047\n")
    base = dict(cwd=cwd, study="protocol_example",
                region_file=region_file, file_path=fx,
                prefix="protocol_example", suffix="fine_mapping.rds")

    # cis_results_export: combine per-region FMR(s) -> cis_results_db + meta.
    p = run_sos(repo_root / NB, "cis_results_export", base, cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    db = cwd / "protocol_example.ENSG00000283047.cis_results_db.rds"
    assert db.exists(), p.stdout
    meta = cwd / "protocol_example.cis_results_db.tsv"
    assert meta.exists(), p.stdout
    # TSS from getTraitPosition should be a real coordinate for this gene.
    txt = meta.read_text().splitlines()
    assert "TSS" in txt[0]
    hdr = txt[0].lstrip("#").split("\t")
    row = dict(zip(hdr, txt[1].split("\t")))
    assert row["region_id"] == "ENSG00000283047"
    assert row["TSS"].isdigit(), row

    # export_top_loci: per-region BED -> combined study BED.
    p = run_sos(repo_root / NB, "export_top_loci", dict(base, qtl_type="eQTL"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (cwd / "summary" / "protocol_example.eQTL.top_loci.bed.gz").exists(), p.stdout


@pytest.mark.parametrize("wf,pattern", [
    ("susie_pip_landscape_plot", "*.pip_landscape_plot.pdf"),
    ("susie_upsetR_plot", "*.UpSetR.pdf"),
])
def test_plot_wrappers(run_sos, repo_root, tmp_path, wf, pattern):
    fx = repo_root / "tests" / "fixtures" / "mnm_postprocessing"
    cwd = tmp_path / "out"
    base = dict(cwd=cwd, study="protocol_example", rds_path=fx / FMR)
    p = run_sos(repo_root / NB, wf, base, cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    hits = glob.glob(str(cwd / "**" / pattern), recursive=True)
    assert hits, (wf, p.stdout)


@pytest.mark.parametrize("wf,pattern", [
    ("fsusie_extract_effect", "*.estimated_effect.tsv"),
    ("fsusie_affected_region", "*.affected_region.tsv"),
])
def test_fsusie_wrappers(run_sos, repo_root, tmp_path, wf, pattern):
    fx = repo_root / "tests" / "fixtures" / "mnm_postprocessing"
    cwd = tmp_path / "out"
    base = dict(cwd=cwd, study="protocol_example",
                rds_path=fx / "protocol_example.fsusie.fine_mapping.rds")
    p = run_sos(repo_root / NB, wf, base, cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    hits = glob.glob(str(cwd / "**" / pattern), recursive=True)
    assert hits, (wf, p.stdout)


def test_overlap_qtl_gwas(run_sos, repo_root, tmp_path):
    # QTL x GWAS top-loci overlap -> block_top_loci for the coloc manifest. Uses
    # the susie_enloc FMR fixtures: the chr1 gene overlaps both chr1 GWAS blocks.
    import csv
    fx = repo_root / "tests" / "fixtures" / "susie_enloc"
    cwd = tmp_path / "out"
    base = dict(cwd=cwd, study="protocol_example",
                qtl_meta_path=fx / "protocol_example.enloc.xqtl_meta.fmr.tsv",
                gwas_meta_path=fx / "protocol_example.enloc.gwas_meta.fmr.tsv",
                qtl_file_path=fx, gwas_file_path=fx)
    p = run_sos(repo_root / NB, "overlap_qtl_gwas", base, cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    out = cwd / "protocol_example.overlapped.gwas.tsv"
    assert out.exists(), p.stdout
    rows = list(csv.DictReader(out.open(), delimiter="\t"))
    assert rows, p.stdout
    btl = rows[0].get("block_top_loci") or ""
    # allele-aware overlapTopLoci finds shared top-loci variants in both blocks.
    assert "chr1_20110062_22020160" in btl, rows[0]
    assert "chr1_22020160_24199848" in btl, rows[0]


def test_mv_susie_vcf(run_sos, repo_root, tmp_path):
    # Per-context fine-mapping VCFs (ES/SE/LP/AF + PIP/CS) from an mvSuSiE FMR.
    fx = repo_root / "tests" / "fixtures" / "mnm_postprocessing"
    cwd = tmp_path / "out"
    base = dict(cwd=cwd, study="protocol_example",
                rds_path=fx / "protocol_example.mvsusie.fine_mapping.rds")
    p = run_sos(repo_root / NB, "mv_susie", base, cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    vcfs = glob.glob(str(cwd / "**" / "*.vcf.bgz"), recursive=True)
    assert vcfs, p.stdout
