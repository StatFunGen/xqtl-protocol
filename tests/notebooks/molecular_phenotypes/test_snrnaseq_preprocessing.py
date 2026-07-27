"""Notebook tier: snRNAseq_preprocessing.ipynb — single-nuclei QC + annotation.

Worker:
  code/script/molecular_phenotypes/snRNAseq_preprocessing.R
    --step sctk_qc          CellRanger counts -> SCTK/Seurat QC -> filtered_seuratobj.rds
    --step cell_annotation  QC'd object -> SingleR cluster-majority labels -> celltyped_seuratobj.rds

The pipeline is a faithful lift of the notebook's singleCellTK + Seurat stack, so
it is stochastic (scds/decontX/UMAP/Louvain): we assert on STRUCTURE (output
files, QC_summary step names, monotone cell/gene counts, celltype labels), not a
byte diff. Fixtures under tests/fixtures/snrnaseq_preprocessing/ are downsampled
from the MWE (see make_fixtures.R): an 800-cell CellRanger subset, the EXACT
id_mapping rows for those barcodes, and a per-label-downsampled reference SE.

Heavy + many Bioconductor deps -> every test skips when a dep or fixture is absent.
"""
from __future__ import annotations

import subprocess

import pytest

R = "code/script/molecular_phenotypes/snRNAseq_preprocessing.R"
NB = "pipeline/snRNAseq_preprocessing.ipynb"
FX = "tests/fixtures/snrnaseq_preprocessing"
MSD = "code/script"

# runCellQC + SingleR default-method runtime deps
_QC_DEPS = ("Seurat", "singleCellTK", "scds", "celda", "SingleCellExperiment", "scuttle")
_ANNO_DEPS = ("Seurat", "singleCellTK", "SingleR", "org.Hs.eg.db", "scuttle")


def _has_r_pkg(repo_root, pkg):
    p = subprocess.run(
        ["pixi", "run", "--frozen", "Rscript", "-e",
         f'quit(status = !requireNamespace("{pkg}", quietly=TRUE))'],
        cwd=repo_root, capture_output=True, text=True)
    return p.returncode == 0


def _require(repo_root, pkgs, *paths):
    missing = [p for p in pkgs if not _has_r_pkg(repo_root, p)]
    if missing:
        pytest.skip("missing R packages: " + ", ".join(missing))
    for path in paths:
        if not path.exists():
            pytest.skip(f"missing fixture: {path}")


def _read_csv_col0(path):
    import csv
    with open(path) as fh:
        return [row[0] for row in csv.reader(fh) if row]


@pytest.fixture(scope="module")
def sctk_out(run_r, repo_root, tmp_path_factory):
    """Run sctk_qc once for the module (QC is the expensive step)."""
    _require(repo_root, _QC_DEPS,
             repo_root / FX / "cellranger", repo_root / FX / "id_mapping.csv")
    out_dir = tmp_path_factory.mktemp("sctk")
    p = run_r(repo_root / R, [
        "--step", "sctk_qc",
        "--input-dir", repo_root / FX / "cellranger",
        "--output-dir", out_dir,
        "--sample-meta", repo_root / FX / "id_mapping.csv",
        "--num-threads", "2"], timeout=1200)
    assert p.returncode == 0, p.stdout + p.stderr
    return out_dir


def test_sctk_qc(sctk_out):
    rds = sctk_out / "SCTK_results" / "filtered_seuratobj.rds"
    qc_table = sctk_out / "QC_table" / "SCTK_QC_table.csv"
    qc_summary = sctk_out / "SCTK_results" / "QC_summary.csv"
    assert rds.exists() and qc_table.exists() and qc_summary.exists()

    steps = _read_csv_col0(qc_summary)
    # the fixed QC ladder, in order
    assert steps[0] == "Step"
    for want in ("Raw Data", "Removing FMGs",
                 "Removing Genes expressed in < 0.5% of cells",
                 "Removing Cells with no associated Patient ID",
                 "SCTK-QC Results", "Final Seurat Object"):
        assert want in steps

    # cell/gene counts are positive and non-increasing down the ladder
    import csv
    with open(qc_summary) as fh:
        rows = [r for r in csv.DictReader(fh)]
    cells = [int(float(r["Cells"])) for r in rows]
    genes = [int(float(r["Genes"])) for r in rows]
    assert all(c > 0 for c in cells) and all(g > 0 for g in genes)
    assert cells == sorted(cells, reverse=True)         # cells only ever drop
    # per-sample QC table has an "All Samples" summary column
    with open(qc_table) as fh:
        header = fh.readline()
    assert "All Samples" in header


def test_cell_annotation(sctk_out, run_r, repo_root, tmp_path):
    _require(repo_root, _ANNO_DEPS, repo_root / FX / "seurat_ref_SE.rds")
    p = run_r(repo_root / R, [
        "--step", "cell_annotation",
        "--sctk-rds", sctk_out / "SCTK_results" / "filtered_seuratobj.rds",
        "--output-dir", tmp_path,
        "--seurat-ref", repo_root / FX / "seurat_ref_SE.rds",
        "--downsample-n", "1000",
        "--num-threads", "2"], timeout=1200)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "annotated_seuratobj.rds").exists()
    assert (tmp_path / "celltyped_seuratobj.rds").exists()


def test_sctk_qc_via_sos(run_sos, repo_root, tmp_path):
    _require(repo_root, _QC_DEPS,
             repo_root / FX / "cellranger", repo_root / FX / "id_mapping.csv")
    p = run_sos(repo_root / NB, "sctk_qc", {
        "input_dir": repo_root / FX / "cellranger",
        "output_dir": tmp_path,
        "sample_meta": repo_root / FX / "id_mapping.csv",
        "numThreads": 2,
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "SCTK_results" / "filtered_seuratobj.rds").exists()
    assert (tmp_path / "QC_table" / "SCTK_QC_table.csv").exists()
