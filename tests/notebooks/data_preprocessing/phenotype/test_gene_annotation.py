"""Notebook tier: gene_annotation.ipynb steps driven end-to-end via SoS on the
committed chr22 fixtures (tests/fixtures/gene_annotation). Each step's worker is
the ported R script code/script/data_preprocessing/phenotype/gene_annotation.R.

annotate_coord_biomart is exercised end-to-end against Ensembl (needs network in CI);
it self-skips if Ensembl is unreachable.
"""
from __future__ import annotations

import pytest

NB = "pipeline/gene_annotation.ipynb"
FX = "tests/fixtures/gene_annotation"
COLLAPSED = "Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz"
EXON = "Homo_sapiens.GRCh38.103.chr22.exon.gtf.gz"


def _base(repo_root, out):
    return dict(cwd=out, modular_script_dir=repo_root / "code/script")


def test_annotate_coord_gene(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    fx = repo_root / FX
    p = run_sos(repo_root / NB, "annotate_coord",
                {**_base(repo_root, out),
                 "phenoFile": fx / "protocol_example.rnaseq.bed.gz",
                 "coordinate_annotation": fx / COLLAPSED,
                 "phenotype_id_column": "gene_id"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.rnaseq.bed.bed.gz").exists()
    assert (out / "protocol_example.rnaseq.bed.bed.gz.tbi").exists()
    assert (out / "protocol_example.rnaseq.bed.region_list.txt").exists()
    assert (out / "protocol_example.rnaseq.bed.gene_list.tsv").exists()


def test_annotate_coord_protein(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    fx = repo_root / FX
    p = run_sos(repo_root / NB, "annotate_coord",
                {**_base(repo_root, out),
                 "phenoFile": fx / "protocol_example.protein.no_coord.tsv",
                 "coordinate_annotation": fx / COLLAPSED,
                 "molecular_trait_type": "protein",
                 "phenotype_id_column": "gene_id"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.protein.no_coord.bed.gz").exists()
    assert (out / "protocol_example.protein.no_coord.bed.gz.tbi").exists()


def test_annotate_coord_atac(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    fx = repo_root / FX
    p = run_sos(repo_root / NB, "annotate_coord",
                {**_base(repo_root, out),
                 "phenoFile": fx / "protocol_example.atac.tsv",
                 "coordinate_annotation": fx / "protocol_example.atac.coordinate_index.tsv",
                 "molecular_trait_type": "atac"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.atac.bed.gz").exists()
    assert (out / "protocol_example.atac.bed.gz.tbi").exists()


def test_annotate_leafcutter_isoforms(run_sos, repo_root, tmp_path):
    # Runs the [map_leafcutter_cluster_to_gene] -> [annotate_leafcutter_isoforms] chain.
    out = tmp_path / "out"
    fx = repo_root / FX
    p = run_sos(repo_root / NB, "annotate_leafcutter_isoforms",
                {**_base(repo_root, out),
                 "phenoFile": fx / "protocol_example.leafcutter.phenotype.bed.gz",
                 "intron_count": fx / "protocol_example.leafcutter.intron_count.tsv",
                 "coordinate_annotation": fx / EXON,
                 "map_stra": "site"},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.leafcutter.intron_count.tsv.leafcutter.clusters_to_genes.txt").exists()
    assert (out / "protocol_example.leafcutter.phenotype.bed.formated.bed.gz").exists()
    assert (out / "protocol_example.leafcutter.phenotype.bed.phenotype_group.txt").exists()


def test_annotate_psichomics_isoforms(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    fx = repo_root / FX
    p = run_sos(repo_root / NB, "annotate_psichomics_isoforms",
                {**_base(repo_root, out),
                 "phenoFile": fx / "protocol_example.psichomics.phenotype.tsv",
                 "coordinate_annotation": fx / EXON},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "protocol_example.psichomics.phenotype.formated.bed.gz").exists()
    assert (out / "protocol_example.psichomics.phenotype.phenotype_group.txt").exists()


def test_annotate_coord_biomart(run_sos, repo_root, tmp_path):
    """End-to-end against Ensembl biomaRt; skipped if Ensembl is unreachable."""
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "annotate_coord_biomart",
                {**_base(repo_root, out),
                 "phenoFile": repo_root / FX / "protocol_example.rnaseq.gene_ID.tsv",
                 "ensembl_version": 115},
                cwd=repo_root, timeout=300)
    if p.returncode != 0:
        blob = (p.stdout + p.stderr).lower()
        if any(k in blob for k in ("could not resolve", "timed out", "timeout", "curl",
                                   "unresponsive", "connection", "ensembl site", "503", "502")):
            pytest.skip("Ensembl biomaRt unreachable (network)")
        assert False, p.stdout + p.stderr
    bed = out / "protocol_example.rnaseq.gene_ID.bed.gz"
    assert bed.exists() and (out / "protocol_example.rnaseq.gene_ID.bed.gz.tbi").exists()
    assert (out / "protocol_example.rnaseq.gene_ID.region_list.txt").exists()
