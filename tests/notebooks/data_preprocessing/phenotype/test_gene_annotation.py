"""Notebook tier: gene_annotation.ipynb steps driven end-to-end via SoS on the
committed chr22 fixtures (tests/fixtures/gene_annotation). Each step's worker is
the ported R script code/script/data_preprocessing/phenotype/gene_annotation.R.

annotate_coord_biomart is exercised end-to-end against Ensembl (needs network in CI).
"""
from __future__ import annotations

from helpers.expected import assert_matches_expected

NB = "pipeline/gene_annotation.ipynb"
FX = "tests/fixtures/gene_annotation"
EXPECTED = "tests/fixtures/gene_annotation/expected"
COLLAPSED = "Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz"
EXON = "Homo_sapiens.GRCh38.103.chr22.exon.gtf.gz"


def _base(repo_root, out):
    return dict(cwd=out, modular_script_dir=repo_root / "code/script")


def _expect(repo_root, produced, *, normalize_paths=False):
    """Value-compare a produced gene_annotation output against its committed snapshot.
    gene_annotation.R is deterministic (rtracklayer/dplyr, no RNG); ``normalize_paths``
    is for the region_list outputs whose trailing ``path`` column embeds the cwd."""
    assert_matches_expected(produced, repo_root / EXPECTED / produced.name,
                            mode="tolerant", rtol=1e-6, atol=1e-8,
                            normalize_paths=normalize_paths)


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
    _expect(repo_root, out / "protocol_example.rnaseq.bed.bed.gz")
    _expect(repo_root, out / "protocol_example.rnaseq.bed.gene_list.tsv")
    _expect(repo_root, out / "protocol_example.rnaseq.bed.region_list.txt", normalize_paths=True)


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
    _expect(repo_root, out / "protocol_example.protein.no_coord.bed.gz")
    _expect(repo_root, out / "protocol_example.protein.no_coord.gene_list.tsv")
    _expect(repo_root, out / "protocol_example.protein.no_coord.region_list.txt", normalize_paths=True)


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
    _expect(repo_root, out / "protocol_example.atac.bed.gz")
    _expect(repo_root, out / "protocol_example.atac.region_list.txt", normalize_paths=True)


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
    _expect(repo_root, out / "protocol_example.leafcutter.intron_count.tsv.leafcutter.clusters_to_genes.txt")
    _expect(repo_root, out / "protocol_example.leafcutter.phenotype.bed.formated.bed.gz")
    _expect(repo_root, out / "protocol_example.leafcutter.phenotype.bed.phenotype_group.txt")


