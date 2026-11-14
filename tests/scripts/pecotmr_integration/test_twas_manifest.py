"""Tier B: twas_manifest.R resolves the twas metas into per-region TWAS units."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/twas_manifest/expected/twas_manifest.tsv"


def test_twas_manifest(run_r, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/twas"
    out = tmp_path / "manifest.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/twas_manifest.R",
              ["--gwas-meta", fx / "protocol_example.twas.gwas_meta.tsv",
               "--xqtl-meta", fx / "protocol_example.twas.xqtl_meta.TAB.tsv",
               "--region-name", "chr22_10000000_19000000", "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert rows[0] == ["region_id", "chrom", "start", "stop", "genes", "weight_files",
                       "gwas_studies", "gwas_files", "gwas_mappings"]
    assert len(rows) == 2
    assert rows[1][0] == "chr22_10000000_19000000" and "ENSG00000130538" in rows[1][4]
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant",
                            rtol=1e-6, atol=1e-8, normalize_paths=True)


TOY_WEIGHTS = "tests/fixtures/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.twas_weights.s4.rds"


def _gene_meta(path, rows):
    path.write_text("#chr\tstart\tend\tregion_id\tTSS\toriginal_data\tcontexts\n"
                    + "".join(f"chr22\t10000000\t19000000\t{g}\t15528191\t{w}\tbulk_rnaseq\n"
                              for g, w in rows))
    return path


def _gene_manifest(run_r, repo_root, meta, out, *extra):
    return run_r(repo_root / "code/script/pecotmr_integration/twas_manifest.R",
                 ["--gwas-meta", repo_root / "tests/fixtures/twas/protocol_example.twas.gwas_meta.tsv",
                  "--xqtl-meta", meta, "--unit", "gene", *extra, "--output", out], timeout=120)


def test_twas_manifest_gene_unit(run_r, repo_root, tmp_path):
    """--unit gene: one row per gene, named by the gene and spanning its meta window."""
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG00000130538", repo_root / TOY_WEIGHTS)])
    out = tmp_path / "manifest.tsv"
    p = _gene_manifest(run_r, repo_root, meta, out, "--min-cv-rsq", "0")
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [l.split("\t") for l in out.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1][:4] == ["ENSG00000130538", "chr22", "10000000", "19000000"]


def test_twas_manifest_gene_unit_rejects_repeated_gene(run_r, repo_root, tmp_path):
    w = repo_root / TOY_WEIGHTS
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG00000130538", w), ("ENSG00000130538", w)])
    p = _gene_manifest(run_r, repo_root, meta, tmp_path / "manifest.tsv")
    assert p.returncode != 0
    assert "one xQTL meta row per gene" in p.stdout + p.stderr


def test_twas_manifest_prefilter_keeps_unreadable_weights(run_r, repo_root, tmp_path):
    """A weight file the CV prefilter cannot read is kept (with a warning) for the
    twas step to report, instead of failing the manifest."""
    bad = tmp_path / "corrupt.rds"
    bad.write_text("not an RDS file")
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG00000130538", repo_root / TOY_WEIGHTS),
                                              ("ENSG_CORRUPT", bad)])
    out = tmp_path / "manifest.tsv"
    p = _gene_manifest(run_r, repo_root, meta, out, "--min-cv-rsq", "0.01", "--max-cv-pval", "Inf")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "could not read weights of ENSG_CORRUPT" in p.stdout + p.stderr
    ids = [l.split("\t")[0] for l in out.read_text().splitlines()[1:]]
    assert "ENSG_CORRUPT" in ids


def test_twas_manifest_prefilter_drops_failing_genes(run_r, repo_root, tmp_path):
    """A CV r2 cutoff no model can reach drops the gene, so nothing is left."""
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG00000130538", repo_root / TOY_WEIGHTS)])
    p = _gene_manifest(run_r, repo_root, meta, tmp_path / "manifest.tsv",
                       "--min-cv-rsq", "2", "--max-cv-pval", "Inf")
    assert p.returncode != 0
    assert "kept 0 of 1 genes" in p.stdout + p.stderr


@pytest.mark.parametrize("max_p, kept", [("0.05", True), ("0.02", False)])
def test_twas_manifest_prefilter_pvalue_cutoff(run_r, repo_root, tmp_path, max_p, kept):
    """The CV p-value cutoff: the toy gene's best model (mrash, rsq 0.089, CV
    p 0.022) passes p < 0.05 and no model passes p < 0.02. The toy weights have
    no adj_rsq_pval, so this also covers the fall-back to pval, and they carry
    an ensemble method without CV metrics, which must be skipped."""
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG00000130538", repo_root / TOY_WEIGHTS)])
    p = _gene_manifest(run_r, repo_root, meta, tmp_path / "manifest.tsv",
                       "--min-cv-rsq", "0.01", "--max-cv-pval", max_p)
    out = p.stdout + p.stderr
    if kept:
        assert p.returncode == 0, out
        assert "kept 1 of 1 genes" in out
    else:
        assert p.returncode != 0
        assert "kept 0 of 1 genes" in out


def test_twas_manifest_gene_unit_needs_window(run_r, repo_root, tmp_path):
    meta = tmp_path / "meta.tsv"
    meta.write_text("#chr\tstart\tend\tregion_id\tTSS\toriginal_data\tcontexts\n"
                    f"chr22\tNA\t19000000\tENSG00000130538\t15528191\t{repo_root / TOY_WEIGHTS}\tbulk_rnaseq\n")
    p = _gene_manifest(run_r, repo_root, meta, tmp_path / "manifest.tsv")
    assert p.returncode != 0
    assert "needs a start and end for every gene" in p.stdout + p.stderr


def test_twas_manifest_gene_unit_file_name_safe(run_r, repo_root, tmp_path):
    """sQTL-style IDs with ':' become file-name safe unit IDs."""
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG00000130538:clu_1", repo_root / TOY_WEIGHTS)])
    out = tmp_path / "manifest.tsv"
    p = _gene_manifest(run_r, repo_root, meta, out)
    assert p.returncode == 0, p.stdout + p.stderr
    row = out.read_text().splitlines()[1].split("\t")
    assert row[0] == "ENSG00000130538_clu_1"
    assert row[4] == "ENSG00000130538:clu_1"


def test_twas_manifest_gene_unit_rejects_name_clash(run_r, repo_root, tmp_path):
    w = repo_root / TOY_WEIGHTS
    meta = _gene_meta(tmp_path / "meta.tsv", [("ENSG1:a", w), ("ENSG1_a", w)])
    p = _gene_manifest(run_r, repo_root, meta, tmp_path / "manifest.tsv")
    assert p.returncode != 0
    assert "give the same file name" in p.stdout + p.stderr
