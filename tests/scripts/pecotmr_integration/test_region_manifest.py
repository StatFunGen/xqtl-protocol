"""Tier B: manifest-builder wrappers produce well-formed TSVs from the fixtures."""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected

EXPECTED = "tests/fixtures/region_manifest/expected/region_manifest.tsv"
SCRIPT = "code/script/pecotmr_integration/region_manifest.R"


def test_region_manifest(run_r, repo_root, qtl_mini, tmp_path):
    out = tmp_path / "regions.tsv"
    p = run_r(repo_root / SCRIPT,
              ["--pheno-manifest", qtl_mini / "protocol_example.pheno_manifest_context.tsv",
               "--customized-association-windows", qtl_mini / "association_windows.bed",
               "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = out.read_text().splitlines()
    assert lines, "region manifest is empty"
    cols = lines[0].split("\t")
    assert {"region_id", "chr", "start", "end"} <= set(cols)
    assert any("ENSG00000283047" in ln for ln in lines[1:])
    assert_matches_expected(out, repo_root / EXPECTED, mode="tolerant", rtol=1e-6, atol=1e-8)


def _manifest_rows(run_r, repo_root, tmp_path, pheno, *args):
    out = tmp_path / "manifest.tsv"
    p = run_r(repo_root / SCRIPT,
              ["--pheno-manifest", pheno, *args, "--output", out], timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [line.split("\t") for line in out.read_text().splitlines()]
    return [dict(zip(rows[0], row)) for row in rows[1:]]


def test_region_manifest_selection(run_r, repo_root, tmp_path):
    """Gene mode ([susie_twas] / [mnm]) runs on region_name, so it must keep an
    ID as given (a versioned ENSG ID) while region_id is made filename-safe.
    --region-name-list reads like --region-list and, like --region-name, wins
    over it."""
    pheno = tmp_path / "pheno.tsv"
    pheno.write_text("#chr\tstart\tend\tID\n"
                     "chr22\t100\t200\tENSG00000000001.5\n"
                     "chr22\t300\t400\tENSG00000000002\n")
    names = tmp_path / "names.txt"
    names.write_text("# genes to run\n\nchr22 0 0 ENSG00000000002\n"
                     "  ENSG00000000001.5  \nENSG00000000002\n")
    regions = tmp_path / "regions.txt"
    regions.write_text("ENSG00000000001.5\n")

    rows = _manifest_rows(run_r, repo_root, tmp_path, pheno,
                          "--region-name-list", names, "--region-list", regions)
    assert [r["region_name"] for r in rows] == ["ENSG00000000002", "ENSG00000000001.5"]
    assert [r["region_id"] for r in rows] == ["ENSG00000000002", "ENSG00000000001_5"]
    rows = _manifest_rows(run_r, repo_root, tmp_path, pheno, "--region-list", regions)
    assert [r["region_name"] for r in rows] == ["ENSG00000000001.5"]


def test_region_manifest_gene_mode_without_coordinates(run_r, repo_root, tmp_path):
    """qtl_dataset_construct does not need #chr/start/end, and neither does gene
    mode: it lists the genes with empty coordinates. Region mode still needs them
    for each gene's window, and gene mode still rejects an ID that is not a gene."""
    pheno = tmp_path / "pheno.tsv"
    pheno.write_text("ID\tpath\tcond\n"
                     "ENSG00000000001.5\tx.bed.gz\tcontext1\n"
                     "ENSG00000000002\tx.bed.gz\tcontext1\n")
    rows = _manifest_rows(run_r, repo_root, tmp_path, pheno, "--gene-mode")
    assert [r["region_name"] for r in rows] == ["ENSG00000000001.5", "ENSG00000000002"]
    assert all(r["chr"] == r["start"] == r["end"] == r["ld_block"] == "" for r in rows)
    for args, error in [((), "needs ID, #chr/chrom, start, end columns"),
                        (("--gene-mode", "--region-name", "ENSG00000000009"),
                         "Gene 'ENSG00000000009' not found in --pheno-manifest")]:
        p = run_r(repo_root / SCRIPT, ["--pheno-manifest", pheno, *args,
                                       "--output", tmp_path / "rejected.tsv"], timeout=120)
        assert p.returncode != 0
        assert error in p.stdout + p.stderr


@pytest.mark.parametrize("flag", ["--region-list", "--customized-association-windows"])
def test_region_manifest_missing_file(run_r, repo_root, qtl_mini, tmp_path, flag):
    """A list or windows file that is given must exist: skipping it would
    quietly analyze every region instead."""
    p = run_r(repo_root / SCRIPT,
              ["--pheno-manifest", qtl_mini / "protocol_example.pheno_manifest_context.tsv",
               flag, tmp_path / "missing.txt", "--output", tmp_path / "manifest.tsv"],
              timeout=120)
    assert p.returncode != 0
    assert "not found" in p.stdout + p.stderr
