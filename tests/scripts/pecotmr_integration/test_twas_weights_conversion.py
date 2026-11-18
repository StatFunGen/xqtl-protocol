"""Tier B: twas_weights_conversion.R edge cases on small legacy weight files
built here: weights without variant names, genes with no usable weights."""
from __future__ import annotations

import pytest

BUILD = r'''
args <- commandArgs(TRUE); out <- args[1]; mode <- args[2]
ids <- c("chr22:15500100:A:G", "chr22:15500200:C:T", "chr22:15500300:G:A")
w <- switch(mode,
  zero    = list(enet_weights = matrix(0, 2, 1, dimnames = list(ids[1:2], "y"))),
  preset  = list(enet_weights = c(0.1, -0.2)),
  nomatch = list(enet_weights = c(0.1, -0.2, 0.3, 0.4)))
perf <- matrix(c(0.3, 0.09, 0.08, 0.01, 1, 1), 1,
               dimnames = list(NULL, c("corr", "rsq", "adj_rsq", "pval", "RMSE", "MAE")))
co <- list(twas_weights = w,
           twas_cv_result = list(performance = list(enet_performance = perf)),
           region_info = list(grange = list(chrom = "22", start = 15000000L, end = 16000000L),
                              region_coord = list(chrom = "22", start = 15500000L, end = 15500001L)),
           variant_names = ids,
           preset_variants_result = list(variant_names = ids[2:3]))
saveRDS(list(ENSG00000000001 = list(bulk_rnaseq_ENSG00000000001 = co)), out)
'''


def _convert(run_r, repo_root, tmp_path, mode):
    build = tmp_path / "build.R"
    build.write_text(BUILD)
    legacy = tmp_path / f"{mode}.ENSG00000000001.univariate_twas_weights.rds"
    assert run_r(build, [legacy, mode], timeout=120).returncode == 0
    out = tmp_path / "out" / "toy.ENSG00000000001.twas_weights.s4.rds"
    meta = tmp_path / "meta_row.tsv"
    p = run_r(repo_root / "code/script/pecotmr_integration/twas_weights_conversion.R",
              ["--legacy", legacy, "--study", "toy", "--output", out, "--meta-row", meta],
              timeout=300)
    return p, out, meta


def test_conversion_ids_from_preset_variants(run_r, read_rds, repo_root, tmp_path):
    """Weights without names take their IDs from the vector of matching length
    (here preset_variants_result, 2 of the 3 variant_names)."""
    p, out, meta = _convert(run_r, repo_root, tmp_path, "preset")
    assert p.returncode == 0, p.stdout + p.stderr
    info = read_rds(out)
    assert info["class"] == "TwasWeights"
    assert info["nrow"] == 1
    assert meta.read_text().split("\t")[3] == "ENSG00000000001"


def test_conversion_stops_without_matching_ids(run_r, repo_root, tmp_path):
    p, _, _ = _convert(run_r, repo_root, tmp_path, "nomatch")
    assert p.returncode != 0
    assert "No variant IDs of length 4" in p.stdout + p.stderr


def test_conversion_gene_without_usable_weights(run_r, repo_root, tmp_path):
    """All-zero weights: an empty result and an empty meta row, so the gene is
    left out of the meta table instead of failing the conversion."""
    p, out, meta = _convert(run_r, repo_root, tmp_path, "zero")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "No usable (non-zero) weights" in p.stdout + p.stderr
    assert out.exists()
    assert meta.read_text() == ""
