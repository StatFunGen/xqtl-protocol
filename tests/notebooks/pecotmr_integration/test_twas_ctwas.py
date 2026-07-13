"""Notebook tier: twas_ctwas.ipynb cTWAS chain on the blessed strong-signal fixture.

Runs the full assemble -> estimate-priors -> fine-map chain over the 20-block
chr22 LD grid with the blessed ctwas weights, reproducing the documented
gene-Z 5.462 / susie_pip 1.0. Exercises: the shared-var get_analysis_regions
provider, the block-grid-vs-sketch collapse in gwas_sumstats_construct.R, the
region-provenance stamp in legacy_ctwas_weights_to_s4.R, and assembleCtwasInputs
/ estCtwasParam / finemapCtwasRegions via the ctwas_* wrappers.
"""
from __future__ import annotations

import pytest


def test_ctwas(run_sos, run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures"
    cwd = tmp_path / "ctwas"
    s4 = tmp_path / "protocol_example.ctwas_weights.s4.rds"

    # (1) Convert the blessed legacy ctwas weights -> S4 TwasWeights. The
    # converter stamps each gene's region provenance (chrom/p0/p1), which
    # assembleCtwasInputs needs for LD-block placement.
    r = run_r(
        repo_root / "code/script/pecotmr_integration/legacy_ctwas_weights_to_s4.R",
        ["--legacy", str(fx / "twas/protocol_example.ctwas_weights.chr22.rds"),
         "--study", "protocol_example", "--method", "susie",
         "--ld-meta", str(fx / "ld_reference/ld_meta_file.ctwas.tsv"),
         "--ld-block", "chr22:10516173-17414263",
         "--output", str(s4)])
    assert r.returncode == 0, r.stdout + r.stderr
    assert s4.exists(), r.stdout

    # (2) Full cTWAS chain: assemble + estimate priors + fine-map. The 20-block
    # ld_meta doubles as the block grid (ctwas_manifest.R) and the LD sketch
    # (gwas_sumstats_construct.R collapses it to the one chr22 panel).
    p = run_sos(
        repo_root / "pipeline/twas_ctwas.ipynb", "ctwas",
        {
            "run_param_est": True,
            "run_finemapping": True,
            "cwd": cwd,
            "name": "protocol_example",
            "gwas_meta_data": fx / "twas/protocol_example.twas.gwas_meta.tsv",
            "xqtl_meta_data": fx / "twas/protocol_example.twas.xqtl_meta.TAB.tsv",
            "ld_meta_data": fx / "ld_reference/ld_meta_file.ctwas.tsv",
            "region-name": "chr22_10000000_19000000",
            "twas_weights": s4,
        },
        cwd=repo_root, timeout=1200)
    assert p.returncode == 0, p.stdout + p.stderr

    # (3) CtwasResult with the blessed numeric parity (gene susie_pip 1.0,
    # gene-Z 5.462). keep_snps defaults True, so the SNP background is retained.
    fm = cwd / "ctwas/protocol_example.ctwas_finemap.rds"
    assert fm.exists(), p.stdout
    info = read_rds(fm)
    assert info["class"] == "CtwasResult"
    assert info["geneMaxPip"] == pytest.approx(1.0, abs=1e-3), info
    assert info["geneTopZ"] == pytest.approx(5.462, abs=0.01), info
