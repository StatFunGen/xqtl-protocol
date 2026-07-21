"""Notebook tier: twas_ctwas.ipynb cTWAS chain on the blessed strong-signal fixture.

Runs the full assemble -> estimate-priors -> fine-map chain over the 20-block
chr22 LD grid with the blessed ctwas weights, reproducing the documented
gene-Z 5.462 / susie_pip 1.0. Exercises: the shared-var get_analysis_regions
provider, the block-grid-vs-sketch collapse in gwas_sumstats_construct.R, the
region provenance carried on the committed S4 TwasWeights fixture, and
assembleCtwasInputs / estCtwasParam / finemapCtwasRegions via the ctwas_* wrappers.
"""
from __future__ import annotations

import pytest


def test_ctwas(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures"
    cwd = tmp_path / "ctwas"
    # The blessed cTWAS weights as committed S4 TwasWeights (region-provenance
    # stamped -- chrom/p0/p1 -- which assembleCtwasInputs needs for LD-block
    # placement; the legacy_ctwas_weights_to_s4 conversion step is retired).
    s4 = fx / "twas/protocol_example.ctwas_weights.s4.chr22.rds"

    # Full cTWAS chain: assemble + estimate priors + fine-map. The 20-block
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
