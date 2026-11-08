"""Notebook tier: twas_ctwas.ipynb [twas] step on the toy TWAS weights, and the
cTWAS chain on the blessed strong-signal fixture.

[twas] scores one region's genes and must reproduce the twas.R expected output.
The cTWAS test runs the full assemble -> estimate-priors -> fine-map chain over
the 20-block chr22 LD grid with the blessed ctwas weights, reproducing the documented
gene-Z 3.168 / susie_pip 1.0. Exercises: the shared-var get_analysis_regions
provider, the block-grid-vs-sketch collapse in gwas_sumstats_construct.R, the
region provenance carried on the committed S4 TwasWeights fixture, and
assembleCtwasInputs / estCtwasParam / finemapCtwasRegions via the ctwas_* wrappers.
The assemble step alone also runs with the weights given as a list file and as a
quoted wildcard, and rejects the two weight options given together.
"""
from __future__ import annotations

import pytest

from helpers.expected import assert_matches_expected


def test_twas(run_sos, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures"
    # The committed xQTL meta tables point at legacy weights; [twas] reads the S4
    # TwasWeights that twas.R's own test scores.
    tw = fx / "twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.twas_weights.s4.rds"
    meta = tmp_path / "xqtl_meta.tsv"
    meta.write_text("#chr\tstart\tend\tregion_id\tTSS\toriginal_data\tcontexts\n"
                    f"chr22\t10000000\t19000000\tENSG00000130538\t15528191\t{tw}\tbulk_rnaseq\n")
    cwd = tmp_path / "out"
    p = run_sos(
        repo_root / "pipeline/twas_ctwas.ipynb", "twas",
        {
            "cwd": cwd,
            "name": "protocol_example",
            "gwas_meta_data": fx / "twas/protocol_example.twas.gwas_meta.tsv",
            "xqtl_meta_data": meta,
            "ld_meta_data": fx / "ld_reference/ld_meta_file.tsv",
            "region-name": "chr22_10000000_19000000",
            # twas.R's default, under which its expected output was made (the
            # notebook defaults to 0.05)
            "rsq_pval_cutoff": "inf",
            "modular_script_dir": repo_root / "code/script",
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr

    # The genes reach twas.R as a per-gene table (a chromosome-wide region's
    # lists are too long for the command line).
    region = cwd / "twas" / "protocol_example.chr22_10000000_19000000"
    table = (region.parent / f"{region.name}.twas_genes.tsv").read_text().splitlines()
    assert table[0] == "twas_weights\tfine_mapping_result\toutput"
    assert [row.split("\t")[0] for row in table[1:]] == [str(tw)]
    assert_matches_expected(region.parent / f"{region.name}.ENSG00000130538.twas.rds",
                            fx / "twas/expected/twas.chr22.rds",
                            mode="tolerant", rtol=1e-6, atol=1e-8)


# The blessed cTWAS weights as committed S4 TwasWeights (region-provenance
# stamped -- chrom/p0/p1 -- which assembleCtwasInputs needs for LD-block
# placement; the legacy_ctwas_weights_to_s4 conversion step is retired).
CTWAS_WEIGHTS = "twas/protocol_example.ctwas_weights.s4.chr22.rds"


def _ctwas_params(repo_root, cwd, **weights):
    """The ctwas workflow's options, with the weights given as --twas-weights
    or --twas-weights-list."""
    fx = repo_root / "tests" / "fixtures"
    # The 20-block ld_meta doubles as the block grid (ctwas_manifest.R) and the
    # LD sketch (gwas_sumstats_construct.R collapses it to the one chr22 panel).
    return {
        "cwd": cwd,
        "name": "protocol_example",
        "gwas_meta_data": fx / "twas/protocol_example.twas.gwas_meta.tsv",
        "xqtl_meta_data": fx / "twas/protocol_example.twas.xqtl_meta.TAB.tsv",
        "ld_meta_data": fx / "ld_reference/ld_meta_file.ctwas.tsv",
        "region-name": "chr22_10000000_19000000",
        **weights,
        # The committed weights carry susie only; pecotmr resolves the
        # method implicitly just for an 'ensemble' entry.
        "ctwas_method": "susie",
        "modular_script_dir": repo_root / "code/script",
    }


def test_ctwas(run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "ctwas"
    s4 = repo_root / "tests" / "fixtures" / CTWAS_WEIGHTS

    # Full cTWAS chain: assemble + estimate priors + fine-map.
    p = run_sos(
        repo_root / "pipeline/twas_ctwas.ipynb", "ctwas",
        {"run_param_est": True, "run_finemapping": True,
         **_ctwas_params(repo_root, cwd, twas_weights=s4)},
        cwd=repo_root, timeout=1200)
    assert p.returncode == 0, p.stdout + p.stderr

    # (3) CtwasResult with the blessed numeric parity (gene susie_pip 1.0,
    # gene-Z 3.168 = ctwas 0.6.0's LD-adjusted burden Z; the pre-0.6.0 5.462
    # under-adjusted the LD). keep_snps defaults True, so the SNP background
    # is retained.
    fm = cwd / "ctwas/protocol_example.ctwas_finemap.rds"
    assert fm.exists(), p.stdout
    info = read_rds(fm)
    assert info["class"] == "CtwasResult"
    assert info["geneMaxPip"] == pytest.approx(1.0, abs=1e-3), info
    assert info["geneTopZ"] == pytest.approx(3.168, abs=0.01), info


@pytest.mark.parametrize("source", ["twas_weights_list", "quoted_wildcard"])
def test_ctwas_assemble_weights(run_sos, repo_root, tmp_path, source):
    # A list file is how a genome-wide set of weights gets in (too long for the
    # command line); a quoted wildcard is expanded by SoS. The folder name has a
    # space, so the step must quote each path.
    folder = tmp_path / "prebuilt weights"
    folder.mkdir()
    s4 = folder / "protocol_example.ctwas_weights.s4.chr22.rds"
    s4.symlink_to(repo_root / "tests" / "fixtures" / CTWAS_WEIGHTS)
    if source == "twas_weights_list":
        listing = tmp_path / "weights.txt"
        listing.write_text(f"{s4}\n")
        weights = {"twas_weights_list": listing}
    else:
        weights = {"twas_weights": f"{folder}/*.rds"}
    cwd = tmp_path / "ctwas"
    p = run_sos(repo_root / "pipeline/twas_ctwas.ipynb", "ctwas",
                _ctwas_params(repo_root, cwd, **weights),
                cwd=repo_root, timeout=1200)
    assert p.returncode == 0, p.stdout + p.stderr

    # The step hands ctwas_assemble.R the same paths, one per line.
    out = cwd / "ctwas"
    assert (out / "protocol_example.ctwas_inputs.weights.txt").read_text().splitlines() == [str(s4)]
    assert (out / "protocol_example.ctwas_inputs.rds").exists()


def test_ctwas_weights_options_exclusive(run_sos, repo_root, tmp_path):
    s4 = repo_root / "tests" / "fixtures" / CTWAS_WEIGHTS
    listing = tmp_path / "weights.txt"
    listing.write_text(f"{s4}\n")
    p = run_sos(repo_root / "pipeline/twas_ctwas.ipynb", "ctwas",
                _ctwas_params(repo_root, tmp_path / "ctwas",
                              twas_weights=s4, twas_weights_list=listing),
                cwd=repo_root, timeout=600)
    assert p.returncode != 0
    assert "Pass --twas-weights or --twas-weights-list, not both." in p.stdout + p.stderr
