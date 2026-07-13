"""Notebook tier: sldsc_enrichment.ipynb S-LDSC post-processing on downsized fixtures.

The upstream steps (make_annotation_files_ldscore / munge_sumstats_polyfun /
get_heritability) run polyfun + ldsc over the ~350M S-LDSC reference panel and are
OUT of CI scope: polyfun is not in the pixi env, and they are external-tool
orchestration, not pecotmr wrappers. This tests the two pecotmr wrapper steps --
`postprocess` and `meta_subset` -- which consume polyfun's small per-trait OUTPUTS
(a ~2MB fixture: the .results/.log/.part_delete triples + the target .annot.gz).
`--maf-cutoff 0` opts out of MAF filtering, so no .frq reference is needed.
"""
from __future__ import annotations

import pytest


def test_postprocess_and_meta_subset(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests" / "fixtures" / "sldsc_enrichment"
    cwd = tmp_path / "sldsc"

    # (1) postprocess: bundle polyfun's per-trait outputs into a SldscData and run
    # the DerSimonian-Laird random-effects meta -> per_trait / meta / params.
    p = run_sos(
        repo_root / "pipeline/sldsc_enrichment.ipynb", "postprocess",
        {
            "cwd": cwd,
            "annotation_name": "protocol_example",
            "traits_file": fx / "sumstats_test_all.txt",
            "heritability_cwd": fx / "sldsc_heritability",
            "target_categories": "ANNOT_0",
            "target_categories_label": "protocol_example_annotation",
            "target_anno_dir": fx / "target_anno",
            "maf_cutoff": 0,
            "polyfun_path": ".",       # unused by postprocess; global param must be set
            "python_exec": "python",
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    pp = cwd / "protocol_example.sldsc_postprocess.rds"
    assert pp.exists(), p.stdout
    info = read_rds(pp)
    assert info["class"] == "list"
    assert {"per_trait", "meta", "params"}.issubset(set(info["names"])), info

    # (2) meta_subset: re-meta a trait subset off the postprocess RDS (no rerun).
    p = run_sos(
        repo_root / "pipeline/sldsc_enrichment.ipynb", "meta_subset",
        {
            "cwd": cwd,
            "annotation_name": "protocol_example",
            "postprocess_rds": pp,
            "subset_traits_file": fx / "sumstats_test_category1.txt",
            "subset_name": "category1",
            "target_categories": "ANNOT_0",
            "polyfun_path": ".",
            "python_exec": "python",
        },
        cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    meta = cwd / "protocol_example.category1.meta.rds"
    assert meta.exists(), p.stdout
    minfo = read_rds(meta)
    assert minfo["class"] == "list"
    assert "enrichment" in minfo["names"], minfo
