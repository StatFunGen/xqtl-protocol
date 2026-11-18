"""Workflow Builder commands are generated from the module pages (website/builder_registry.py).

The selection rules fail closed: one example is used automatically, several need a
`builder` tag, and anything ambiguous or missing is an error naming the cell.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("builder_registry", REPO / "website/builder_registry.py")
reg = importlib.util.module_from_spec(spec)
sys.modules["builder_registry"] = reg  # dataclasses look the module up by name
spec.loader.exec_module(reg)


def cand(cell, wf, cmd="sos run pipeline/m.ipynb x", tags=(), meta=None):
    return reg.Candidate("m", cell, wf, cmd, list(tags), meta or {})


def test_single_untagged_example_is_used():
    c = cand(1, "qc")
    assert reg.select([c], "m", "qc") is c


def test_several_untagged_examples_need_a_tag():
    with pytest.raises(reg.RegistryError, match="tag the builder's one"):
        reg.select([cand(1, "qc"), cand(2, "qc")], "m", "qc")


def test_tagged_example_wins_over_untagged():
    tagged = cand(2, "qc", tags=["builder"])
    assert reg.select([cand(1, "qc"), tagged], "m", "qc") is tagged


def test_two_tagged_examples_are_an_error():
    with pytest.raises(reg.RegistryError, match="2 cells tagged"):
        reg.select([cand(1, "qc", tags=["builder"]), cand(2, "qc", tags=["builder"])], "m", "qc")


def test_named_variant_is_selected_by_its_tag():
    std = cand(1, "cis")
    inter = cand(2, "cis", tags=["builder:interaction"])
    assert reg.select([std, inter], "m", "cis", "interaction") is inter
    assert reg.select([std, inter], "m", "cis") is std  # the plain workflow ignores the variant


def test_missing_example_is_an_error():
    with pytest.raises(reg.RegistryError, match="no example command"):
        reg.select([cand(1, "other")], "m", "qc")


def test_combined_run_matches_its_first_workflow():
    c = cand(1, "qtl_dataset_construct+susie_twas")
    assert reg.select([c], "m", "qtl_dataset_construct") is c


def test_declared_inputs_outputs_are_not_inferred():
    c = cand(1, "qc", meta={"inputs": {"geno": {"kind": "genotype_plink", "source": "user"}},
                            "outputs": {"qc": {"kind": "genotype_plink", "path": "{cwd}/{name}.qc.bed"}}})
    io, inferred = reg.io_of(c)
    assert not inferred and io == {"i": ["user"], "o": ["{cwd}/{name}.qc.bed"]}


def test_inference_skips_placeholders_and_names():
    c = cand(1, "qc", cmd="sos run pipeline/m.ipynb qc --genoFile a/b.bed --name x/y "
                          "--ref <path/to/ref> --cwd out/qc")
    io, inferred = reg.io_of(c)
    assert inferred and io == {"i": ["a/b.bed"], "o": ["out/qc"]}


def test_build_is_deterministic():
    first, second = reg.build(), reg.build()
    assert reg.dump(first[0]) == reg.dump(second[0]) and reg.dump(first[1]) == reg.dump(second[1])
    assert first[2] == second[2]
