"""Workflow Builder commands are generated from the module pages (website/builder_registry.py).

The selection rules fail closed: one example is used automatically, several need a
`builder` tag, and anything ambiguous or missing is an error naming the cell.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shlex
import shutil
import subprocess
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


def test_declared_inputs_outputs_are_shown():
    c = cand(1, "qc", meta={"inputs": {"geno": {"kind": "plink", "path": "a/b.bed"}},
                            "outputs": {"cwd": {"kind": "directory", "path": "output/qc"}}})
    assert reg.io_of(c) == {"i": ["a/b.bed"], "o": ["output/qc"]}


def _step(m, wf, cmd, ins, outs):
    return (m, wf), reg.Candidate(m, 1, wf, cmd, [], {"inputs": ins, "outputs": outs})


UP = _step("a", "make", "sos run pipeline/a.ipynb make --cwd output/a",
           {}, {"cwd": {"kind": "directory", "path": "output/a"}})


def _down(source="a.make.cwd", path="output/a/x.bed", extra=""):
    ins = {"geno": {"kind": "plink", "path": path}}
    if source:
        ins["geno"]["source"] = source
    return _step("b", "use", f"sos run pipeline/b.ipynb use --geno {path}{extra} --cwd output/b",
                 ins, {"cwd": {"kind": "directory", "path": "output/b"}})


def test_linked_input_passes():
    assert reg.check_io(dict([UP, _down()]), set()) == []


def test_input_in_command_must_be_declared():
    errs = reg.check_io(dict([UP, _down(extra=" --ref tests/fixtures/x.fa")]), set())
    assert any("not declared as an input" in e for e in errs)


def test_source_must_be_a_builder_output():
    errs = reg.check_io(dict([UP, _down(source="a.other.cwd")]), set())
    assert any("not a builder step output" in e for e in errs)


def test_input_must_sit_under_its_source():
    errs = reg.check_io(dict([UP, _down(path="output/z/x.bed")]), set())
    assert any("is not under" in e for e in errs)


def test_unlinked_output_input_is_an_error_unless_known():
    step = dict([UP, _down(source=None)])
    assert any("made by no builder step" in e for e in reg.check_io(step, set()))
    assert reg.check_io(step, {("b|use", "output/a/x.bed")}) == []


def test_stale_known_gap_is_an_error():
    errs = reg.check_io(dict([UP, _down()]), {("b|use", "output/a/x.bed")})
    assert any("no longer applies" in e for e in errs)


# ---- the repository itself -------------------------------------------------------

BUILT = reg.build()
SELECTED = BUILT[3]


def test_registry_has_no_errors():
    assert BUILT[2] == []


def test_committed_builder_matches_generator():
    """Edit the module page and run builder_registry.py --write; never edit CMDLIB/WIO by hand."""
    s = "".join(json.loads(reg.BUILDER.read_text(encoding="utf-8"))["cells"][0]["source"])
    for name, obj in (("CMDLIB", BUILT[0]), ("WIO", BUILT[1])):
        i = s.index(name + "=") + len(name) + 1
        committed, _ = json.JSONDecoder().raw_decode(s[i:])
        assert committed == obj, f"{name} in the builder is out of date: run website/builder_registry.py --write"


def _tested_steps():
    covered = set()
    for f in (REPO / "tests" / "notebooks").rglob("test_*.py"):
        t = f.read_text(encoding="utf-8")
        nbs = set(re.findall(r"(\w+)\.ipynb", t))
        pieces = {p for lit in re.findall(r"""["']([\w+]+)["']""", t) for p in lit.split("+")}
        covered |= {k for k in SELECTED if k[0] in nbs and k[1].split("+")[0] in pieces}
    return covered


UNTESTED = REPO / "website" / "builder" / "untested_steps.json"


def test_every_builder_step_is_run_by_a_notebook_test():
    """A new builder step needs a notebook test; a step that gains one leaves untested_steps.json."""
    known = set(json.loads(UNTESTED.read_text(encoding="utf-8"))["steps"])
    missing = {f"{m}|{wf}" for m, wf in SELECTED} - {f"{m}|{wf}" for m, wf in _tested_steps()}
    assert missing - known == set(), "builder steps with no notebook test"
    assert known - missing == set(), "now tested or gone: remove from untested_steps.json"


def _help(notebook):
    r = subprocess.run([shutil.which("sos"), "run", str(notebook), "-h"],
                       capture_output=True, text=True, timeout=300, cwd=REPO)
    return r.stdout + r.stderr


def _norm(opt):
    opt = opt.lstrip("-").replace("-", "_").lower()
    return opt[3:] if opt.startswith("no_") else opt


@pytest.mark.skipif(shutil.which("sos") is None, reason="needs sos")
@pytest.mark.parametrize("module", sorted({m for m, _ in SELECTED}))
def test_builder_options_exist(module):
    """Every workflow and --option in a builder command is listed by `sos run <module> -h`."""
    h = _help(reg.notebooks()[module])
    opts = {_norm(o.replace("[no-]", "")) for o in re.findall(r"(?<![\w-])--(?:\[no-\])?[A-Za-z][\w-]*", h)}
    wfs = set(re.findall(r"^\s{2}(\w+)\s*$", h.split("Sections")[0], re.M))
    bad = []
    for (m, wf), c in SELECTED.items():
        if m != module:
            continue
        toks = shlex.split(c.command.lstrip("# ").replace("`", " "))
        bad += [f"{wf}: workflow {w}" for w in toks[3].split("+") if wfs and w not in wfs]
        bad += [f"{wf}: {t}" for t in toks[4:] if t.startswith("--") and _norm(t) not in opts]
    assert bad == []


def test_build_is_deterministic():
    second = reg.build()
    assert reg.dump(BUILT[0]) == reg.dump(second[0]) and reg.dump(BUILT[1]) == reg.dump(second[1])
    assert BUILT[2] == second[2]


def test_path_args_skip_names_and_keep_placeholders():
    cmd = "sos run pipeline/m.ipynb qc --genoFile a/b.bed --name x/y --ref <path/to/ref> --cwd out/qc"
    assert reg.path_args(cmd) == [("--genoFile", "a/b.bed"), ("--ref", "<path/to/ref>"), ("--cwd", "out/qc")]
