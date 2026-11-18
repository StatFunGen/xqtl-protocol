#!/usr/bin/env python3
"""Build the Workflow Builder's command and file lists from the module pages.

The module notebooks are the single source of truth. For every workflow the builder
offers (website/builder/workflows.json), this script finds the example command on
the module page, checks it, and writes two JSON objects into the builder notebook:

  CMDLIB  {module: [{"wf": workflow, "cmd": command}, ...]}
  WIO     {"module|workflow": {"i": [input paths], "o": [output paths]}}

Selecting the command (fails closed, never guesses):
  * A code cell tagged ``builder`` (or ``builder:<variant>`` for a named variant)
    supplies the command for the workflows it runs.
  * Without a tag, a workflow with exactly one example on its page is used.
  * Several untagged examples, several tagged ones, or none at all is an error
    that names the notebook, the cell and the workflow.

Inputs and outputs:
  * Declared in the tagged cell's metadata, ``"builder": {"inputs": {...},
    "outputs": {...}}``, each entry with a ``kind`` and either ``source`` (``user``
    or ``<module>.<workflow>.<output>``) or ``path``.
  * Otherwise inferred from the command's options for display only, and reported
    as inferred.

Notebooks are read as JSON and never executed. Output is deterministic.

Usage:
  python website/builder_registry.py --check   # report, exit 1 on any error
  python website/builder_registry.py --write   # also update the builder notebook
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / "website" / "builder" / "workflows.json"
BUILDER = REPO / "code" / "SoS" / "xqtl_protocol_workflow_builder.ipynb"
SKIP_DIRS = ("graveyard", "misc")
# a commented-out example ("# sos run ...") is kept commented
CMD_RE = re.compile(r"(?:#\s*)?sos run pipeline/(?P<nb>[\w]+)\.ipynb\s+(?P<wf>[\w+:]+)[^\n]*")
OUT_FLAGS = {"--cwd", "--output-dir", "--wd", "--output_dir"}
NOT_PATHS = {"--name", "--container", "--modular-script-dir", "--modular_script_dir",
             "--study", "--context", "--genome", "--chrom"}


@dataclass
class Candidate:
    notebook: str
    cell: int
    workflow: str          # as written, e.g. "qtl_dataset_construct+susie_twas"
    command: str
    tags: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def variants(self):
        return {t.split(":", 1)[1] if ":" in t else "" for t in self.tags
                if t == "builder" or t.startswith("builder:")}

    @property
    def where(self):
        return f"{self.notebook} cell {self.cell}"


class RegistryError(Exception):
    pass


def notebooks():
    out = {}
    for f in sorted((REPO / "code" / "SoS").rglob("*.ipynb")):
        if any(d in f.parts for d in SKIP_DIRS) or f.name.startswith("xqtl_protocol_"):
            continue
        out.setdefault(f.stem, f)
    return out


def candidates(path: Path) -> list[Candidate]:
    """Every `sos run pipeline/<this module>.ipynb <workflow>` command in the code cells."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", "")).replace("\\\n", " ")
        meta = cell.get("metadata", {})
        for m in CMD_RE.finditer(text):
            if m.group("nb") != path.stem:
                continue
            cmd = re.sub(r"\s+", " ", m.group(0)).strip()
            wf = m.group("wf").split(":")[0]
            found.append(Candidate(path.stem, i, wf, cmd, list(meta.get("tags", [])), meta.get("builder", {})))
    return found


def select(cands: list[Candidate], module: str, wf: str, variant: str = "") -> Candidate:
    # exact workflow token first; otherwise a combined run that starts with it
    same = [c for c in cands if c.workflow == wf] or [c for c in cands if c.workflow.split("+")[0] == wf]
    tagged = [c for c in same if variant in c.variants]
    if len(tagged) == 1:
        return tagged[0]
    if len(tagged) > 1:
        raise RegistryError(f"{module}|{wf}: {len(tagged)} cells tagged for this workflow "
                            f"({', '.join(c.where for c in tagged)}); keep one tag")
    if variant:
        raise RegistryError(f"{module}|{wf}: no cell tagged builder:{variant}")
    # cells tagged only for a named variant are not candidates for the plain workflow
    untagged = [c for c in same if not c.variants]
    if len(untagged) == 1:
        return untagged[0]
    if not same:
        raise RegistryError(f"{module}|{wf}: no example command on the module page")
    raise RegistryError(f"{module}|{wf}: {len(untagged)} untagged examples on the page "
                        f"({', '.join(c.where for c in untagged)}); tag the builder's one with 'builder'")


def io_of(c: Candidate) -> tuple[dict, bool]:
    """Declared inputs/outputs, or ones inferred from the options (second value: inferred?)."""
    if c.meta.get("inputs") or c.meta.get("outputs"):
        ins = [v.get("path") or v.get("source") for v in c.meta.get("inputs", {}).values()]
        outs = [v.get("path") for v in c.meta.get("outputs", {}).values()]
        return {"i": [x for x in ins if x], "o": [x for x in outs if x]}, False
    toks = c.command.split()
    ins, outs, flag = [], [], None
    for t in toks[4:]:
        if t.startswith("-"):
            flag = t
            continue
        if not flag or flag in NOT_PATHS or "/" not in t or t.startswith(("`", "http", "<")):
            continue
        (outs if flag in OUT_FLAGS else ins).append(t)
    return {"i": ins, "o": outs}, True


def build():
    want = json.loads(WORKFLOWS.read_text(encoding="utf-8"))
    paths = notebooks()
    cmdlib, wio, errors, inferred = {}, {}, [], []
    cache = {}
    for module, wfs in want.items():
        if module not in paths:
            errors.append(f"{module}: no module notebook found")
            continue
        cands = cache.setdefault(module, candidates(paths[module]))
        for entry in wfs:
            wf, variant = (entry, "") if isinstance(entry, str) else (entry["wf"], entry.get("variant", ""))
            try:
                c = select(cands, module, wf, variant)
            except RegistryError as e:
                errors.append(str(e))
                continue
            cmdlib.setdefault(module, []).append({"wf": wf, "cmd": c.command})
            io, guessed = io_of(c)
            wio[f"{module}|{wf}"] = io
            if guessed:
                inferred.append(f"{module}|{wf}")
    return cmdlib, wio, errors, inferred


def dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def replace_json(text: str, name: str, obj) -> str:
    start = text.index(name + "=") + len(name) + 1
    _, n = json.JSONDecoder().raw_decode(text[start:])
    return text[:start] + dump(obj) + text[start + n:]


def write(cmdlib, wio):
    raw = BUILDER.read_text(encoding="utf-8")
    nb = json.loads(raw)
    cell = nb["cells"][0]
    s = "".join(cell["source"])
    s = replace_json(replace_json(s, "CMDLIB", cmdlib), "WIO", wio)
    lines = s.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    indent = 1 if raw.startswith('{\n "') else 2
    BUILDER.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--diff", action="store_true", help="list commands that differ from the builder's current ones")
    args = ap.parse_args(argv)
    cmdlib, wio, errors, inferred = build()
    n = sum(len(v) for v in cmdlib.values())
    print(f"{n} commands resolved; {len(errors)} errors; {len(inferred)} with inferred inputs/outputs")
    for e in errors:
        print("ERROR", e)
    if args.diff:
        s = "".join(json.loads(BUILDER.read_text(encoding="utf-8"))["cells"][0]["source"])
        i = s.index("CMDLIB=") + 7
        old, _ = json.JSONDecoder().raw_decode(s[i:])
        oldc = {(m, e["wf"]): e["cmd"] for m, l in old.items() for e in l}
        changed = [k for m, l in cmdlib.items() for e in l
                   if oldc.get(k := (m, e["wf"])) is not None and re.sub(r"\s+", " ", oldc[k]).strip() != e["cmd"]]
        print(f"{len(changed)} resolved commands differ from the builder's current ones")
        for m, wf in changed:
            print("  DIFF", f"{m}|{wf}")
    if errors:
        return 1
    if args.write:
        write(cmdlib, wio)
        print(f"wrote CMDLIB and WIO into {BUILDER.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
