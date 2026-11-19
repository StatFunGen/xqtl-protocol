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

Inputs and outputs (declared, never guessed):
  * Each selected cell carries ``"builder": {"inputs": {...}, "outputs": {...}}`` in
    its metadata. An input is keyed by its option name and has a ``kind``, the
    ``path`` used in the command and, for files made by another step, a ``source``
    ``<module>.<workflow>.<output>`` (``user`` for placeholders). An output has a
    ``kind`` and a ``path``. An optional ``after`` list names steps (``module.workflow``)
    whose files the command reads without naming them, e.g. a config in a shared --cwd.
    ``source: setup`` marks a file the cell's own setup lines
    write before the command (shell lines above ``sos run`` in a tagged cell are kept).
  * Checks: every file path in the command is declared and every declared path is in
    the command; a ``source`` names an existing output of another builder step, the
    input sits at or under that output's path, and the kinds agree (a directory
    output accepts any file under it); every ``output/...`` input has a ``source``
    unless it is listed in website/builder/known_gaps.json; every
    ``tests/fixtures/...`` path exists.

Notebooks are read as JSON and never executed. Output is deterministic.

Usage:
  python website/builder_registry.py --check   # report, exit 1 on any error
  python website/builder_registry.py --write   # also update the builder notebook
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / "website" / "builder" / "workflows.json"
KNOWN_GAPS = REPO / "website" / "builder" / "known_gaps.json"
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
        tags = list(meta.get("tags", []))
        for k, m in enumerate(CMD_RE.finditer(text)):
            if m.group("nb") != path.stem:
                continue
            cmd = re.sub(r"\s+", " ", m.group(0)).strip()
            wf = m.group("wf").split(":")[0]
            # a builder cell may prepare its inputs with shell lines before the command
            setup = [l.strip() for l in text[:m.start()].splitlines()
                     if l.strip() and not l.strip().startswith("#")] if k == 0 else []
            if setup and any(t == "builder" or t.startswith("builder:") for t in tags):
                cmd = "\n".join(setup + [cmd])
            found.append(Candidate(path.stem, i, wf, cmd, tags, meta.get("builder", {})))
    return found


def select(cands: list[Candidate], module: str, wf: str, variant: str = "") -> Candidate:
    # exact workflow token first; otherwise a combined run that starts with it
    same = ([c for c in cands if c.workflow == wf]
            or [c for c in cands if c.workflow.split("+")[0] == wf]
            # a chained example such as qtl_dataset_construct+susie_twas serves its last workflow
            or [c for c in cands if "+" in c.workflow and c.workflow.split("+")[-1] == wf])
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


def sos_line(command: str) -> str:
    """The `sos run` line of a command that may start with setup lines."""
    return command.split("\n")[-1].lstrip("# ")


def setup_lines(command: str) -> str:
    return "\n".join(command.split("\n")[:-1])


def path_args(command: str) -> list[tuple[str, str]]:
    """(option, value) for every file-like value in a command (output folders included)."""
    toks = shlex.split(sos_line(command).replace("`", " "))[4:]
    out, flag = [], None
    for t in toks:
        if t.startswith("--"):
            flag = t
            continue
        if t.startswith("-") and len(t) == 2:
            flag = None
            continue
        if not flag or flag in NOT_PATHS:
            continue
        if ("/" in t or t.startswith("<")) and not t.startswith(("http", "s/")) and '"' not in t:
            out.append((flag, t))
    return out


def io_of(c: Candidate) -> dict:
    """The declared inputs and outputs as the builder shows them."""
    ins = [v["path"] for v in c.meta.get("inputs", {}).values()]
    outs = [v["path"] for v in c.meta.get("outputs", {}).values()]
    return {"i": list(dict.fromkeys(ins)), "o": list(dict.fromkeys(outs))}


def check_io(selected: dict, gaps: set) -> list[str]:
    """Declared I/O agrees with the commands and every upstream link resolves."""
    errors, used_gaps = [], set()
    outputs = {f"{m}.{wf}.{name}": o for (m, wf), c in selected.items()
               for name, o in c.meta.get("outputs", {}).items()}
    for (m, wf), c in selected.items():
        key = f"{m}|{wf}"
        if not c.meta.get("outputs") and not c.meta.get("inputs"):
            errors.append(f"{key}: {c.where} has no declared inputs/outputs")
            continue
        for a in c.meta.get("after", []):
            # a step that reads an earlier step's files without naming them on the command
            # line (e.g. a config written into the shared --cwd) lists that step here
            if tuple(a.split(".", 1)) not in selected:
                errors.append(f"{key}: after names {a}, which is not a builder step")
        args = path_args(c.command)
        in_cmd = {v for f, v in args if f not in OUT_FLAGS}
        out_cmd = {v.rstrip("/") for f, v in args if f in OUT_FLAGS}
        declared = {d["path"] for d in c.meta.get("inputs", {}).values()}
        for v in sorted(in_cmd - declared):
            errors.append(f"{key}: {v} is in the command but not declared as an input ({c.where})")
        for v in sorted(declared - in_cmd):
            errors.append(f"{key}: declared input {v} is not in the command ({c.where})")
        for v in sorted(out_cmd - {o["path"] for o in c.meta.get("outputs", {}).values()}):
            errors.append(f"{key}: output folder {v} is not declared ({c.where})")
        for f in re.findall(r"tests/fixtures/[\w./-]+", setup_lines(c.command)):
            if not (REPO / f).exists():
                errors.append(f"{key}: fixture {f} in the setup lines does not exist")
        for name, d in c.meta.get("inputs", {}).items():
            path, src = d["path"], d.get("source")
            if path.startswith("tests/fixtures/") and not any(ch in path for ch in "$*{<") \
                    and not (REPO / path).exists():
                errors.append(f"{key}: fixture {path} does not exist")
            if src == "setup":
                st = setup_lines(c.command)
                if path not in st and path.rsplit("/", 1)[0] + "/" not in st:
                    errors.append(f"{key}: input {name} is declared as made by the setup lines, "
                                  f"but they do not mention {path}")
            elif src and src != "user":
                o = outputs.get(src)
                if o is None:
                    errors.append(f"{key}: input {name} names {src}, which is not a builder step output")
                elif not (path == o["path"] or path.startswith(o["path"] + "/")):
                    errors.append(f"{key}: input {name} ({path}) is not under {src} ({o['path']})")
                elif o["kind"] != "directory" and o["kind"] != d["kind"]:
                    errors.append(f"{key}: input {name} is {d['kind']} but {src} is {o['kind']}")
            elif path.startswith("output/") and not src:
                if (key, path) in gaps:
                    used_gaps.add((key, path))
                else:
                    errors.append(f"{key}: input {name} ({path}) is made by no builder step; "
                                  "link it with a source or fix the route")
    for k, p in sorted(gaps - used_gaps):
        errors.append(f"known_gaps.json: {k} {p} no longer applies; remove the entry")
    return errors


def build():
    want = json.loads(WORKFLOWS.read_text(encoding="utf-8"))
    paths = notebooks()
    cmdlib, wio, errors, selected = {}, {}, [], {}
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
            # a variant of a workflow already listed (e.g. TensorQTL cis + cis:interaction)
            # gets its own key, so both commands are kept
            key = f"{wf}:{variant}" if variant and (module, wf) in selected else wf
            cmdlib.setdefault(module, []).append({"wf": key, "cmd": c.command})
            wio[f"{module}|{key}"] = io_of(c)
            selected[(module, key)] = c
    gaps = {(g["step"], g["input"]) for g in json.loads(KNOWN_GAPS.read_text(encoding="utf-8"))["gaps"]}
    errors += check_io(selected, gaps)
    return cmdlib, wio, errors, selected


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
    cmdlib, wio, errors, _ = build()
    n = sum(len(v) for v in cmdlib.values())
    print(f"{n} commands resolved; {len(errors)} errors")
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
