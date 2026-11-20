#!/usr/bin/env python3
"""Build the Workflow Builder's command and file lists from the module pages.

The module notebooks are the single source of truth. For every workflow the builder
offers (website/builder/workflows.json), this script finds the example command on
the module page and checks it. With --write it assembles the builder page into the
builder notebook from website/builder/page.html (template), page.js (logic), page.css
and page_content.json (method choices, intros, example outputs), with this data:

  cmd       {"module|workflow": command}
  io        {"module|workflow": {"i": [input paths], "o": [output paths]}}
  untested  steps not run on the example data, with their reasons (untested_steps.json)
  prev      the first rows of each example input file the commands read
  src, opt  each input's declared producing step and the option it is passed to
  ext       the inputs of each step that the user supplies (no producing step, no setup line)

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
PAGE = REPO / "website" / "builder"
UNTESTED = PAGE / "untested_steps.json"
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


PREVIEW_ROWS, PREVIEW_COLS = 3, 6


def preview(path: str):
    """The header and first rows of a text example file, or None for binary files and folders."""
    import gzip
    f = REPO / path
    if not f.is_file():
        return None
    # PLINK genotypes are binary; show the variant list (.bim) that sits next to them
    if f.suffix == ".bed" and f.with_suffix(".bim").is_file():
        out = preview(str(f.with_suffix(".bim").relative_to(REPO)))
        if out:  # .bim has no header line: name its six columns
            out["rows"] = [["chromosome", "variant", "cM", "position", "allele 1", "allele 2"][:out["cols"]]] + out["rows"][:PREVIEW_ROWS]
            out["via"] = f.with_suffix(".bim").name
        return out
    try:
        opener = gzip.open if f.suffix in (".gz", ".bgz") else open
        with opener(f, "rt", encoding="utf-8") as h:
            lines = []
            for line in h:
                if "\x00" in line:
                    return None
                if line.strip():
                    lines.append(line.rstrip("\n\r"))
                if len(lines) > PREVIEW_ROWS:
                    break
    except (UnicodeDecodeError, OSError, EOFError):
        return None
    if not lines:
        return None
    sep = "\t" if "\t" in lines[0] else ("," if f.name.endswith((".csv", ".csv.gz")) else None)
    rows = [l.split(sep) for l in lines]
    return {"rows": [r[:PREVIEW_COLS] for r in rows], "cols": len(rows[0])}


FIXTURE_SKIP = (".tbi", ".bai", ".csi", ".log", ".md", "expected_manifest.tsv")


TEST_FILE_RE = re.compile(r"[\"']([\w.\-]+\.(?:gz|tsv|txt|bed|csv|rds|bim|fam|vcf|list|gct))[\"']")


def consumer_example(module: str, option: str):
    """A preview of the fixture the module's own tests pass to this option, or None."""
    key = module.lower().replace("_", "")
    pats = [f'"--{option}"', f'"--{option.replace("_", "-")}"', f"'--{option}'"]
    for test in sorted((REPO / "tests").rglob("test_*.py")):
        if key not in test.stem.lower().replace("_", ""):
            continue
        text = test.read_text(encoding="utf-8")
        for pat in pats:
            for hit in re.finditer(re.escape(pat), text):
                name = TEST_FILE_RE.search(text[hit.end():hit.end() + 200])
                if name and (f := next((REPO / "tests" / "fixtures").rglob(name.group(1)), None)):
                    if (out := _fixture_preview(f)):
                        return out
    return None


def fixture_example(path: str, source: str):
    """A preview of the test fixture that shows what a step's output looks like, or None.

    Looks in tests/fixtures/<module>/ (expected outputs first) for a file whose name ends the
    same way as the output, e.g. ...Marchenko_PC.gz or ...phenotype_by_chrom_files.txt. PLINK
    genotypes (.bed) match the .bim of the same workflow, shown as its variant list."""
    module, workflow = source.split(".")[:2]
    root = REPO / "tests" / "fixtures" / module.lower()
    if not root.is_dir():
        return None
    files = sorted((f for f in root.rglob("*") if f.is_file() and not f.name.endswith(FIXTURE_SKIP)),
                   key=lambda f: ("expected" not in f.parts, str(f)))
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".bed") and not name.endswith((".bed.gz",)):
        bims = [f for f in files if f.suffix == ".bim"]
        pick = [f for f in bims if workflow in f.name] or bims
        return _fixture_preview(pick[0]) if pick else None
    toks = name.split(".")

    def shared(f):  # how many trailing dot-separated parts match
        t, n = f.name.split("."), 0
        while n < min(len(t), len(toks)) and t[-1 - n] == toks[-1 - n]:
            n += 1
        return n
    best = max(files, key=shared, default=None)
    if best is None or shared(best) < 2:
        return None
    return _fixture_preview(best)


def _fixture_preview(f: Path):
    rel = str(f.relative_to(REPO))
    out = preview(rel)
    if out is not None and f.suffix == ".bim":  # no header line: name its six columns
        out["rows"] = [["chromosome", "variant", "cM", "position", "allele 1", "allele 2"][:out["cols"]]] + out["rows"][:PREVIEW_ROWS]
    if out is not None:
        out["ex"] = f.name
    return out


def _src_step(source: str) -> str:
    """'module.workflow.output' -> 'module|workflow'."""
    module, rest = source.split(".", 1)
    return f"{module}|{rest.rsplit('.', 1)[0]}"


def page_data(cmdlib, wio, selected):
    """Everything the page shows that comes from the repository."""
    untested = json.loads(UNTESTED.read_text(encoding="utf-8"))
    content = json.loads((PAGE / "page_content.json").read_text(encoding="utf-8"))
    fixtures = sorted({p for io in wio.values() for p in io["i"] if p.startswith("tests/fixtures/")})
    prev = {p: v for p in fixtures if (v := preview(p))}
    # files made by an earlier step: show an example from the test fixtures, either the file the
    # reading module's own tests pass to that option, or the making module's expected output
    for (m, wf), c in sorted(selected.items()):
        for opt, v in c.meta.get("inputs", {}).items():
            src, path = v.get("source"), v.get("path", "")
            if path in prev or src in (None, "user", "setup") or not path.startswith("output/"):
                continue
            if (ex := consumer_example(m, opt) or fixture_example(path, src)):
                prev[path] = ex
    data = {"cmd": {f"{m}|{e['wf']}": e["cmd"] for m, l in cmdlib.items() for e in l}, "io": wio,
            "untested": untested["steps"], "untestedWhy": untested["reasons"], "prev": prev,
            "after": {f"{m}|{wf}": [a.replace(".", "|", 1) for a in c.meta.get("after", [])]
                      for (m, wf), c in sorted(selected.items()) if c.meta.get("after")},
            # where each input comes from: the declared producing step ("module|workflow")
            "src": {f"{m}|{wf}": srcs for (m, wf), c in sorted(selected.items())
                    if (srcs := {v["path"]: _src_step(v["source"]) for v in c.meta.get("inputs", {}).values()
                                 if v.get("source") not in (None, "user", "setup")})},
            # the option each input is given to, to label it on the page
            "opt": {f"{m}|{wf}": {v["path"]: name for name, v in c.meta.get("inputs", {}).items()}
                    for (m, wf), c in sorted(selected.items()) if c.meta.get("inputs")},
            # inputs a user supplies: declared without a producing step and not written by setup lines
            "ext": {f"{m}|{wf}": paths for (m, wf), c in sorted(selected.items())
                    if (paths := [v["path"] for v in c.meta.get("inputs", {}).values()
                                  if v.get("source", "user") == "user"])}}
    data.update({k: content[k] for k in ("meth", "intro", "bench", "outPrev", "terms")})
    return data


HEADER = "# xQTL Analysis Workflow Builder\n\n"
# In the site's frame: hide the duplicate title and open links in the main window.
EMBED = ("if(window.self!==window.top){document.documentElement.classList.add('xq-embedded');"
         "var b=document.createElement('base');b.target='_top';document.head.appendChild(b);}")


def page_template() -> str:
    """page.html as XML, so the browser keeps <sc-for> inside table rows where it was written."""
    t = re.sub(r"<!--.*?-->", "", (PAGE / "page.html").read_text(encoding="utf-8"), flags=re.S)
    return re.sub(r"<(input|br|img|hr)\b([^>]*?)\s*/?>", r"<\1\2 />", t).strip()


def render_cell(cmdlib, wio, selected) -> str:
    """The builder notebook's one markdown cell: a short intro and the page as raw HTML."""
    data = dump(page_data(cmdlib, wio, selected)).replace("</", "<\\/")
    tpl = page_template()
    css = (PAGE / "page.css").read_text(encoding="utf-8").strip()
    js = (PAGE / "page.js").read_text(encoding="utf-8").strip()
    html = ('<div class="xqb" id="xqb"></div>\n'
            f'<script type="text/xml" id="xqb-tpl"><root>{tpl}</root></script>\n'
            f'<script type="application/json" id="xqb-data">{data}</script>\n'
            f"<style>\n{css}\n</style>\n<script>{EMBED}</script>\n<script>\n{js}\n</script>")
    return HEADER + "````{raw} html\n" + html + "\n````\n"


def write(cmdlib, wio, selected):
    raw = BUILDER.read_text(encoding="utf-8")
    nb = json.loads(raw)
    lines = render_cell(cmdlib, wio, selected).split("\n")
    nb["cells"][0]["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    indent = 1 if raw.startswith('{\n "') else 2
    BUILDER.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")


def committed_cell() -> str:
    return "".join(json.loads(BUILDER.read_text(encoding="utf-8"))["cells"][0]["source"])


def committed_data():
    """The data object in the committed builder notebook."""
    s = committed_cell()
    i = s.index('id="xqb-data">') + len('id="xqb-data">')
    return json.loads(s[i:s.index("</script>", i)].replace("<\\/", "</"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--diff", action="store_true", help="list commands that differ from the builder's current ones")
    args = ap.parse_args(argv)
    cmdlib, wio, errors, selected = build()
    n = sum(len(v) for v in cmdlib.values())
    print(f"{n} commands resolved; {len(errors)} errors")
    for e in errors:
        print("ERROR", e)
    if args.diff:
        try:
            oldc = committed_data()["cmd"]
        except ValueError:
            oldc = {}
        changed = [k for m, l in cmdlib.items() for e in l
                   if (k := f"{m}|{e['wf']}") in oldc and oldc[k] != e["cmd"]]
        print(f"{len(changed)} resolved commands differ from the builder's current ones")
        for k in changed:
            print("  DIFF", k)
    if errors:
        return 1
    if args.write:
        write(cmdlib, wio, selected)
        print(f"wrote the builder page into {BUILDER.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
