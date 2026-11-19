"""Validate paths in commands that users can copy from active documentation."""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
SOS = REPO / "code" / "SoS"
LANDING = SOS / "xqtl_protocol_workflow_builder.html"
FIXTURE_PATH = re.compile(r"tests/fixtures/[A-Za-z0-9_.@+/*?\[\]{}-]+")
STALE_INPUT = re.compile(r"(?<![A-Za-z0-9_./-])input/[A-Za-z0-9_.@+/*?\[\]{}-]+")
LOCAL_ABSOLUTE = re.compile(r"(?:/restricted/projectnb/|/projectnb/|/Users/|/home/|/gpfs/|/mnt/|/scratch/|~/)")
# Jupyter Book 2 publishes slugified, extensionless page URLs, so the builder page
# links to /reference-data-preparation rather than /reference_data_preparation.html.
DOC_MAP = re.compile(r"var DOC = (\{.*?\});")
DATA = re.compile(r'<script type="application/json" id="xqb-data">(.*?)</script>', re.S)

def _myst_slug(name):
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")

def _documented_slugs():
    m = DOC_MAP.search(LANDING.read_text())
    assert m, f"No module page links (var DOC) found in {LANDING.name}; the link scheme changed."
    return {v for v in json.loads(m.group(1)).values() if v}

def _notebooks_by_slug():
    by_slug = {}
    for path in SOS.rglob("*.ipynb"):
        if "graveyard" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        by_slug.setdefault(_myst_slug(path.name), []).append(path)
    return by_slug

def _active_notebooks():
    by_slug = _notebooks_by_slug()
    found = []
    for slug in sorted(_documented_slugs()):
        found += by_slug.get(slug, [])
    return sorted(set(found))

def _notebook_commands(path):
    notebook = json.loads(path.read_text())
    commands = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if re.search(r"(?m)^\s*sos\s+run\b", source):
            commands.append(source)
    return commands

def _landing_commands():
    data = json.loads(DATA.search(LANDING.read_text()).group(1).replace("<\\/", "</"))
    return [data["cmd"][k] for k in sorted(data["cmd"])]

def _cases():
    cases = []
    for notebook in _active_notebooks():
        for index, command in enumerate(_notebook_commands(notebook), 1):
            name = f"{notebook.relative_to(REPO)}::command-{index}"
            cases.append(pytest.param(name, command, id=name))
    for index, command in enumerate(_landing_commands(), 1):
        cases.append(pytest.param(f"landing::command-{index}", command, id=f"landing-{index}"))
    return cases

def _fixture_exists(path_text):
    candidate = REPO / path_text
    return bool(glob.glob(str(candidate))) if glob.has_magic(path_text) else candidate.exists()

@pytest.mark.parametrize("source,command", _cases())
def test_documented_command_paths(source, command):
    failures = [f"stale path {p!r}" for p in sorted(set(STALE_INPUT.findall(command)))]
    failures += [f"missing fixture {p!r}" for p in sorted(set(FIXTURE_PATH.findall(command))) if not _fixture_exists(p)]
    local = LOCAL_ABSOLUTE.search(command)
    if local:
        failures.append(f"machine-local absolute path containing {local.group(0)!r}")
    assert not failures, source + ":\n" + "\n".join(f"- {item}" for item in failures)

def test_active_notebook_links_resolve():
    slugs = _documented_slugs()
    resolved = set(_notebooks_by_slug())
    assert slugs <= resolved, "Missing active notebooks: " + ", ".join(sorted(slugs - resolved))
