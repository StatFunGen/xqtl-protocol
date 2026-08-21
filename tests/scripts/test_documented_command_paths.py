"""Validate paths in commands that users can copy from active documentation."""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
SOS = REPO / "code" / "SoS"
LANDING = SOS / "xqtl_protocol_landing_page.html"
FIXTURE_PATH = re.compile(r"tests/fixtures/[A-Za-z0-9_.@+/*?\[\]{}-]+")
STALE_INPUT = re.compile(r"(?<![A-Za-z0-9_./-])input/[A-Za-z0-9_.@+/*?\[\]{}-]+")
LOCAL_ABSOLUTE = re.compile(r"(?:/restricted/projectnb/|/projectnb/|/Users/|/home/|/gpfs/|/mnt/|/scratch/|~/)")
DOC_STEM = re.compile(r'data-doc="[^"]*/([^/"]+)\.html"')
JSON_CMD = re.compile(r'"cmd":("(?:\\.|[^"\\])*")')

def _active_notebooks():
    stems = set(DOC_STEM.findall(LANDING.read_text()))
    found = []
    for stem in sorted(stems):
        found += [p for p in SOS.rglob(f"{stem}.ipynb") if "graveyard" not in p.parts and ".ipynb_checkpoints" not in p.parts]
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
    return [json.loads(m.group(1)) for m in JSON_CMD.finditer(LANDING.read_text())]

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
    stems = set(DOC_STEM.findall(LANDING.read_text()))
    resolved = {path.stem for path in _active_notebooks()}
    assert stems <= resolved, "Missing active notebooks: " + ", ".join(sorted(stems - resolved))
