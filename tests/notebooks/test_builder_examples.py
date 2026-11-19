"""Notebook tier: run every Workflow Builder command exactly as the builder shows it.

The commands come from the builder registry (website/builder_registry.py), which reads
them from the module pages' example cells. Each runs unchanged from a scratch folder that
links pipeline/, code/ and tests/ from the repository, so a pass here means the command a
user copies from the builder runs on the example data.

Steps are grouped by their declared inputs: a step whose input names another step's
output (``source: module.workflow.output``) runs after that step, in the same folder.
Each connected group is one test, in its own class so pytest-xdist can spread the groups
over workers.

Steps listed in website/builder/untested_steps.json are skipped, each with a reason there
(large reference downloads, software not in the environment, unfinished modules). A step
that needs a skipped step's output must be skipped too; the test below checks that.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.sos_runner import _worker_env, sos_bin

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "website"))
import builder_registry as reg  # noqa: E402

UNTESTED = REPO / "website" / "builder" / "untested_steps.json"


def _plan():
    cmdlib, wio, errors, selected = reg.build()
    assert not errors, errors
    skip = set(json.loads(UNTESTED.read_text(encoding="utf-8"))["steps"])
    order = [(m, e["wf"]) for m, entries in cmdlib.items() for e in entries]
    cmds = {(m, e["wf"]): e["cmd"] for m, entries in cmdlib.items() for e in entries}
    deps = {}
    for key in order:
        ups = set()
        for d in selected[key].meta.get("inputs", {}).values():
            src = d.get("source")
            if src and src not in ("user", "setup"):
                m, rest = src.split(".", 1)
                ups.add((m, rest.rsplit(".", 1)[0]))
        ups |= {tuple(a.split(".", 1)) for a in selected[key].meta.get("after", [])}
        ups.discard(key)
        deps[key] = ups
    # connected groups of runnable steps (union-find over the dependency edges)
    parent = {k: k for k in order}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for k in order:
        for u in deps[k]:
            if u in parent:
                parent[find(k)] = find(u)
    groups = {}
    for k in order:
        if f"{k[0]}|{k[1]}" not in skip:
            groups.setdefault(find(k), []).append(k)
    plans = []
    for members in groups.values():
        done, seq, todo = set(), [], list(members)
        while todo:                                   # topological, registry order on ties
            ready = [k for k in todo if all(u in done or u not in members for u in deps[k])]
            assert ready, f"dependency cycle among {todo}"
            for k in ready:
                seq.append(k); done.add(k); todo.remove(k)
        plans.append(seq)
    return plans, cmds, wio, deps, skip


PLANS, CMDS, WIO, DEPS, SKIP = _plan()


def test_skipped_steps_have_no_runnable_dependents():
    bad = [f"{m}|{wf} needs {u[0]}|{u[1]}" for (m, wf), ups in DEPS.items()
           if f"{m}|{wf}" not in SKIP for u in ups if f"{u[0]}|{u[1]}" in SKIP]
    assert bad == [], "steps that need a skipped step must be listed in untested_steps.json too"


def _repo_state():
    """Untracked and changed files under the repo's tests/ and code/ folders."""
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        return None
    return subprocess.run([git, "status", "--porcelain", "--untracked-files=all", "--", "tests", "code"],
                          cwd=REPO, capture_output=True, text=True).stdout


def _task_errors(tmp_path, n=3, lines=25):
    """The end of the newest step .stderr files, where SoS leaves the real error."""
    errs = sorted((f for f in tmp_path.rglob("*.stderr") if f.stat().st_size > 0),
                  key=lambda f: f.stat().st_mtime)[-n:]
    return "".join(f"\n--- {f.relative_to(tmp_path)}\n"
                   + "\n".join(f.read_text(errors="replace").splitlines()[-lines:]) for f in errs)


def _run_group(steps, tmp_path):
    for d in ("pipeline", "tests"):
        (tmp_path / d).symlink_to(REPO / d)
    # code/ is mirrored as real folders holding links to the files, not linked as a whole:
    # a workflow that runs inside code/... and writes to ../../../output must land in this
    # scratch folder, as it would in a checkout, not next to the repository's own code/.
    for root, dirs, files in os.walk(REPO / "code"):
        rel = Path(root).relative_to(REPO)
        (tmp_path / rel).mkdir(exist_ok=True)
        for f in files:
            (tmp_path / rel / f).symlink_to(Path(root) / f)
        for d in [d for d in dirs if (Path(root) / d).is_symlink()]:
            (tmp_path / rel / d).symlink_to(Path(root) / d)
            dirs.remove(d)
    env = _worker_env() or dict(os.environ)
    before = _repo_state()
    for m, wf in steps:
        script = CMDS[(m, wf)].replace("sos run ", f"{sos_bin()} run ")
        p = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path,
                           capture_output=True, text=True, timeout=2400, env=env)
        assert p.returncode == 0, f"{m}|{wf}\n{script}\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}{_task_errors(tmp_path)}"
        for o in WIO.get(f"{m}|{wf}", {}).get("o", []):
            o = tmp_path / o
            # any result file counts, including an empty `.done` marker; SoS logs do not
            made = [f for f in o.rglob("*") if f.is_file()
                    and not f.name.endswith((".stderr", ".stdout", ".sh"))] if o.is_dir() else ([o] if o.exists() else [])
            assert made, f"{m}|{wf}: {o} has no output files"
        # a command must write under output/, never next to the example files it reads
        assert _repo_state() == before, f"{m}|{wf} wrote into the repository:\n{_repo_state()}"


def _make(steps):
    name = "_".join(f"{m}_{wf}" for m, wf in steps[:1]).replace("+", "_").replace(":", "_")

    @pytest.mark.skipif(shutil.which(sos_bin()) is None, reason="needs sos")
    def test_runs(self, tmp_path):
        _run_group(steps, tmp_path)

    test_runs.__doc__ = " -> ".join(f"{m}|{wf}" for m, wf in steps)
    return type(f"TestBuilder_{name}", (), {"test_runs": test_runs})


for _steps in PLANS:
    _cls = _make(_steps)
    globals()[_cls.__name__] = _cls
del _steps, _cls
