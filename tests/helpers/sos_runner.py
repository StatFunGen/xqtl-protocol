"""Run SoS notebook steps as subprocesses (the production entrypoint).

A notebook test drives ``sos run <notebook> <step> --params`` exactly as a user
would, so it covers the SoS cell orchestration (manifest/param wiring, path
resolution, step chaining) that the direct script tests can't.

SoS writes each step's tool output to per-step ``<output>.stderr`` files (declared
in the cells as ``stderr = f'{_output}.stderr'``) on EVERY run, but its own process
output only *references* those paths on failure without printing them — so the real
R/bash traceback is invisible in CI. ``run_sos`` therefore reads those ``.stderr``
files back and appends their content to the returned ``stderr`` so the actual error
surfaces in the pytest assertion message (and in the CI log).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MAX_PER_FILE = 8000            # cap each .stderr file in the appended dump
_MAX_TOTAL = 40000             # overall cap so a runaway log can't bloat the message

# SoS binds its ZMQ controller to the host's routable IP (utils.get_localhost_ip),
# which on a CI runner is intermittently unassignable -> the run aborts before any
# step with a ZMQError. Pure infra flake, not a test failure, so it's safe to retry.
_ZMQ_FLAKE_RE = re.compile(
    r"ZMQError|Can't assign requested address|Address already in use|Cannot bind",
    re.IGNORECASE,
)


def sos_bin() -> str:
    return os.environ.get("XQTL_SOS") or "sos"


def _worker_env():
    """Isolate the SoS runtime per pytest-xdist worker so the notebook tier is
    safe to run in parallel.

    SoS hardcodes its runtime tree under ``expanduser("~")`` — the per-workflow
    ``exec_dir`` (``~/.sos/<md5(cwd)>``) plus the shared ``~/.sos/{signatures,
    tasks,workflows}`` — with no SoS-specific env override, and the process
    ``cwd`` is load-bearing here (tests set it to the repo root for relative
    ``code/script`` resolution), so cwd can't be the isolation seam. HOME is the
    only lever that fully separates two concurrent ``sos run`` processes.

    We isolate PER WORKER, not per call: under ``--dist loadscope`` a test file
    runs entirely on one worker and a worker runs its tests serially, so a
    per-worker HOME removes all cross-worker collisions with no intra-worker
    change, and any HOME-based R data cache (the sesame ghcr / AnnotationHub
    caches) is still built just once on the worker that owns that file.

    Returns an env dict, or ``None`` when not running under xdist so serial runs
    inherit the ambient environment exactly as before (no behavior change).
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")     # e.g. "gw0"; unset when serial
    if not worker:
        return None
    home = Path(tempfile.gettempdir()) / "xqtl_test_home" / worker
    (home / ".sos").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = str(home)
    return env


def _step_stderr_dump(proc_text, params) -> str:
    """Collect the SoS per-step ``.stderr`` files — both the paths SoS references in
    ``proc_text`` (on failure it prints ``stderr=<path>``) and any found by scanning
    the output directories passed in ``params`` — and return their content, capped."""
    clean = _ANSI.sub("", proc_text or "")
    files: list[str] = []
    seen: set[str] = set()

    def _add(p):
        p = str(p)
        if p not in seen:
            seen.add(p)
            files.append(p)

    for m in re.finditer(r"stderr=(\S+\.stderr)", clean):
        _add(m.group(1))
    for val in (params or {}).values():                 # scan output dirs given as params
        d = Path(str(val))
        try:
            if d.is_dir():
                for f in sorted(d.rglob("*.stderr")):
                    _add(f)
        except OSError:
            continue

    chunks, total = [], 0
    for f in files:
        try:
            content = Path(f).read_text(errors="replace")
        except OSError:
            continue
        if not content.strip():
            continue
        if len(content) > _MAX_PER_FILE:
            content = "...(truncated, tail shown)...\n" + content[-_MAX_PER_FILE:]
        chunk = f"\n===== step stderr: {f} =====\n{content}"
        chunks.append(chunk)
        total += len(chunk)
        if total >= _MAX_TOTAL:
            chunks.append("\n...(further step logs truncated)...")
            break
    return "".join(chunks)


def run_sos(notebook, step, params=None, cwd=None, timeout: int = 900, retries: int = 1):
    """Run one SoS step. `params` is a dict; a value of ``True`` is emitted as a
    bare flag, a list/tuple as space-separated values, anything else stringified.
    Returns the CompletedProcess (assert on ``.returncode`` in the test). The
    per-step ``.stderr`` files are appended to ``.stderr`` so the real error shows;
    a timeout is returned as returncode 124 (not raised) with the partial logs.

    A failed run whose output bears the SoS ZMQ controller-bind flake signature is
    retried up to ``retries`` times (default 1) — that failure is host-network
    infra, not the notebook under test. A timeout is never retried."""
    cmd = [sos_bin(), "run", str(notebook), step]
    for key, val in (params or {}).items():
        cmd.append(f"--{key}")
        if val is True:
            continue                                   # bare flag
        if isinstance(val, (list, tuple)):
            cmd.extend(str(v) for v in val)
        else:
            cmd.append(str(val))
    cmd.append("-j1")

    run_cwd = str(cwd) if cwd else None
    env = _worker_env()                                # per-xdist-worker HOME, or None (serial)

    attempt = 0
    while True:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=run_cwd, env=env)
        except subprocess.TimeoutExpired as e:
            # On timeout, TimeoutExpired.stdout/.stderr come back as BYTES even with
            # text=True (the decode only happens on normal completion) — decode them.
            def _s(x):
                if x is None:
                    return ""
                return x.decode(errors="replace") if isinstance(x, bytes) else x
            out, err = _s(e.stdout), _s(e.stderr)
            dump = _step_stderr_dump(out + err, params)
            err = f"{err}\n[run_sos] TIMEOUT after {timeout}s{dump}"
            return subprocess.CompletedProcess(cmd, returncode=124, stdout=out, stderr=err)

        if (proc.returncode != 0 and attempt < retries
                and _ZMQ_FLAKE_RE.search(_ANSI.sub("", (proc.stdout or "") + (proc.stderr or "")))):
            attempt += 1
            continue                                   # transient controller-bind flake — rerun
        break

    dump = _step_stderr_dump(proc.stdout + proc.stderr, params)
    note = f"\n[run_sos] retried {attempt}x after a ZMQ controller-bind flake" if attempt else ""
    if dump or note:
        proc.stderr = (proc.stderr or "") + note + dump
    return proc
