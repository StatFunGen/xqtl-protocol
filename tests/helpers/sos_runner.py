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
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MAX_PER_FILE = 8000            # cap each .stderr file in the appended dump
_MAX_TOTAL = 40000             # overall cap so a runaway log can't bloat the message


def sos_bin() -> str:
    return os.environ.get("XQTL_SOS") or "sos"


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


def run_sos(notebook, step, params=None, cwd=None, timeout: int = 900):
    """Run one SoS step. `params` is a dict; a value of ``True`` is emitted as a
    bare flag, a list/tuple as space-separated values, anything else stringified.
    Returns the CompletedProcess (assert on ``.returncode`` in the test). The
    per-step ``.stderr`` files are appended to ``.stderr`` so the real error shows;
    a timeout is returned as returncode 124 (not raised) with the partial logs."""
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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=run_cwd)
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

    dump = _step_stderr_dump(proc.stdout + proc.stderr, params)
    if dump:
        proc.stderr = (proc.stderr or "") + dump
    return proc
