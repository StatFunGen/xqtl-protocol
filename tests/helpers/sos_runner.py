"""Run SoS notebook steps as subprocesses (the production entrypoint).

A notebook test drives ``sos run <notebook> <step> --params`` exactly as a user
would, so it covers the SoS cell orchestration (manifest/param wiring, path
resolution, step chaining) that the direct script tests can't.
"""
from __future__ import annotations

import os
import subprocess


def sos_bin() -> str:
    return os.environ.get("XQTL_SOS") or "sos"


def run_sos(notebook, step, params=None, cwd=None, timeout: int = 900):
    """Run one SoS step. `params` is a dict; a value of ``True`` is emitted as a
    bare flag, a list/tuple as space-separated values, anything else stringified.
    Returns the CompletedProcess (assert on ``.returncode`` in the test)."""
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
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, cwd=str(cwd) if cwd else None)
