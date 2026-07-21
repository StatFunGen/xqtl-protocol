"""Run R wrapper scripts as subprocesses and read their RDS outputs.

The wrappers are thin CLI scripts, so the tests exercise them exactly as the
pipeline does: ``Rscript <script> --args`` -> output file. `run_r` returns the
completed process (rc + captured stdout/stderr); `read_rds` shells out to
``rds_probe.R`` to get a JSON structural summary of an RDS the R side wrote.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

HELPERS_DIR = Path(__file__).resolve().parent
RDS_PROBE = HELPERS_DIR / "rds_probe.R"


def rscript_bin() -> str:
    """Rscript to use: ``XQTL_RSCRIPT`` override, else the one on PATH (pixi)."""
    return os.environ.get("XQTL_RSCRIPT") or "Rscript"


def run_r(script, args=(), timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an R wrapper. Returns the CompletedProcess (never raises on rc!=0;
    the test asserts on ``.returncode`` so failures show the captured output)."""
    cmd = [rscript_bin(), str(script), *(str(a) for a in args)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def read_rds(path, timeout: int = 120) -> dict:
    """Structural summary (class / nrow / accessors) of an RDS, via rds_probe.R."""
    p = subprocess.run(
        [rscript_bin(), str(RDS_PROBE), str(path)],
        capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise AssertionError(f"rds_probe.R failed on {path}:\n{p.stdout}\n{p.stderr}")
    return json.loads(p.stdout)
