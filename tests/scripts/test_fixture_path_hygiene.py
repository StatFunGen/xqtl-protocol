"""Guard: no MACHINE-LOCAL absolute paths may be committed in COMPARISON-TARGET fixtures.

A machine-local path (a user home like ``/Users/<me>`` or ``/home/<user>``, or a per-user
tmp dir like ``/var/folders/...`` / ``/tmp/...``) baked into a generated fixture matches on
the machine that made it but not on another — the failure mode that broke the ctwas
fixtures. This test fails if any such path is found in a comparison target so a new one
can't slip in.

SCOPE — comparison targets only: files under an ``expected/`` dir or named ``expected*``.
Those are generated locally, so a home/tmp path in them is a dynamic per-run path. Raw
INPUT fixtures are out of scope: they may carry immutable upstream provenance (a BAM
``@PG`` command line, a captured ``devtools::load_all`` call, a bcftools ``##command``)
that is identical on every machine and is never value-compared, so it can't break CI.

It scans every format a path can hide in: plain text, gzip'd text, parquet string
columns, and — critically — RDS (gzip-compressed, so a plain grep can't see inside;
they're walked in R via audit_fixture_paths.R, including S4 slots and closure envs).

The pattern is OS-AGNOSTIC — it flags a user-home or per-user tmp dir on either macOS
(``/Users/...``, ``/var/folders/...``) OR Linux (``/home/<user>/...``, ``/tmp/...``), since
fixtures may be generated on either and those absolute paths are regenerated per run (so
they differ across machines). It does NOT match a non-home provenance root baked STATICALLY
into the upstream toy data (e.g. a BAM ``@PG`` line's ``/restricted/projectnb/...``), which
is a fixed cluster path — not a home/tmp shape — and is load-bearing for the exact compare.
"""
from __future__ import annotations

import gzip
import re
import subprocess
from pathlib import Path

from helpers.r_runner import rscript_bin

FIX = "tests/fixtures"
MACHINE_PATH = re.compile(r"/Users/|/home/[A-Za-z][A-Za-z0-9._-]*/|/var/folders/|/private/var/folders/|/tmp/")


def _is_comparison_target(p: Path, root: Path) -> bool:
    """A fixture we value-compare: under an ``expected/`` dir or named ``expected*``."""
    return "/expected/" in str(p.relative_to(root)) or p.name.startswith("expected")


def _scan_text_and_gz(root: Path) -> list[str]:
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or not _is_comparison_target(p, root):
            continue
        try:
            text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
        except (UnicodeDecodeError, OSError, EOFError, gzip.BadGzipFile):
            continue                                    # binary / unreadable -> not a text path carrier
        for i, line in enumerate(text.splitlines(), 1):
            m = MACHINE_PATH.search(line)
            if m:
                hits.append(f"{p.relative_to(root)}:{i}: ...{line[max(0, m.start() - 8):m.start() + 70]}...")
                break
    return hits


def _scan_parquet(root: Path) -> list[str]:
    try:
        import pyarrow as pa                            # noqa: PLC0415
        import pyarrow.parquet as pq                    # noqa: PLC0415
    except ImportError:
        return []
    hits = []
    for p in sorted(root.rglob("*.parquet")):
        if not _is_comparison_target(p, root):
            continue
        try:
            t = pq.read_table(p)
        except Exception:                               # noqa: BLE001
            continue
        for c in t.column_names:
            if not pa.types.is_string(t.schema.field(c).type):
                continue
            for v in t.column(c).to_pylist():
                if v and MACHINE_PATH.search(str(v)):
                    hits.append(f"{p.relative_to(root)} col {c!r}: {str(v)[:90]}")
                    break
    return hits


def _scan_rds(repo_root: Path) -> list[str]:
    r = subprocess.run(
        [rscript_bin(), str(repo_root / "tests/helpers/audit_fixture_paths.R"), str(repo_root / FIX)],
        capture_output=True, text=True, timeout=600)
    # exit 0 = clean; exit 1 = hits printed on stdout; anything else = the walker itself failed
    assert r.returncode in (0, 1), f"audit_fixture_paths.R errored:\n{r.stdout}\n{r.stderr}"
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def test_fixtures_have_no_machine_local_paths(repo_root):
    root = repo_root / FIX
    hits = _scan_text_and_gz(root) + _scan_parquet(root) + _scan_rds(repo_root)
    assert not hits, (
        "machine-local absolute paths found in committed fixtures "
        "(basename them, or compare with normalize_paths and neutralize the fixture):\n  "
        + "\n  ".join(hits))
