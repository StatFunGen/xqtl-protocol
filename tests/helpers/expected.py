"""Compare a freshly produced pipeline output against its committed expected-output
fixture (``tests/fixtures/<domain>/expected/...``).

Three comparison modes, chosen per output in the domain's ``expected_manifest.tsv``:

- ``exact``       byte-identical logical content. ``.gz`` / ``.bgz`` are decompressed
                  first so gzip mtime / OS bytes don't cause false diffs; ``.rds`` is
                  compared with R ``identical()`` (via ``rds_compare.R``).
- ``tolerant``    numeric fields within ``(rtol, atol)``, everything else exact.
                  Dispatch by logical extension:
                    ``.rds``      -> ``rds_compare.R`` (``all.equal`` tolerance)
                    ``.parquet``  -> pyarrow (optional dependency)
                    tabular text  -> cell-wise numeric compare (stdlib only)
                    other text    -> falls back to exact bytes
- ``projection``  compare a stable, architecture-robust view produced by a caller
                  ``project(path) -> object`` callable (e.g. credible-set membership,
                  a PIP-thresholded variant set, sorted top-k). Use for chaotic
                  iterative outputs (susie / mvsusie / mash) whose raw floats
                  legitimately differ across the x86 / ARM CI runners.

Every mismatch raises ``AssertionError`` with the first differing location, so the
failure is actionable in CI. The helper is intentionally dependency-light: tabular
text and gzip are handled with the stdlib, RDS is delegated to R (which already has
the pecotmr / jsonlite stack), and parquet degrades to a clear error if pyarrow is
absent.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import subprocess
from pathlib import Path

HELPERS_DIR = Path(__file__).resolve().parent
RDS_COMPARE = HELPERS_DIR / "rds_compare.R"

_GZ_SUFFIXES = (".gz", ".bgz")
_TABULAR_EXTS = {".tsv", ".csv", ".txt", ".bed", ".tab", ".gct", ".map",
                 ".afreq", ".fam", ".bim", ".eigenvec", ".region_list", ".gtf", ".vcf"}

try:                                                     # reuse the test suite's Rscript resolver
    from helpers.r_runner import rscript_bin
except Exception:                                        # pragma: no cover - import fallback
    def rscript_bin() -> str:
        return os.environ.get("XQTL_RSCRIPT") or "Rscript"


# --------------------------------------------------------------------------- I/O
def _read_bytes(path) -> bytes:
    """Raw bytes, transparently decompressing ``.gz`` / ``.bgz`` so we compare the
    logical content (gzip carries an mtime + OS byte that vary run-to-run)."""
    path = Path(path)
    if path.suffix.lower() in _GZ_SUFFIXES:
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return path.read_bytes()


def _read_text(path) -> str:
    return _read_bytes(path).decode("utf-8")


def _logical_ext(path) -> str:
    """Extension with a trailing ``.gz`` / ``.bgz`` stripped (``x.tsv.gz`` -> ``.tsv``)."""
    p = Path(path)
    if p.suffix.lower() in _GZ_SUFFIXES:
        p = p.with_suffix("")
    return p.suffix.lower()


def _looks_text(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return b"\x00" not in raw
    except UnicodeDecodeError:
        return False


# ------------------------------------------------------------------- numeric compare
def _as_float(cell: str):
    """Return the float value of a cell, or ``None`` if it is not numeric."""
    try:
        return float(cell)
    except (ValueError, TypeError):
        return None


def _is_abs_path(cell: str) -> bool:
    """A machine-specific absolute file path (basename-compared under normalize_paths)."""
    return cell.startswith("/") and " " not in cell and cell.count("/") >= 2


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * abs(b)


# ------------------------------------------------------------------- table compare
def _auto_delimiter(text: str, ext: str) -> str | None:
    if "\t" in text:
        return "\t"
    if ext == ".csv":
        return ","
    return None                                          # None -> split on any whitespace


def _rows(text: str, delimiter: str | None) -> list[list[str]]:
    lines = text.split("\n")
    if lines and lines[-1] == "":                        # drop the single trailing newline
        lines.pop()
    out = []
    for line in lines:
        line = line.rstrip("\r")
        cells = line.split(delimiter) if delimiter is not None else line.split()
        out.append([c.strip() for c in cells])
    return out


def _drop_lines(text: str, patterns) -> str:
    """Remove whole lines matching any regex in ``patterns`` (used to mask volatile
    header lines — e.g. a GTF's ``##date`` / ``##source-version`` or a VCF's
    ``##fileDate`` — that carry no data but change run-to-run)."""
    compiled = [re.compile(p) for p in patterns]
    return "\n".join(ln for ln in text.split("\n")
                     if not any(rx.search(ln) for rx in compiled))


def _compare_table(produced, expected, rtol, atol, delimiter, normalize_paths=False,
                   ignore_lines=None) -> str | None:
    """Cell-wise compare: numeric cells within tolerance, everything else exact.
    With ``normalize_paths``, absolute-path cells compare by basename. With
    ``ignore_lines`` (a list of regexes), whole lines matching any pattern are
    dropped from BOTH sides first (mask volatile header lines). Returns ``None``
    on match, else a human-readable description of the first diff."""
    ptext, etext = _read_text(produced), _read_text(expected)
    if ignore_lines:
        ptext, etext = _drop_lines(ptext, ignore_lines), _drop_lines(etext, ignore_lines)
    delim = delimiter if delimiter is not None else _auto_delimiter(etext, _logical_ext(expected))
    prows, erows = _rows(ptext, delim), _rows(etext, delim)

    if len(prows) != len(erows):
        return f"row count differs: produced {len(prows)} vs expected {len(erows)}"
    for i, (pr, er) in enumerate(zip(prows, erows), start=1):
        if len(pr) != len(er):
            return f"row {i}: cell count differs ({len(pr)} vs {len(er)})\n  produced: {pr}\n  expected: {er}"
        for j, (pc, ec) in enumerate(zip(pr, er), start=1):
            if pc == ec:
                continue
            if normalize_paths and _is_abs_path(pc) and _is_abs_path(ec):
                if os.path.basename(pc) == os.path.basename(ec):
                    continue
            pf, ef = _as_float(pc), _as_float(ec)
            if pf is not None and ef is not None:
                if _close(pf, ef, rtol, atol):
                    continue
                return (f"row {i} col {j}: numeric mismatch beyond tol "
                        f"(rtol={rtol}, atol={atol}): {pc} vs {ec}")
            return f"row {i} col {j}: value differs: {pc!r} vs {ec!r}"
    return None


# --------------------------------------------------------------------- rds compare
def _compare_rds(produced, expected, tolerance: float, normalize_paths: bool = False,
                 sort_rows: bool = False) -> str | None:
    """Delegate to rds_compare.R. ``tolerance < 0`` means identical() (exact);
    ``normalize_paths`` basename-normalizes absolute-path strings first; ``sort_rows``
    canonicalizes each data.frame's row order (for unordered-set outputs). The two flags
    are passed positionally ("0"/"1") so the arg contract is stable."""
    cmd = [rscript_bin(), str(RDS_COMPARE), str(produced), str(expected), str(tolerance),
           "1" if normalize_paths else "0", "1" if sort_rows else "0"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise AssertionError(f"rds_compare.R failed:\n{p.stdout}\n{p.stderr}")
    verdict = json.loads(p.stdout)
    if verdict.get("equal"):
        return None
    return "RDS differs: " + "; ".join(verdict.get("report", ["(no detail)"]))


# ------------------------------------------------------------------ parquet compare
def _compare_parquet(produced, expected, rtol, atol) -> str | None:
    try:
        import pyarrow.parquet as pq                     # noqa: PLC0415
    except ImportError as e:                             # pragma: no cover
        raise AssertionError(
            "parquet comparison needs pyarrow; add it to the env or compare a "
            "text projection of this output instead") from e
    pt = pq.read_table(str(produced))
    et = pq.read_table(str(expected))
    if pt.column_names != et.column_names:
        return f"parquet columns differ: {pt.column_names} vs {et.column_names}"
    if pt.num_rows != et.num_rows:
        return f"parquet row count differs: {pt.num_rows} vs {et.num_rows}"
    for name in et.column_names:
        pcol, ecol = pt.column(name).to_pylist(), et.column(name).to_pylist()
        for i, (pv, ev) in enumerate(zip(pcol, ecol)):
            if pv == ev:
                continue
            if isinstance(pv, (int, float)) and isinstance(ev, (int, float)):
                if _close(float(pv), float(ev), rtol, atol):
                    continue
            return f"parquet column {name!r} row {i}: {pv} vs {ev}"
    return None


# ------------------------------------------------------------------------- public
def assert_matches_expected(produced, expected, *, mode: str = "exact",
                            rtol: float = 1e-6, atol: float = 1e-8,
                            project=None, delimiter: str | None = None,
                            normalize_paths: bool = False,
                            ignore_lines: list[str] | None = None,
                            sort_rows: bool = False) -> None:
    """Assert ``produced`` matches the committed ``expected`` fixture.

    mode:
      ``exact``       decompressed-byte identity (``.rds`` via identical()).
      ``tolerant``    numeric fields within (rtol, atol); dispatch by file type.
      ``projection``  ``project(produced) == project(expected)`` (supply ``project``).

    ``delimiter`` overrides table delimiter auto-detection for the tolerant path.
    ``ignore_lines`` (tolerant text path only) is a list of regexes; whole lines
    matching any pattern are dropped from both sides before comparing — use it to
    mask volatile header lines (GTF ``##date`` / ``##source-version``, VCF
    ``##fileDate``) that carry no data. Raises ``AssertionError`` locating the
    first mismatch.
    """
    produced, expected = Path(produced), Path(expected)
    if not expected.exists():
        raise AssertionError(f"expected fixture missing: {expected}")
    if not produced.exists():
        raise AssertionError(f"produced output missing: {produced}")

    ext = _logical_ext(expected)

    if mode == "projection":
        if project is None:
            raise ValueError("mode='projection' requires a project(path) callable")
        pv, ev = project(produced), project(expected)
        if pv != ev:
            raise AssertionError(
                f"projection mismatch for {produced.name}:\n  produced: {pv}\n  expected: {ev}")
        return

    if mode == "exact":
        if ext == ".rds":
            diff = _compare_rds(produced, expected, tolerance=-1.0,
                                normalize_paths=normalize_paths, sort_rows=sort_rows)
        else:
            diff = None if _read_bytes(produced) == _read_bytes(expected) else \
                "decompressed bytes differ"
        if diff:
            raise AssertionError(f"{produced.name}: {diff}")
        return

    if mode == "tolerant":
        if ext == ".rds":
            diff = _compare_rds(produced, expected, tolerance=rtol,
                                normalize_paths=normalize_paths, sort_rows=sort_rows)
        elif ext == ".parquet":
            diff = _compare_parquet(produced, expected, rtol, atol)
        elif ext in _TABULAR_EXTS or _looks_text(_read_bytes(expected)):
            diff = _compare_table(produced, expected, rtol, atol, delimiter,
                                  normalize_paths=normalize_paths, ignore_lines=ignore_lines)
        else:                                            # opaque binary -> exact bytes
            diff = None if _read_bytes(produced) == _read_bytes(expected) else \
                "binary bytes differ (no tolerant comparator for this type)"
        if diff:
            raise AssertionError(f"{produced.name}: {diff}")
        return

    raise ValueError(f"unknown mode: {mode!r} (expected exact|tolerant|projection)")


def load_manifest(path) -> list[dict]:
    """Read an ``expected_manifest.tsv`` (columns: output, mode, rtol, atol, notes).

    Blank lines and ``#`` comments are skipped. ``rtol`` / ``atol`` are parsed to
    float when present. The projection callable (for ``mode=projection`` rows) is
    supplied by the test, keyed on ``output`` — it can't live in a TSV.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        header = ["output", "mode", "rtol", "atol", "notes"]
        rec = dict(zip(header, parts + [""] * (len(header) - len(parts))))
        for k in ("rtol", "atol"):
            rec[k] = float(rec[k]) if rec[k] not in ("", "-") else None
        rows.append(rec)
    return rows
