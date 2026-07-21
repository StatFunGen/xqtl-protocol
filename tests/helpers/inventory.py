"""Session-end inventory: which wrapper scripts got exercised vs. not.

A light coverage signal at the *script* granularity (distinct from line
coverage): after a run it prints how many of the pecotmr_integration wrappers
some test actually invoked, and names the ones still untested — so gaps in the
suite are visible without hunting.
"""
from __future__ import annotations

from pathlib import Path


def wrapper_scripts(repo_root: Path) -> list[str]:
    """Non-legacy pecotmr_integration CLI wrapper basenames (Rscript shebang);
    sourced helper libraries like manifest_common.R are excluded."""
    d = repo_root / "code" / "script" / "pecotmr_integration"
    out = []
    for p in d.glob("*.R"):
        if p.name.startswith("legacy_"):
            continue
        first = p.read_text().splitlines()[0] if p.stat().st_size else ""
        if first.startswith("#!"):
            out.append(p.name)
    return sorted(out)


def report_untested(repo_root: Path, touched: set[str], terminalreporter=None) -> None:
    scripts = wrapper_scripts(repo_root)
    untested = [s for s in scripts if s not in touched]
    n = len(scripts) - len(untested)
    lines = [
        "",
        f"wrapper inventory: {n}/{len(scripts)} pecotmr_integration script(s) exercised",
    ]
    if untested:
        lines.append("  untested: " + ", ".join(untested))
    msg = "\n".join(lines)
    if terminalreporter is not None:
        terminalreporter.write_line(msg)
    else:
        print(msg)
