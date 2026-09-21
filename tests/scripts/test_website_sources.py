"""Guard the website sources that the Jupyter Book build derives its pages from."""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "website"))
import build_support  # noqa: E402

COMMITTED_WIDGET = REPO / "code" / "SoS" / "xqtl_protocol_workflow_builder.html"


def test_committed_widget_matches_notebook():
    """The standalone builder page is generated from its notebook, never edited.

    MyST cannot render the widget on a book page, so the build extracts it into a
    standalone page. The committed copy exists because the test suite reads it;
    this keeps the two from drifting apart.
    """
    assert build_support.extract_widget_html() == COMMITTED_WIDGET.read_text(encoding="utf-8"), (
        f"{COMMITTED_WIDGET.relative_to(REPO)} is out of sync with its notebook. "
        f"Regenerate it with:\n"
        f"  python website/build_support.py --widget-file {COMMITTED_WIDGET.relative_to(REPO)}"
    )


def test_toc_entries_exist():
    """Every myst.yml entry points at a real file, except the generated builder page."""
    missing = [
        entry for entry in build_support.toc_entries()
        if not (REPO / entry).exists() and (REPO / entry) != build_support.STAGED_WIDGET
    ]
    assert not missing, "myst.yml lists files that do not exist: " + ", ".join(missing)


def test_toc_page_slugs_are_unique():
    """MyST derives page URLs from basenames, so two entries must not share one.

    A collision would silently publish one page and drop the other.
    """
    by_slug: dict[str, list[str]] = {}
    for entry in build_support.toc_entries():
        slug = re.sub(r"[^a-z0-9]+", "-", Path(entry).stem.lower()).strip("-")
        by_slug.setdefault(slug, []).append(entry)
    collisions = {slug: names for slug, names in by_slug.items() if len(names) > 1}
    assert not collisions, f"TOC entries share a published URL: {collisions}"
