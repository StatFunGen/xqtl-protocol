#!/usr/bin/env python3
"""Build-time helpers for the FunGen-xQTL Jupyter Book site.

Jupyter Book 2 (MyST) builds straight from the repository layout: the TOC in
myst.yml points at notebooks where they live, MyST derives a flat page slug from
each basename (code/SoS/reference_data/reference_data.ipynb -> /reference-data/)
and resolves relative links and images itself. Nothing needs staging or
rewriting, so this script covers only the three things MyST does not do:

  --stage-widget  The workflow builder is a self-contained HTML widget. MyST
                  renders a {raw} html block as escaped text and strips <style>
                  and <script> from page content, so the widget cannot live on a
                  book page. This writes a slimmed copy of its notebook (widget
                  replaced by a link) for the TOC to point at.
  --widget        Write the widget itself as a standalone page, extracted from
                  the same notebook so the two cannot drift.
  --redirects     Jupyter Book 1 published /<basename>.html and older builds
                  published /code/.../<name>.html. MyST publishes /<slug>/, so
                  emit redirect stubs for both legacy forms. Slugs are read back
                  from MyST's own build output rather than recomputed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import nbformat
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "code" / "SoS"
CONFIG = REPO_ROOT / "myst.yml"

RAW_HTML_BLOCK_RE = re.compile(r"^(`{3,})\{raw\}[ \t]+html[ \t]*\n(.*?)\n\1[ \t]*$", re.S | re.M)

WIDGET_NOTEBOOK = SOURCE_ROOT / "xqtl_protocol_workflow_builder.ipynb"
WIDGET_PAGE = "xqtl_protocol_workflow_builder.html"
STAGED_WIDGET = REPO_ROOT / "website" / "_generated" / "xqtl_protocol_workflow_builder.ipynb"
WIDGET_PREFIX = (
    '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    "<title>xQTL Analysis Workflow Builder</title><style>"
    "body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,Roboto,sans-serif;"
    "margin:0;padding:24px;max-width:1120px;margin-inline:auto;color:#2c3338}"
    "</style></head><body>\n"
)
WIDGET_SUFFIX = "</body></html>\n"
# Absolute: MyST would otherwise prepend BASE_URL to a root-relative path, or
# content-hash anything it can resolve as a local asset.
WIDGET_URL = f"https://statfungen.github.io/xqtl-protocol/{WIDGET_PAGE}"
WIDGET_LINK = f"**[Open the xQTL Analysis Workflow Builder]({WIDGET_URL})**\n"


def toc_entries():
    """Yield every (file, path) pair in the myst.yml table of contents."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def walk(items):
        for item in items:
            if "file" in item:
                yield item["file"]
            yield from walk(item.get("children", []))

    return list(walk(config["project"]["toc"]))


def toc_notebooks():
    return [f for f in toc_entries() if f.endswith(".ipynb")]


# --------------------------------------------------------------------------- #
# Workflow builder widget
# --------------------------------------------------------------------------- #


def _markdown_cells(notebook):
    return [c for c in notebook.cells if c.cell_type == "markdown"]


def extract_widget_html():
    """Return the standalone builder page, built from its notebook."""
    notebook = nbformat.read(WIDGET_NOTEBOOK, as_version=4)
    blocks = []
    for cell in _markdown_cells(notebook):
        blocks += [m.group(2) for m in RAW_HTML_BLOCK_RE.finditer(cell.source)]
    if len(blocks) != 1:
        raise SystemExit(
            f"Expected exactly one ```{{raw}} html block in {WIDGET_NOTEBOOK.name}, found {len(blocks)}."
        )
    return WIDGET_PREFIX + blocks[0] + "\n" + WIDGET_SUFFIX


def stage_widget_notebook():
    """Write the book-page copy of the builder notebook: widget replaced by a link."""
    notebook = nbformat.read(WIDGET_NOTEBOOK, as_version=4)
    replaced = 0
    for cell in _markdown_cells(notebook):
        cell.source, count = RAW_HTML_BLOCK_RE.subn(lambda _: WIDGET_LINK, cell.source)
        replaced += count
    if replaced != 1:
        raise SystemExit(f"Expected to replace exactly one widget block, replaced {replaced}.")
    STAGED_WIDGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, STAGED_WIDGET)
    print(f"Staged book page {STAGED_WIDGET.relative_to(REPO_ROOT)}")


def write_widget(html_root: Path):
    target = html_root / WIDGET_PAGE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(extract_widget_html(), encoding="utf-8")
    print(f"Wrote standalone {WIDGET_PAGE} ({target.stat().st_size} bytes) under {html_root}")


# --------------------------------------------------------------------------- #
# Legacy URL redirects
# --------------------------------------------------------------------------- #


def published_slugs(html_root: Path):
    """Map source path -> published slug, read from MyST's own build output."""
    slugs = {}
    for page in sorted(html_root.glob("*.json")):
        try:
            data = json.loads(page.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if "slug" in data and "location" in data:
            slugs[data["location"].lstrip("/")] = data["slug"]
    if not slugs:
        raise SystemExit(f"No MyST page manifests found under {html_root}; run the build first.")
    return slugs


def redirect_document(target: str):
    escaped = html.escape(target, quote=True)
    js_target = json.dumps(target)
    return (
        "<!doctype html>\n"
        '<meta charset="utf-8">\n'
        "<title>Redirecting...</title>\n"
        f'<meta http-equiv="refresh" content="0; url={escaped}">\n'
        f'<link rel="canonical" href="{escaped}">\n'
        f"<script>location.replace({js_target});</script>\n"
        f'<p>Redirecting to <a href="{escaped}">{escaped}</a>.</p>\n'
    )


def write_redirect(path: Path, target: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(target, path.parent).replace(os.sep, "/")
    path.write_text(redirect_document(rel_target), encoding="utf-8")


def write_redirects(html_root: Path):
    slugs = published_slugs(html_root)
    missing = [f for f in toc_notebooks() if f not in slugs]
    if missing:
        raise SystemExit(
            "These TOC notebooks were not published (basename collision after "
            "slugification?):\n  " + "\n  ".join(missing)
        )

    count = 0
    for source, slug in sorted(slugs.items()):
        path = Path(source)
        if path.suffix != ".ipynb":
            continue  # README.md / CONTRIBUTORS.md have no legacy URLs
        is_widget = path.name == WIDGET_NOTEBOOK.name
        target = html_root / (WIDGET_PAGE if is_widget else f"{slug}/index.html")

        old_paths = []
        if not is_widget:
            # Jupyter Book 1 published /<basename>.html.
            old_paths.append(html_root / f"{path.stem}.html")
        # The builder page is generated elsewhere; its legacy URLs follow the
        # notebook it is generated from.
        source = WIDGET_NOTEBOOK.relative_to(REPO_ROOT) if is_widget else path
        try:
            rel = source.relative_to("code/SoS").with_suffix(".html")
        except ValueError:
            rel = None  # generated pages have no repository-relative legacy URL
        if rel is not None:
            old_paths += [html_root / prefix / rel for prefix in ("code", "code/SoS")]

        for old in old_paths:
            write_redirect(old, target)
            count += 1
    print(f"Wrote {count} compatibility redirects under {html_root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-widget", action="store_true",
                        help="write the slimmed builder notebook the TOC points at")
    parser.add_argument("--widget", type=Path, metavar="HTML_ROOT",
                        help="write the standalone builder page into a built site")
    parser.add_argument("--widget-file", type=Path,
                        help="write the standalone builder page to this path (refreshes the committed copy)")
    parser.add_argument("--redirects", type=Path, metavar="HTML_ROOT",
                        help="write legacy-URL redirect stubs into a built site")
    args = parser.parse_args(argv)

    if not any([args.stage_widget, args.widget, args.widget_file, args.redirects]):
        parser.error("provide --stage-widget, --widget, --widget-file or --redirects")
    if args.stage_widget:
        stage_widget_notebook()
    if args.widget:
        write_widget(args.widget)
    if args.widget_file:
        args.widget_file.write_text(extract_widget_html(), encoding="utf-8")
        print(f"Wrote {args.widget_file}")
    if args.redirects:
        write_redirects(args.redirects)


if __name__ == "__main__":
    main()
