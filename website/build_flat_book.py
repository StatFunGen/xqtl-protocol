#!/usr/bin/env python3
"""Stage xQTL protocol notebooks as a flat Jupyter Book source tree.

The repository keeps notebooks under code/SoS/<topic>/, but the public site
uses flat page names such as reference_data.html. This script creates a
temporary build source with notebook basenames at the root, rewrites temporary
page-to-page links to those flat names, and can emit compatibility redirects for
older code/... URLs after the HTML build.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "code" / "SoS"
IMAGE_ROOT = REPO_ROOT / "code" / "images"
WEBSITE_ROOT = REPO_ROOT / "website"
SITE_NETLOC = "statfungen.github.io"
SITE_PREFIX = "/xqtl-protocol/"
MARKDOWN_LINK_RE = re.compile(r"(!?\[[^\]]*\]\()([^)]*)(\))")
HTML_SRC_RE = re.compile(r"(\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)
TOC_FILE_RE = re.compile(r"\bfile:\s*([^\s#]+)")


def iter_files(root: Path, pattern: str):
    for path in sorted(root.rglob(pattern)):
        if ".ipynb_checkpoints" not in path.parts and path.is_file():
            yield path


def unique_by_basename(paths, label: str):
    seen = {}
    duplicates = {}
    for path in paths:
        name = path.name
        if name in seen:
            duplicates.setdefault(name, [seen[name]]).append(path)
        else:
            seen[name] = path
    if duplicates:
        lines = [f"Duplicate {label} basenames are not compatible with flat URLs:"]
        for name, dupes in sorted(duplicates.items()):
            rels = ", ".join(str(p.relative_to(SOURCE_ROOT)) for p in dupes)
            lines.append(f"  {name}: {rels}")
        raise SystemExit("\n".join(lines))
    return seen


def notebook_index():
    notebooks = list(iter_files(SOURCE_ROOT, "*.ipynb"))
    return unique_by_basename(notebooks, "notebook")


def asset_index():
    assets = [
        (path, path.relative_to(SOURCE_ROOT))
        for path in iter_files(SOURCE_ROOT, "*")
        if path.suffix != ".ipynb" and path.name != "README.md"
    ]
    if IMAGE_ROOT.exists():
        assets.extend((path, Path("code/images") / path.name) for path in iter_files(IMAGE_ROOT, "*"))

    by_name = {}
    for path, _ in assets:
        by_name.setdefault(path.name, path)
    return assets, by_name


def parse_toc_files():
    toc = WEBSITE_ROOT / "_toc.yml"
    files = []
    for match in TOC_FILE_RE.finditer(toc.read_text(encoding="utf-8")):
        value = match.group(1).strip('"\'')
        if value.endswith(".ipynb"):
            files.append(value)
    return files


def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def flat_page_for_path(path: str, notebooks: dict[str, Path], assets: dict[str, Path]):
    if not path or path.startswith("#"):
        return None
    name = PurePosixPath(path).name
    if not name:
        return None

    suffix = PurePosixPath(name).suffix
    stem = PurePosixPath(name).stem
    if suffix == ".ipynb" and name in notebooks:
        return name
    if suffix == ".html" and f"{stem}.ipynb" in notebooks:
        return f"{stem}.html"
    if not suffix and f"{name}.ipynb" in notebooks:
        return f"{name}.html"
    if suffix and name in assets:
        return name
    return None


def rewrite_url(url: str, notebooks: dict[str, Path], assets: dict[str, Path]):
    if not url or url.startswith("#") or url.startswith("mailto:"):
        return url
    parsed = urlsplit(url)

    if parsed.scheme and not (
        parsed.scheme in {"http", "https"}
        and parsed.netloc == SITE_NETLOC
        and parsed.path.startswith(SITE_PREFIX)
    ):
        return url

    path = parsed.path
    if parsed.netloc == SITE_NETLOC and path.startswith(SITE_PREFIX):
        path = path.removeprefix(SITE_PREFIX)

    flat = flat_page_for_path(path, notebooks, assets)
    if not flat:
        return url

    if parsed.netloc == SITE_NETLOC:
        return urlunsplit((parsed.scheme, parsed.netloc, SITE_PREFIX + flat, parsed.query, parsed.fragment))
    return urlunsplit(("", "", flat, parsed.query, parsed.fragment))


def rewrite_markdown(text: str, notebooks: dict[str, Path], assets: dict[str, Path]):
    def replace_markdown(match):
        return f"{match.group(1)}{rewrite_url(match.group(2), notebooks, assets)}{match.group(3)}"

    def replace_src(match):
        return f"{match.group(1)}{rewrite_url(match.group(2), notebooks, assets)}{match.group(3)}"

    text = MARKDOWN_LINK_RE.sub(replace_markdown, text)
    text = HTML_SRC_RE.sub(replace_src, text)
    return text


def write_flat_markdown(src: Path, dst: Path, notebooks: dict[str, Path], assets: dict[str, Path]):
    text = src.read_text(encoding="utf-8")
    dst.write_text(rewrite_markdown(text, notebooks, assets), encoding="utf-8")


def write_flat_notebook(src: Path, dst: Path, notebooks: dict[str, Path], assets: dict[str, Path]):
    nb = json.loads(src.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            text = "".join(source)
            cell["source"] = rewrite_markdown(text, notebooks, assets).splitlines(keepends=True)
        elif isinstance(source, str):
            cell["source"] = rewrite_markdown(source, notebooks, assets)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def stage_book(output: Path):
    notebooks = notebook_index()
    assets, assets_by_name = asset_index()
    toc_files = parse_toc_files()

    missing = [name for name in toc_files if name not in notebooks]
    nested = [name for name in toc_files if "/" in name or "\\" in name]
    if nested:
        raise SystemExit("TOC notebook entries should be flat basenames only:\n  " + "\n  ".join(nested))
    if missing:
        raise SystemExit("TOC notebooks were not found under code/SoS:\n  " + "\n  ".join(missing))

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    write_flat_markdown(REPO_ROOT / "README.md", output / "README.md", notebooks, assets_by_name)
    website_out = output / "website"
    website_out.mkdir()
    for name in ["_config.yml", "_toc.yml", "references.bib", "xqtl_wf.png", "xqtl_wf.svg"]:
        src = WEBSITE_ROOT / name
        if src.exists():
            shutil.copy2(src, website_out / name)

    for asset, rel in assets:
        copy_or_link(asset, output / rel)
        copy_or_link(asset, output / asset.name)

    for name, src in notebooks.items():
        write_flat_notebook(src, output / name, notebooks, assets_by_name)

    print(f"Staged {len(notebooks)} notebooks as flat book sources in {output}")


def redirect_document(target: str):
    escaped = html.escape(target, quote=True)
    js_target = json.dumps(target)
    return (
        "<!doctype html>\n"
        "<meta charset=\"utf-8\">\n"
        "<title>Redirecting...</title>\n"
        f"<meta http-equiv=\"refresh\" content=\"0; url={escaped}\">\n"
        f"<link rel=\"canonical\" href=\"{escaped}\">\n"
        f"<script>location.replace({js_target});</script>\n"
        f"<p>Redirecting to <a href=\"{escaped}\">{escaped}</a>.</p>\n"
    )


def write_redirect(path: Path, target: Path, html_root: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(target, path.parent).replace(os.sep, "/")
    path.write_text(redirect_document(rel_target), encoding="utf-8")


def write_redirects(html_root: Path):
    notebooks = notebook_index()
    toc_files = parse_toc_files()
    count = 0
    for name in toc_files:
        src = notebooks[name]
        rel_html = src.relative_to(SOURCE_ROOT).with_suffix(".html")
        target = html_root / f"{Path(name).stem}.html"
        for old_prefix in ("code", "code/SoS"):
            redirect_path = html_root / old_prefix / rel_html
            write_redirect(redirect_path, target, html_root)
            count += 1
    print(f"Wrote {count} compatibility redirects under {html_root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="temporary flat Jupyter Book source directory to create")
    parser.add_argument("--redirects", type=Path, help="built HTML directory where old code/... redirects should be written")
    args = parser.parse_args(argv)

    if not args.output and not args.redirects:
        parser.error("provide --output, --redirects, or both")
    if args.output:
        stage_book(args.output)
    if args.redirects:
        write_redirects(args.redirects)


if __name__ == "__main__":
    main()
