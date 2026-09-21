"""Every committed notebook must be valid nbformat 4.5 with cell ids.

This is checked rather than repaired: `nbformat.validator.normalize()` assigns
random cell ids, so repairing during the site build would rewrite notebooks on
every run and the build workflow's add-and-commit step would push that churn.

Run `python tests/scripts/test_notebook_format.py --fix` to normalize anything
this catches.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import nbformat
import pytest

REPO = Path(__file__).resolve().parents[2]


def notebooks():
    return [
        path for path in sorted(REPO.rglob("*.ipynb"))
        # pipeline/ holds symlinks the build regenerates; _build is output.
        if not path.is_symlink()
        and not {".git", "_build", ".ipynb_checkpoints"} & set(path.parts)
    ]


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: str(p.relative_to(REPO)))
def test_notebook_is_valid_nbformat(path):
    """Read and validate with warnings promoted to errors.

    A missing cell id is only a warning today but is documented to become a hard
    error in a future nbformat release, so it has to fail here.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
    except Exception as exc:  # noqa: BLE001 - surfacing the reason is the point
        pytest.fail(
            f"{path.relative_to(REPO)} is not valid nbformat 4.5:\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"Fix with: python tests/scripts/test_notebook_format.py --fix"
        )


def _fix():
    fixed = []
    for path in notebooks():
        before = path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            notebook = nbformat.read(path, as_version=4)
        notebook.nbformat_minor = max(notebook.get("nbformat_minor", 0), 5)
        nbformat.validator.normalize(notebook)
        after = nbformat.writes(notebook) + "\n"
        if after != before:
            path.write_text(after, encoding="utf-8")
            fixed.append(path.relative_to(REPO))
    print(f"Normalized {len(fixed)} notebook(s)")
    for path in fixed:
        print(f"  {path}")
    if not fixed:
        print("  (nothing to do)")


if __name__ == "__main__":
    import sys

    if "--fix" not in sys.argv:
        raise SystemExit(__doc__)
    _fix()
