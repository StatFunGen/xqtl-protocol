"""Guard: every pecotmr call in a wrapper must still resolve against the INSTALLED pecotmr.

A runtime test only covers a wrapper some test actually runs, and most of the
``pecotmr_integration`` wrappers have none (the suite prints the untested list after
every run). Worse, a wrapper that wraps its accessor call in ``tryCatch`` can swallow an
upstream rename and still exit 0 — pecotmr 0.9.0 dropping the ``get`` prefix from every
accessor turned ``fine_mapping_pip_plot.R`` into a blank "No usable PIP vectors" stub
while its test stayed green.

This check parses every wrapper WITHOUT running it, so it sees all of them, and fails on:

* ``UNRESOLVED`` — a function called by a name nothing provides: not defined in the file,
  not in a file it ``source()``s, not exported by a package it attaches (following the
  ``Depends`` closure that ``library()`` attaches too), not in base R. An upstream rename
  looks exactly like this.
* ``BAD-ARG`` — a named argument that is not in the callee's formals, for a pecotmr
  function whose signature has no ``...`` to absorb it.

SCOPE is ``code/script/pecotmr_integration`` — the wrappers that hold the pecotmr API
contract. The checker itself takes any root, so ``Rscript tests/helpers/wrapper_api_check.R
code/script`` sweeps the whole tree on demand (it reports real findings in the vendored
h3k9ac scripts, which nothing runs and which are out of scope here).

Blind spots, documented rather than silently missed: ``do.call(fn, args)`` argument names
are a runtime value, not syntax; and the four ``...``-dispatching S4 generics
(``fineMappingPipeline``, ``twasWeightsPipeline``, ``colocboostPipeline``,
``qtlAssociationPostprocess``) have their methods' formals rewritten to the generic's
``(data, ...)``, so only their runtime tests cover their argument lists.
"""
from __future__ import annotations

import subprocess

from helpers.r_runner import rscript_bin

SCOPE = "code/script/pecotmr_integration"


def test_wrappers_match_installed_pecotmr_api(repo_root):
    r = subprocess.run(
        [rscript_bin(), str(repo_root / "tests/helpers/wrapper_api_check.R"), SCOPE],
        capture_output=True, text=True, cwd=repo_root, timeout=600)
    # 0 = clean, 1 = findings printed on stdout; anything else = the checker broke
    assert r.returncode in (0, 1), (
        f"wrapper_api_check.R errored:\n{r.stdout}\n{r.stderr}")
    assert r.returncode == 0, (
        "wrapper(s) call a pecotmr API that the installed pecotmr does not provide:\n"
        + r.stdout)
