"""Notebook tier: run Workflow Builder commands exactly as the builder shows them.

Each command below is taken from the builder registry (website/builder_registry.py),
which reads it from the module page's tagged example cell, and is run unchanged from a
scratch folder that links pipeline/, code/ and tests/ from the repository. So a pass
here means the command a user copies from the builder runs on the example data.

Only self-contained commands are listed: ones whose inputs are example files, or files
their own setup lines create. Commands that read another step's output/ are covered by
module tests that build those inputs. mixture_prior ud_unconstrained is left out: the udr
TED update stops on the small example data (it needs larger z-scale input).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.sos_runner import _worker_env, sos_bin

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "website"))
import builder_registry as reg  # noqa: E402

STEPS = {
    "mixture_prior.ipynb": ["flash", "flash_nonneg", "ud",
                            "vhat_identity", "vhat_simple", "vhat_mle",
                            "vhat_simple_specific", "vhat_corshrink_xcondition", "plot_U"],
    "mash_posterior.ipynb": ["mash_posterior_contrast", "feature_score_meta",
                             "feature_score_finemap", "feature_score_nsig", "feature_pval_pair"],
}
CASES = [(nb[:-len(".ipynb")], wf) for nb, wfs in STEPS.items() for wf in wfs]

_CMDS = None


def _commands():
    global _CMDS
    if _CMDS is None:
        cmdlib, wio, errors, _ = reg.build()
        assert not errors, errors
        _CMDS = {(m, e["wf"]): (e["cmd"], wio.get(f"{m}|{e['wf']}", {})) for m, l in cmdlib.items() for e in l}
    return _CMDS


@pytest.mark.skipif(shutil.which(sos_bin()) is None, reason="needs sos")
@pytest.mark.parametrize("module,wf", CASES, ids=[f"{m}|{w}" for m, w in CASES])
def test_builder_command_runs(module, wf, tmp_path):
    cmd, io = _commands()[(module, wf)]
    for d in ("pipeline", "code", "tests"):
        (tmp_path / d).symlink_to(REPO / d)
    script = cmd.replace("sos run ", f"{sos_bin()} run ")
    p = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path,
                       capture_output=True, text=True, timeout=1800,
                       env=_worker_env() or dict(os.environ))
    assert p.returncode == 0, f"{script}\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}"
    outs = [tmp_path / o for o in io.get("o", [])]
    assert outs, "no declared output"
    for o in outs:
        made = [f for f in o.rglob("*") if f.is_file() and f.stat().st_size > 0
                and not f.name.endswith((".stderr", ".stdout"))] if o.is_dir() else ([o] if o.exists() else [])
        assert made, f"{o} has no output files"
