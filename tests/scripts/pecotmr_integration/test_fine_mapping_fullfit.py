"""Tier B: fullFit per-CS variant-level columns populate on a REAL credible set.

The protocol_example toy inputs produce no credible sets when re-fit (diffuse
signal over LD), so the wide fullFit columns can only be demonstrated on a
genuine fit. The committed enloc MiGA_eQTL fit is a real chr22 SuSiE fit with a
credible set AND the full per-effect posterior matrices (alpha / mu / mu2 /
lbf_variable / X_column_scale_factors). We reprocess it through pecotmr's
buildTopLoci(fullFit=TRUE, fullFitAlphaOnly=FALSE) and assert every wide per-CS
column lands with real (unscaled) values — an integration guard that the
installed pecotmr the wrappers run against actually carries the feature.

The reprocessing R is written to tmp at test time (not a committed wrapper): it
packages the fit's OWN stored credible sets into the coverage-indexed csTables
shape computeCsTables() returns, then calls the real buildTopLoci.
"""
from __future__ import annotations

import subprocess

import pytest

from helpers import r_runner

_REPROCESS_R = r"""
suppressPackageStartupMessages(library(pecotmr))
a   <- commandArgs(TRUE)
pv  <- readRDS(a[1])$ENSG00000142798$MiGA_THA_eQTL$preset_variants_result
fit <- pv$susie_result_trimmed
vn  <- pv$variant_names
# The fit's own stored credible sets, repackaged as computeCsTables() would.
csTables <- list(
  list(sets = fit$sets, cs_corr = fit$cs_corr, pip = fit$pip),   # 0.95 primary
  fit$sets_secondary$coverage_0.7,
  fit$sets_secondary$coverage_0.5)
names(csTables) <- vapply(c(0.95, 0.7, 0.5), pecotmr:::formatCsColumn,
                          character(1), method = "susie")
attr(csTables, "coverage") <- c(0.95, 0.7, 0.5)
tl <- as.data.frame(pecotmr:::buildTopLoci(fit, csTables, variantNames = vn,
        method = "susie", signalCutoff = 0,
        fullFit = TRUE, fullFitAlphaOnly = FALSE))
write.table(tl, a[2], sep = "\t", quote = FALSE, row.names = FALSE, na = "")
"""

_FIXTURE = ("tests/fixtures/susie_enloc/protocol_example.enloc.MiGA_eQTL."
            "ENSG00000142798.univariate_susie_twas_weights.rds")


def test_fullfit_wide_columns_on_real_credible_set(repo_root, tmp_path):
    fit = repo_root / _FIXTURE
    assert fit.exists(), f"missing committed CS-bearing fixture: {fit}"

    script = tmp_path / "reprocess_fullfit.R"
    script.write_text(_REPROCESS_R)
    out = tmp_path / "toploci_fullfit.tsv"
    p = subprocess.run([r_runner.rscript_bin(), str(script), str(fit), str(out)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    rows = [ln.split("\t") for ln in out.read_text().splitlines()]
    header = rows[0]
    # The always-on scalar plus all four widened columns for the single CS (cs1).
    for col in ("within_cs_pip", "within_cs_pip_cs1", "cs_logbf_cs1",
                "cs_effect_cs1", "cs_effect_var_cs1"):
        assert col in header, f"missing fullFit column {col!r} in {header}"

    idx = {c: header.index(c) for c in header}
    # within_cs_pip is populated for exactly the credible-set member(s); every
    # widened column must carry a real value on that row (proves the full-matrix
    # path ran, not just alpha).
    members = [r for r in rows[1:] if r[idx["within_cs_pip"]] not in ("", "NA")]
    assert members, "within_cs_pip never populated (no CS member row)"
    for r in members:
        for col in ("within_cs_pip_cs1", "cs_logbf_cs1", "cs_effect_cs1",
                    "cs_effect_var_cs1"):
            assert r[idx[col]] not in ("", "NA"), f"{col} empty on a CS member"
        # within_cs_pip is that variant's alpha in its assigned effect: the
        # scalar and the widened cs1 alpha column must agree on the member row.
        assert r[idx["within_cs_pip"]] == r[idx["within_cs_pip_cs1"]]
