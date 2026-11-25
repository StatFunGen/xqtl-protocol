"""Tier B: twas_gwas_window.R -> the window a [twas] job builds its GwasSumStats
over: its analysis unit, widened to every variant of its genes' TWAS weights and
fine-mapping results."""
from __future__ import annotations

import subprocess

from helpers.r_runner import rscript_bin

SCRIPT = "code/script/pecotmr_integration/twas_gwas_window.R"
# The toy gene's weights span chr22:10685239-18850784.
TW = "tests/fixtures/twas/protocol_example.twas.reshaped_toy.chr22_ENSG00000130538.twas_weights.s4.rds"


def _gene_table(path, rows):
    path.write_text("twas_weights\tfine_mapping_result\toutput\n"
                    + "".join(f"{w}\t{f}\t{o}\n" for w, f, o in rows))
    return path


def _window(run_r, repo_root, table, unit):
    p = run_r(repo_root / SCRIPT, ["--gene-table", table, "--unit", unit], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout.strip()


def test_window_covers_the_weights(run_r, repo_root, tmp_path):
    """A region holds genes by TSS, so the toy gene's weights reach well past a
    1 Mb region around it; a fine-mapping result that does not exist is none, as
    in twas.R. The window never shrinks below the unit."""
    table = _gene_table(tmp_path / "genes.tsv",
                        [(repo_root / TW, tmp_path / "none.rds", tmp_path / "out.rds")])
    assert _window(run_r, repo_root, table, "chr22:15000000-16000000") == "chr22:10685239-18850784"
    assert _window(run_r, repo_root, table, "chr22:5000000-25000000") == "chr22:5000000-25000000"


def test_window_covers_the_fine_mapping_result(run_r, repo_root, tmp_path):
    """MR scores the fine-mapping result against the same GWAS, so a result that
    reaches past the weights widens the window to it."""
    fmr = tmp_path / "fine_mapping.rds"
    p = subprocess.run([rscript_bin(), "-e", f"""
        suppressPackageStartupMessages(library(pecotmr))
        ids <- c("chr22:9000000:A:G", "chr22:20000000:G:A")
        saveRDS(QtlFineMappingResult(
            studyName = "protocol_example", context = "bulk_rnaseq",
            trait = "ENSG00000130538", method = "susie",
            entry = list(FineMappingRow(variantIds = ids, susieFit = list(),
                topLoci = data.frame(variant_id = ids, pip = c(0.5, 0.2),
                    posterior_mean = c(0.1, 0.2), posterior_sd = c(0.05, 0.05))))),
            "{fmr}")
    """], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    table = _gene_table(tmp_path / "genes.tsv", [(repo_root / TW, fmr, tmp_path / "out.rds")])
    assert _window(run_r, repo_root, table, "chr22:15000000-16000000") == "chr22:9000000-20000000"
