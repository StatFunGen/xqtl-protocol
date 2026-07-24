"""Notebook tier: GRM.ipynb — the R reformatting step extracted to GRM.R.

grm_1 (LOCO list) is pure SoS; grm_2 runs `gcta --mbfile --make-grm-gz` (thin inline
call, renamed from the retired `gcta64`); grm_3/grm_4 now call GRM.R (format_apex /
apex_list, vroom/dplyr).

Fixtures (tests/fixtures/grm), from the MWE:
  protocol_example.genotype.chr22.grm.gz / .grm.id   a GCTA GRM + its sample IDs
  expected/protocol_example.genotype.chr22.apex.grm  the readr-produced APEX reference

format_apex maps the integer GRM indices back to sample IDs -> byte-identical to the reference.
"""
from __future__ import annotations

import gzip

WORKER = "code/script/data_preprocessing/genotype/GRM.R"
NB = "pipeline/GRM.ipynb"
FIX = "tests/fixtures/grm"


def test_grm_format_apex(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    out = tmp_path / "apex.grm"
    p = run_r(repo_root / WORKER, [
        "--step", "format_apex",
        "--grm", fix / "protocol_example.genotype.chr22.grm.gz",
        "--id", fix / "protocol_example.genotype.chr22.grm.id",
        "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_text() == (fix / "expected" / "protocol_example.genotype.chr22.apex.grm").read_text()


def test_grm_loco_workflow(run_sos, repo_root, tmp_path):
    # Full LOCO GRM: grm_1 (list) -> grm_2 (gcta) -> grm_3/grm_4 (GRM.R). With two
    # chromosomes, the chr21 GRM is built leave-one-out from chr22 and vice versa.
    fix = repo_root / FIX
    geno_list = tmp_path / "geno.list"
    geno_list.write_text(f"21\t{fix}/geno/chr21.bed\n22\t{fix}/geno/chr22.bed\n")
    p = run_sos(repo_root / NB, "grm", {
        "genoFile": geno_list, "cwd": tmp_path, "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    for chrom in ("chr21", "chr22"):
        grm = tmp_path / f"{chrom}.grm.gz"
        apex = tmp_path / f"{chrom}.apex.grm"
        assert grm.exists() and apex.exists()
        # gcta writes the lower triangle of the 60-sample GRM
        assert sum(1 for _ in gzip.open(grm, "rt")) == 60 * 61 // 2
        # GRM.R format_apex header + sample-ID-mapped rows
        assert apex.read_text().splitlines()[0] == "#id1\tid2\tkinship"
