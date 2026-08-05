"""Notebook tier: genotype_alignment.ipynb — align multi-cohort .bim tables.

Worker: code/script/misc/genotype_alignment.R (vroom + dplyr; Rsamtools bgzip/tabix).
For one chromosome it flips variants whose alleles are the reference cohort's swap and
writes the de-duplicated union (chr,pos,alt,ref) as a bgzip+tabix'd 4-column table; the
first cohort is the reference.

Fixtures (derived from qtl_mini/example.chr22.bim): two cohorts of chr22 A/T variants —
cohortB carries 2 variants written as the T/A swap (must flip back to A/T), 2 identical
overlaps, and 2 new variants — so the merged union is 8 variants, all A/T, pos-sorted.
"""
from __future__ import annotations

import gzip

R = "code/script/misc/genotype_alignment.R"
NB = "pipeline/genotype_alignment.ipynb"
FIX = "tests/fixtures/genotype_alignment"
MSD = "code/script"
COHORT_A = FIX + "/cohortA/cohortA.22.bim"
COHORT_B = FIX + "/cohortB/cohortB.22.bim"

EXPECTED = [
    ["22", "10414272", "A", "T"], ["22", "10416111", "A", "T"],
    ["22", "10416669", "A", "T"], ["22", "10420511", "A", "T"],
    ["22", "10424460", "A", "T"], ["22", "10424703", "A", "T"],
    ["22", "10425071", "A", "T"], ["22", "10425470", "A", "T"],
]


def _rows(path):
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]


def test_align_worker(run_r, repo_root, tmp_path):
    out = tmp_path / "aligned.22.bim.gz"
    p = run_r(repo_root / R, ["--output", out, repo_root / COHORT_A, repo_root / COHORT_B])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.exists() and (tmp_path / (out.name + ".tbi")).exists()
    # union of both cohorts, cohortB's T/A swaps flipped back to A/T, pos-sorted
    assert _rows(out) == EXPECTED


def test_align_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: discover per-cohort bims, group by chromosome, align to the first cohort.
    p = run_sos(repo_root / NB, "genotype_alignment", {
        "geno_list_paths": [repo_root / FIX / "cohortA", repo_root / FIX / "cohortB"],
        "cwd": tmp_path, "name": "test",
        "modular_script_dir": repo_root / MSD})
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "test.cohortA.cohortB.22.aligned.bim.gz"
    assert out.exists() and (tmp_path / (out.name + ".tbi")).exists()
    assert _rows(out) == EXPECTED
