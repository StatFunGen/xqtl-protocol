"""Notebook tier: ld_prune_reference.ipynb.

Two steps:
  [LD_pruning_1]  plink2 --indep-pairwise per block            (external tool)
  [LD_pruning_2]  code/script/reference_data/ld_prune_reference.R  (variant-id reshape)

Fixtures (tests/fixtures/ld_prune_reference), derived from the MWE:
  genotype/protocol_example.ld_genotype.chr22.{bed,bim,fam}  the LD reference panel
  prune_in/chr22_*.prune.in                                  plink2's 3 per-block outputs
  expected/LD_pruned_variants.txt                            the MWE reference (both cells)

The worker test byte-checks the R reshape against the MWE reference. The workflow test
runs the whole `LD_pruning` workflow (plink2 + R) via `sos run`; our plink2 reproduces the
MWE `.prune.in` byte-for-byte, so the final table matches exactly.
"""
from __future__ import annotations

import csv

NB = "pipeline/ld_prune_reference.ipynb"
FIX = "tests/fixtures/ld_prune_reference"
WORKER = "code/script/reference_data/ld_prune_reference.R"
BLOCKS = ("chr22_10516173_17414263", "chr22_17414263_19608021", "chr22_19608021_22004675")


def test_ld_pruning_2_worker(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    prune = sorted((fix / "prune_in").glob("*.prune.in"))
    out = tmp_path / "LD_pruned_variants.txt"
    p = run_r(repo_root / WORKER, ["--input", *prune, "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_text() == (fix / "expected" / "LD_pruned_variants.txt").read_text()


def test_ld_pruning_workflow(run_sos, repo_root, tmp_path):
    fix = repo_root / FIX
    geno = fix / "genotype" / "protocol_example.ld_genotype.chr22.bed"
    # genotype_list: header row (consumed, pandas header=0 parity) + one row per block
    gl = tmp_path / "genotype.list"
    with open(gl, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["block", "geno"])
        for b in BLOCKS:
            w.writerow([b, str(geno)])
    p = run_sos(repo_root / NB, "LD_pruning", {
        "genotype_list": gl, "cwd": tmp_path,
        "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "LD_pruning" / "LD_pruned_variants.txt"
    assert out.read_text() == (fix / "expected" / "LD_pruned_variants.txt").read_text()
