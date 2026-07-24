"""Notebook tier: splicing_normalization.ipynb workers (leafcutter + psichomics).

splicing_normalization.R ports the leafcutter prepare_phenotype_table.py logic and the
psichomics blocks. Fixtures + references are the 20-sample MWE leafcutter/psichomics outputs.
Float phenotype values are compared NUMERICALLY (identical doubles): Python's shortest-repr
and R's differ cosmetically on ~4% of values in the 16th/17th digit, but the doubles — what
downstream tools parse — are identical.
"""
from __future__ import annotations

import gzip

WORKER = "code/script/molecular_phenotypes/QC/splicing_normalization.R"
FIX = "tests/fixtures/splicing_normalization"


def _rows(text, sep="\t"):
    return [ln.split(sep) for ln in text.splitlines() if ln.strip()]


def _assert_num_match(got_text, exp_text, label_cols, sep="\t", tol=1e-9):
    a, b = _rows(got_text, sep), _rows(exp_text, sep)
    assert len(a) == len(b), f"row count {len(a)} != {len(b)}"
    maxd = 0.0
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert len(ra) == len(rb), f"col count differs at row {i}"
        for j, (x, y) in enumerate(zip(ra, rb)):
            if i == 0 or j < label_cols or x == "NA" or y == "NA":   # header/labels/NA: exact
                assert x == y, f"mismatch row {i} col {j}: {x!r} != {y!r}"
            else:
                maxd = max(maxd, abs(float(x) - float(y)))
    assert maxd < tol, f"max numeric diff {maxd:.2e} exceeds {tol}"


def test_prepare_phenotype(run_r, repo_root, tmp_path):
    counts = tmp_path / "leafcutter_perind.counts.gz"
    counts.write_bytes((repo_root / FIX / "leafcutter_perind.counts.gz").read_bytes())
    p = run_r(repo_root / WORKER, ["--step", "prepare_phenotype", "--input", counts])
    assert p.returncode == 0, p.stdout + p.stderr
    for out, exp in [("leafcutter_perind.counts.gz.phen_chr22", "expected.phen_chr22.gz"),
                     ("leafcutter_perind.counts.gz.qqnorm_chr22", "expected.qqnorm_chr22.gz")]:
        _assert_num_match((tmp_path / out).read_text(),
                          gzip.open(repo_root / FIX / exp, "rt").read(), label_cols=4)
    assert (tmp_path / "leafcutter_perind.counts.gz_phenotype_file_list.txt").exists()


def test_qqnorm(run_r, repo_root, tmp_path):
    out = tmp_path / "qqnorm.txt"
    p = run_r(repo_root / WORKER,
              ["--step", "qqnorm", "--input", repo_root / FIX / "raw_data.txt.gz", "--output", out,
               "--na-rate", 0.4, "--min-variance", 0.001])
    assert p.returncode == 0, p.stdout + p.stderr
    _assert_num_match(out.read_text(),
                      gzip.open(repo_root / FIX / "expected_raw_data.qqnorm.txt.gz", "rt").read(),
                      label_cols=4)


def test_psichomics_bed(run_r, repo_root, tmp_path):
    out = tmp_path / "bedded.txt"
    p = run_r(repo_root / WORKER,
              ["--step", "psichomics_bed", "--input", repo_root / FIX / "psi_raw_data.tsv.gz",
               "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    _assert_num_match(out.read_text(),
                      gzip.open(repo_root / FIX / "expected_bedded.txt.gz", "rt").read(), label_cols=4)
