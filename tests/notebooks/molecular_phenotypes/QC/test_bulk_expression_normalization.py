"""Notebook tier: bulk_expression_normalization.ipynb worker.

bulk_expression_normalization.R ports the GTEx eqtl_prepare_expression logic
(previously bulk_expression_normalization.py, which depended on the Broad `qtl`
package) to edgeR/limma + rtracklayer + Rsamtools. Fixtures are the 60-sample MWE
RNA-seq TPM/count GCTs, the chr22 gene GTF, and the sample->participant lookup.

The committed MWE reference `.expression.bed.gz` predates the current pipeline
convention and carries a bare `chr` header; the expected fixture here restores the
`#chr` header the `.py`/pipeline actually produce (every other phenotype .bed.gz
in the MWE uses `#chr`). Phenotype values are compared NUMERICALLY: the edgeR TMM
+ voom transform + inverse-normal transform reproduce the reference to machine
epsilon (~7e-16); the shortest-repr text differs only cosmetically.
"""
from __future__ import annotations

import gzip

WORKER = "code/script/molecular_phenotypes/QC/bulk_expression_normalization.R"
FIX = "tests/fixtures/bulk_expression_normalization"


def _rows(text, sep="\t"):
    return [ln.split(sep) for ln in text.splitlines() if ln.strip()]


def _assert_num_match(got_text, exp_text, label_cols=4, sep="\t", tol=1e-9):
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


def test_normalize_tmm_cpm_voom(run_r, repo_root, tmp_path):
    fix = repo_root / FIX
    p = run_r(repo_root / WORKER,
              ["--step", "normalize", "--cwd", tmp_path,
               "--tpm-gct", fix / "input.tpm.gct.gz",
               "--counts-gct", fix / "input.geneCount.gct.gz",
               "--annotation-gtf", fix / "annotation.gene.gtf",
               "--sample-participant-lookup", fix / "sample_participant_lookup.txt",
               "--count-threshold", 1, "--quantile-normalize"])
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "input.tmm_cpm_voom.expression.bed.gz"
    assert out.exists() and (tmp_path / (out.name + ".tbi")).exists(), "missing bgzip/tabix output"
    _assert_num_match(gzip.open(out, "rt").read(),
                      gzip.open(fix / "expected.tmm_cpm_voom.expression.bed.gz", "rt").read())
