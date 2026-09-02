"""Notebook tier: mash_preprocessing.ipynb susie_to_mash reads a fine_mapping_meta
of FineMappingResult RDS -> mash_preprocessing.R -> mash_input.rds."""
from __future__ import annotations


def test_mash_preprocessing_susie_to_mash(run_sos, read_rds, repo_root, tmp_path):
    cwd = tmp_path / "mp"
    # the committed meta (repo-relative susie_path) is what the module docs tell users to pass
    meta = repo_root / "tests/fixtures/qtl_mini/fine_mapping_meta.tsv"
    p = run_sos(repo_root / "pipeline/mash_preprocessing.ipynb", "susie_to_mash",
                dict(name="toy_mash", fine_mapping_meta=meta, cwd=cwd,
                     sig_p_cutoff="0.1", n_random="15", n_null="15",
                     modular_script_dir=repo_root / "code/script"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    out = cwd / "toy_mash.mash_input.rds"
    assert out.exists(), p.stdout
    names = set(read_rds(out)["names"])
    assert {"strong.z", "random.z", "null.z", "XtX"}.issubset(names)
