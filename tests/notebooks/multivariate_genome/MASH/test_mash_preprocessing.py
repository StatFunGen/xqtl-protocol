"""Notebook tier: mash_preprocessing.ipynb susie_to_mash reads a fine_mapping_meta
of FineMappingResult RDS -> mash_preprocessing.R -> mash_input.rds."""
from __future__ import annotations


def test_mash_preprocessing_susie_to_mash(run_sos, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/mash"
    cwd = tmp_path / "mp"
    meta = tmp_path / "fmm.tsv"
    meta.write_text("region_id\tsusie_path\n"
                    f"chr22:15528191-15529138\t{fx / 'protocol_example.QtlFineMappingResult.rds'}\n")
    p = run_sos(repo_root / "pipeline/mash_preprocessing.ipynb", "susie_to_mash",
                dict(name="toy_mash", fine_mapping_meta=meta, cwd=cwd,
                     sig_p_cutoff="0.1", n_random="15", n_null="15"),
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    out = cwd / "toy_mash.mash_input.rds"
    assert out.exists(), p.stdout
    names = set(read_rds(out)["names"])
    assert {"strong.z", "random.z", "null.z", "XtX"}.issubset(names)
