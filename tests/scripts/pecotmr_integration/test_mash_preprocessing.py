"""Script tier: mash_preprocessing.R -> mash_input.rds via pecotmr::mashInput.

Covers both object classes mashInput dispatches on:
  * QtlSumStats       -- chained from mash_sumstats_construct.R (strong = the
                         most significant variant per context).
  * FineMappingResult -- the committed protocol_example.QtlFineMappingResult.rds
                         (strong = the lead variant of each credible set).

The assembled list carries the flat strong/random/null .b/.s/.z matrices plus
the strong XtX cross-product. sig_p_cutoff is loosened to 0.1 because the toy
region's signal is weak (the production default 1e-6 would drop every strong
variant on this fixture).
"""
from __future__ import annotations

MASH_KEYS = {
    "strong.b", "strong.s", "strong.z",
    "random.b", "random.s", "random.z",
    "null.b", "null.s", "null.z", "XtX",
}


def test_mash_preprocessing_qtlsumstats_path(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/mash"
    qss = tmp_path / "region1.qss.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_sumstats_construct.R",
              ["--tensorqtl-paths", fx / "Mic_De_Jager_eQTL.tsv",
               fx / "Ast_De_Jager_eQTL.tsv",
               "--conditions", "Mic_De_Jager_eQTL,Ast_De_Jager_eQTL",
               "--region", "chr22:15528191-15529138",
               "--study", "protocol_example", "--output", qss], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr

    out = tmp_path / "mash_input.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_preprocessing.R",
              ["--objects", qss, "--n-random", "15", "--n-null", "15",
               "--sig-p-cutoff", "0.1", "--seed", "1", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "list"
    assert MASH_KEYS.issubset(set(probe["names"]))


def test_mash_preprocessing_finemapping_path(run_r, read_rds, repo_root, tmp_path):
    fx = repo_root / "tests/fixtures/mash"
    out = tmp_path / "mash_input_fmr.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_preprocessing.R",
              ["--objects", fx / "protocol_example.QtlFineMappingResult.rds",
               "--n-random", "15", "--n-null", "15", "--coverage", "0.95",
               "--sig-p-cutoff", "0.1", "--seed", "1", "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "list"
    assert MASH_KEYS.issubset(set(probe["names"]))


def test_mash_preprocessing_independent_variant_list(run_r, read_rds, repo_root, tmp_path):
    """--independent-variant-list restricts the random/null background (correctness
    of the restriction is covered by pecotmr's testthat; here we drive the wiring:
    file read -> mashInput(independentVariants=...) -> valid mash input)."""
    fx = repo_root / "tests/fixtures/mash"
    out = tmp_path / "mash_input_indep.rds"
    p = run_r(repo_root / "code/script/pecotmr_integration/mash_preprocessing.R",
              ["--objects", fx / "protocol_example.QtlFineMappingResult.rds",
               "--n-random", "10", "--n-null", "10", "--coverage", "0.95",
               "--sig-p-cutoff", "0.1", "--seed", "1",
               "--independent-variant-list", fx / "independent_variants.tsv",
               "--output", out], timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    probe = read_rds(out)
    assert probe["class"] == "list"
    assert MASH_KEYS.issubset(set(probe["names"]))
