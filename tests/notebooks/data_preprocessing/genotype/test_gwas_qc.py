"""Notebook tier: GWAS_QC.ipynb — the king_3 step moved into GWAS_QC.sh (king_split).

king_split takes the related-sample list (--remove-samples) and writes an unrelated
subset (related removed) + a related subset (related kept); when the list is empty it
writes an empty related placeholder. Exercised via the worker directly (the notebook
king_3 chains off king_1/king_2 sos_variables that aren't reproducible in isolation).

Fixtures: the committed 60-sample chr22 PLINK bed under tests/fixtures/ld_prune_reference.
"""
from __future__ import annotations

from helpers.expected import assert_matches_expected

SH = "code/script/data_preprocessing/genotype/GWAS_QC.sh"
GR = "code/script/data_preprocessing/genotype/GWAS_QC.R"
NB = "pipeline/GWAS_QC.ipynb"
BED = "tests/fixtures/ld_prune_reference/genotype/protocol_example.ld_genotype.chr22"
GFIX = "tests/fixtures/gwas_qc"
EXP = f"{GFIX}/expected"


def _fam_count(path):
    return sum(1 for _ in open(path))


def _args(tmp_path, bed, related):
    return ["king_split", "--cwd", tmp_path, "--genoFile", bed,
            "--remove-samples", related, "--out-unrelated", tmp_path / "g.unrelated",
            "--out-related", tmp_path / "g.related", "--related-output", tmp_path / "g.related.bed",
            "--numThreads", "1"]


def test_king_split(run_sh, repo_root, tmp_path):
    bed = repo_root / f"{BED}.bed"
    fam = repo_root / f"{BED}.fam"
    with open(fam) as f:
        first5 = ["\t".join(l.split()[:2]) for l in f][:5]
    related = tmp_path / "related.txt"
    related.write_text("\n".join(first5) + "\n")
    p = run_sh(repo_root / SH, _args(tmp_path, bed, related))
    assert p.returncode == 0, p.stdout + p.stderr
    assert _fam_count(tmp_path / "g.unrelated.fam") == 55       # 60 - 5 related removed
    assert _fam_count(tmp_path / "g.related.fam") == 5          # the 5 related kept
    # regression: the exact unrelated/related sample partitions (deterministic plink2 --remove/--keep).
    assert_matches_expected(tmp_path / "g.unrelated.fam", repo_root / EXP / "king_split.unrelated.fam", mode="exact")
    assert_matches_expected(tmp_path / "g.related.fam", repo_root / EXP / "king_split.related.fam", mode="exact")


def test_king_split_no_related(run_sh, repo_root, tmp_path):
    bed = repo_root / f"{BED}.bed"
    related = tmp_path / "related.txt"
    related.write_text("")                                     # no related individuals
    p = run_sh(repo_root / SH, _args(tmp_path, bed, related))
    assert p.returncode == 0, p.stdout + p.stderr
    assert _fam_count(tmp_path / "g.unrelated.fam") == 60       # everyone kept
    assert (tmp_path / "g.related.bed").exists()               # empty placeholder touched
    # regression: with no related samples the unrelated set is the full 60-sample
    # cohort unchanged (deterministic pass-through).
    assert_matches_expected(tmp_path / "g.unrelated.fam", repo_root / f"{BED}.fam", mode="exact")


def test_king_kinship(run_sh, repo_root, tmp_path):
    # the `king` step runs plink2 --make-king-table on the 60-sample bed.
    bed = repo_root / f"{BED}.bed"
    p = run_sh(repo_root / SH, ["king", "--cwd", tmp_path, "--genoFile", bed,
                                "--out-prefix", tmp_path / "out", "--kinship", "0.0625",
                                "--kin-maf", "0.01", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    kin0 = tmp_path / "out.kin0"
    assert kin0.exists()
    header = kin0.read_text().splitlines()[0].split("\t")
    assert header[:4] == ["#FID1", "IID1", "FID2", "IID2"] and "KINSHIP" in header
    # regression: the kinship table (FID/IID pairs + NSNP/HETHET/IBS0/KINSHIP)
    # reproduces the committed fixture; the estimate columns are floats, compared
    # within tolerance so plink2 build differences across CI don't false-fail.
    assert_matches_expected(kin0, repo_root / EXP / "king.kin0", mode="tolerant", rtol=1e-6, atol=1e-8)


def test_qc_no_prune(run_sh, repo_root, tmp_path):
    # basic variant QC filters (MAC/MAF/missingness) via plink2 -> a filtered bed.
    bed = repo_root / f"{BED}.bed"
    p = run_sh(repo_root / SH, ["qc_no_prune", "--cwd", tmp_path, "--genoFile", bed,
                                "--out-prefix", tmp_path / "qc", "--mac-filter", "5",
                                "--maf-filter", "0.01", "--geno-filter", "0.1", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    out_bim = tmp_path / "qc.bim"
    assert out_bim.exists()
    n_out = sum(1 for _ in open(out_bim))
    n_in = sum(1 for _ in open(repo_root / f"{BED}.bim"))
    assert 0 < n_out < n_in                                    # filters removed variants
    # regression: the exact surviving-variant .bim (deterministic MAC/MAF/geno filters).
    assert_matches_expected(out_bim, repo_root / EXP / "qc_no_prune.bim", mode="exact")


def test_king_2_filter_relatedness(run_r, repo_root, tmp_path):
    # GWAS_QC.R king_2: parse the KING kinship table -> related individuals to remove.
    out = tmp_path / "related_id"
    p = run_r(repo_root / GR, ["--step", "king_2", "--input", repo_root / f"{GFIX}/protocol_example.kin0",
                               "--output", out, "--kinship", "0.0625"])
    assert p.returncode == 0, p.stdout + p.stderr
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert lines and all(len(l.split()) == 2 for l in lines)   # FID IID per related sample
    # regression: the exact set of related FID/IID pairs flagged for removal
    # (deterministic greedy prune over the committed kinship table).
    assert_matches_expected(out, repo_root / EXP / "king_2.related_id", mode="exact")


def test_qc_ld_prune(run_sh, repo_root, tmp_path):
    # the `qc` step = filters + LD pruning -> a pruned bed + a .prune.in list.
    bed = repo_root / f"{BED}.bed"
    p = run_sh(repo_root / SH, ["qc", "--cwd", tmp_path, "--genoFile", bed,
                                "--prune-prefix", tmp_path / "qc", "--out-prefix", tmp_path / "qc",
                                "--mac-filter", "5", "--window", "50", "--shift", "10", "--r2", "0.1",
                                "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "qc.bed").exists()
    assert _fam_count(tmp_path / "qc.prune.in") > 0
    # regression: the exact LD-pruned variant set (.prune.in) and the pruned .bim
    # (deterministic plink2 --indep-pairwise + --extract).
    assert_matches_expected(tmp_path / "qc.prune.in", repo_root / EXP / "qc_ld_prune.prune.in", mode="exact")
    assert_matches_expected(tmp_path / "qc.bim", repo_root / EXP / "qc_ld_prune.bim", mode="exact")


def test_sample_overlap(run_sh, repo_root, tmp_path):
    # genotype ∩ phenotype samples (pheno matrix: sample IDs are columns 5+).
    bed = repo_root / f"{BED}.bed"
    pheno = repo_root / f"{GFIX}/protocol_example.pheno.bed"
    p = run_sh(repo_root / SH, ["genotype_phenotype_sample_overlap", "--cwd", tmp_path,
                                "--genoFile", bed, "--phenoFile", pheno, "--name", "ov", "--numThreads", "1"])
    assert p.returncode == 0, p.stdout + p.stderr
    ov = tmp_path / "protocol_example.pheno.bed.sample_overlap.txt"
    assert ov.exists()
    lines = ov.read_text().splitlines()
    assert lines[0] == "genotype_id\tsample_id"
    assert len(lines) - 1 == 60                                # all 60 geno samples present in pheno
    # regression: the exact genotype<->phenotype sample-ID overlap table (deterministic set intersection).
    assert_matches_expected(ov, repo_root / EXP / "sample_overlap.txt", mode="exact")


def test_king_workflow_via_sos(run_sos, repo_root, tmp_path):
    # SoS wiring: `sos run GWAS_QC.ipynb king` chains king_1 (kinship) -> king_2
    # (GWAS_QC.R relatedness) -> king_3 (king_split) into the unrelated/related split.
    p = run_sos(repo_root / NB, "king", {
        "genoFile": repo_root / f"{BED}.bed", "cwd": tmp_path, "name": "t",
        "modular_script_dir": repo_root / "code/script"})
    assert p.returncode == 0, p.stdout + p.stderr
    unrel = list(tmp_path.glob("*.unrelated.bed"))
    assert len(unrel) == 1
    # regression: the full king chain (king_1 kinship -> king_2 relatedness -> king_3
    # split) yields the same unrelated cohort; compare the .fam sample partition.
    assert_matches_expected(unrel[0].with_suffix(".fam"),
                            repo_root / EXP / "king_workflow.unrelated.fam", mode="exact")
