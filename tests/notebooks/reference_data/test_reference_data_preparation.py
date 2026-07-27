"""Notebook tier: reference_data_preparation.ipynb — the steps rewired to R workers.

Workers:
  code/script/reference_data/gff3_to_gtf.R                 -> [gff3_to_gtf]
  code/script/reference_data/reference_data_preparation.R  -> [hg_reference_1] / [hg_gtf_1] / [ercc_gtf] / faidx

Fixtures live in tests/fixtures/reference_data_preparation:
  ERCC92.gtf                        real MWE ERCC annotation (input to ercc_gtf)
  ERCC92.genes.patched.expected.gtf byte-exact reference (original notebook Python logic)
  test_contigs.fa                   tiny multi-contig FASTA exercising the ALT/HLA/decoy name filter
  mini.gff3                         minimal Ensembl-style gene->mRNA->exon/CDS hierarchy

hg_reference_1 / hg_gtf_1 / faidx are *numbered steps* of the multi-step ``hg_reference`` /
``hg_gtf`` workflows, whose downstream steps need kept-external / not-yet-ported tools (picard
CreateSequenceDictionary, collapse_annotation). SoS cannot run a single numbered step in
isolation, so those are exercised at the worker (Rscript) level here; the ``hg_gtf`` workflow
becomes fully SoS-testable once collapse_annotation is ported.
"""
from __future__ import annotations

from pathlib import Path

import gzip

NB = "pipeline/reference_data_preparation.ipynb"
FIX = "tests/fixtures/reference_data_preparation"
GENEFIX = "tests/fixtures/gene_annotation"
WORKER = "code/script/reference_data/reference_data_preparation.R"
COLLAPSE = "code/script/reference_data/collapse_annotation.R"
# 109-gene reformatted chr22 GTF (pre-collapse) and the full-chr22 collapsed reference
CHR22_GTF = GENEFIX + "/Homo_sapiens.GRCh38.103.chr22.exon.gtf.gz"
COLLAPSED_REF = GENEFIX + "/Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz"


def _common(repo_root, out):
    return dict(cwd=out, modular_script_dir=repo_root / "code/script")


def _gene_blocks(text):
    """Map gene_id -> its list of GTF lines (collapse_only is per-gene, so blocks are comparable)."""
    import re
    blocks = {}
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        m = re.search(r'gene_id "([^"]+)"', ln)
        blocks.setdefault(m.group(1), []).append(ln)
    return blocks


def _assert_matches_reference(repo_root, collapsed_text):
    """Every gene block in `collapsed_text` must byte-match the same gene in the reference."""
    got = _gene_blocks(collapsed_text)
    ref = _gene_blocks(gzip.open(repo_root / COLLAPSED_REF, "rt").read())
    assert got, "no collapsed genes produced"
    for gid, lines in got.items():
        assert gid in ref, f"{gid} missing from reference"
        assert lines == ref[gid], f"gene block for {gid} differs from GTEx reference"


def test_ercc_gtf(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "ercc_gtf",
                dict(ercc_gtf=repo_root / FIX / "ERCC92.gtf", **_common(repo_root, out)),
                cwd=repo_root)
    assert p.returncode == 0, p.stdout + p.stderr
    got = (out / "ERCC92.genes.patched.gtf").read_text()
    exp = (repo_root / FIX / "ERCC92.genes.patched.expected.gtf").read_text()
    assert got == exp, "ercc_gtf output differs from the reference (Python) patched GTF"


def test_gff3_to_gtf(run_sos, repo_root, tmp_path):
    out = tmp_path / "out"
    p = run_sos(repo_root / NB, "gff3_to_gtf",
                dict(gff3_file=repo_root / FIX / "mini.gff3", **_common(repo_root, out)),
                cwd=repo_root)
    assert p.returncode == 0, p.stdout + p.stderr
    txt = (out / "mini.gtf").read_text()
    # transcript_id/gene_id propagated down the Parent hierarchy; gene_type kept from biotype
    assert '\ttranscript\t' in txt and '\texon\t' in txt
    assert 'gene_id "ENSGTEST1"' in txt and 'transcript_id "ENSTTEST1"' in txt
    assert 'gene_type "protein_coding"' in txt


def test_hg_reference_1(run_r, repo_root, tmp_path):
    out = tmp_path / "filtered.fasta"
    p = run_r(repo_root / WORKER,
              ["--step", "hg_reference_1", "--input", repo_root / FIX / "test_contigs.fa", "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    heads = [l.split()[0][1:] for l in out.read_text().splitlines() if l.startswith(">")]
    assert heads == ["chr22", "chr1_KI270706v1_random"], heads   # dropped _alt / HLA* / _decoy


def test_faidx(run_r, repo_root, tmp_path):
    fa = tmp_path / "test_contigs.fa"
    fa.write_text((repo_root / FIX / "test_contigs.fa").read_text())
    p = run_r(repo_root / WORKER, ["--step", "faidx", "--input", fa])
    assert p.returncode == 0, p.stdout + p.stderr
    fai = fa.with_suffix(".fa.fai")
    assert fai.exists()
    assert fai.read_text().split("\n")[0].split("\t")[0] == "chr22"


def test_hg_gtf_1(run_r, repo_root, tmp_path):
    # a FASTA whose contigs lack the "chr" prefix -> the GTF's "chr" is stripped to match
    fasta = tmp_path / "nochr.fa"
    fasta.write_text(">22 dna\nACGT\n")
    out = tmp_path / "reformatted.gtf"
    p = run_r(repo_root / WORKER,
              ["--step", "hg_gtf_1", "--hg-reference", fasta, "--hg-gtf", repo_root / CHR22_GTF, "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert lines and all(l.split("\t")[0] == "22" for l in lines[:50]), "chr prefix not stripped"


def test_collapse_annotation(run_r, repo_root, tmp_path):
    """Worker-level byte check: collapse the 109-gene chr22 GTF; each gene block must
    match the GTEx collapsed reference (collapse_only is per-gene, so a subset is comparable)."""
    src = tmp_path / "chr22.gtf"
    src.write_bytes(gzip.open(repo_root / CHR22_GTF, "rb").read())
    out = tmp_path / "collapsed.gtf"
    p = run_r(repo_root / COLLAPSE, ["--collapse-only", "--input", src, "--output", out])
    assert p.returncode == 0, p.stdout + p.stderr
    _assert_matches_reference(repo_root, out.read_text())


def test_hg_gtf_workflow(run_sos, repo_root, tmp_path):
    """Full hg_gtf workflow (hg_gtf_1 reformat -> hg_gtf_2 collapse) via SoS, unlocked now
    that collapse_annotation is ported. Output must match the GTEx collapsed reference per gene."""
    out = tmp_path / "out"
    fasta = tmp_path / "chr.fa"
    fasta.write_text(">chr22 dna\nACGT\n")                       # chr-prefixed: GTF already matches
    p = run_sos(repo_root / NB, "hg_gtf",
                dict(hg_reference=fasta, hg_gtf=repo_root / CHR22_GTF, stranded=True,
                     **_common(repo_root, out)),
                cwd=repo_root)
    assert p.returncode == 0, p.stdout + p.stderr
    res = list(out.glob("*.collapse_only.gene.gtf"))
    assert res, "no collapsed GTF produced by the hg_gtf workflow"
    _assert_matches_reference(repo_root, res[0].read_text())


def test_tad_annotate(run_sos, repo_root, tmp_path):
    """Tad_annotateion via SoS. No MWE data exists for this bespoke step, so the fixtures are
    synthetic (rnaseq region schema) and the expected outputs were validated byte-for-byte
    against the original pandas (the pandas row-index column is intentionally omitted)."""
    out = tmp_path / "out"
    tad = repo_root / FIX / "tad"
    p = run_sos(repo_root / NB, "Tad_annotateion",
                dict(TAD_list=tad / "tad.tsv", region_list=tad / "region.tsv",
                     TAD_genotype=tad / "geno.tsv", **_common(repo_root, out)),
                cwd=repo_root)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "region.annotation").read_text() == (tad / "expected.annotation").read_text()
    assert (out / "region.annotation.genotype_files_list.tsv").read_text() == \
        (tad / "expected.genotype_files_list.tsv").read_text()


def test_psichomics_hg38_annotation(run_sos, repo_root, tmp_path):
    """psi_hg38_annotation via SoS: reconcile the psichomics AH63657 splicing-annotation gene
    symbols to Ensembl IDs (GTF -> HGNC -> VAST -> SUPPA). Needs an AnnotationHub network download
    (cached after first run, like the sesame test); skips gracefully when offline. HGNC db is a
    trimmed chr22 fixture, GTF is the chr22 collapsed gene model."""
    import pytest
    out = tmp_path / "out"
    gtf = tmp_path / "collapsed.gtf"
    gtf.write_bytes(gzip.open(repo_root / COLLAPSED_REF, "rb").read())
    hgnc = tmp_path / "hgnc.tsv"
    hgnc.write_bytes(gzip.open(repo_root / FIX / "hgnc_chr22.tsv.gz", "rb").read())
    p = run_sos(repo_root / NB, "psi_hg38_annotation",
                dict(hg_gtf=gtf, hgrc_db=hgnc, **_common(repo_root, out)),
                cwd=repo_root, timeout=1200)
    blob = p.stdout + p.stderr
    if p.returncode != 0 and any(k in blob for k in
                                 ("AnnotationHub", "download", "Timeout", "curl", "cache", "URL")):
        pytest.skip("psichomics AnnotationHub network unavailable")
    assert p.returncode == 0, blob
    rds = out / "psichomics_hg38_annotation.rds"
    assert rds.exists() and rds.stat().st_size > 50_000, "annotation RDS missing/too small"


def test_suppa_annotation(run_sos, repo_root, tmp_path):
    """SUPPA_annotation workflow: suppa.py generateEvents (SUPPA_annotation_1) ->
    reference_data_preparation.R --step suppa_annot / psichomics (SUPPA_annotation_2).
    Both external SUPPA + R psichomics are exercised on the committed chr22 GTF."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    gtf = tmp_path / "chr22.gtf"
    gtf.write_text(gzip.open(repo_root / CHR22_GTF, "rt").read())
    p = run_sos(repo_root / NB, "SUPPA_annotation",
                {**_common(repo_root, out), "hg_gtf": gtf}, cwd=repo_root, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "hg38.chr22_SE_strict.ioe").exists()             # SUPPA_annotation_1 output
    assert (out / "chr22.SUPPA_annotation.rds").exists()           # SUPPA_annotation_2 output


def test_rsem_index(run_sos, repo_root, tmp_path):
    """RSEM_index: rsem-prepare-reference on a tiny self-consistent genome+GTF."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    p = run_sos(repo_root / NB, "RSEM_index",
                {**_common(repo_root, out),
                 "hg_reference": repo_root / FIX / "rsem_mini.fa",
                 "hg_gtf": repo_root / FIX / "rsem_mini.gtf", "numThreads": 1},
                cwd=repo_root, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    idx = out / "RSEM_Index"
    for f in ("rsem_reference.grp", "rsem_reference.ti", "rsem_reference.seq",
              "rsem_reference.transcripts.fa", "rsem_reference.idx.fa"):
        assert (idx / f).exists(), f"missing {f}"


def test_picard_sequence_dictionary(repo_root, tmp_path):
    """hg_reference_3 picard half: picard CreateSequenceDictionary -> .dict."""
    import subprocess
    fa = tmp_path / "test_contigs.fa"
    fa.write_text((repo_root / FIX / "test_contigs.fa").read_text())
    out = tmp_path / "test_contigs.dict"
    p = subprocess.run(["pixi", "run", "--frozen", "picard", "CreateSequenceDictionary",
                        f"R={fa}", f"O={out}"], cwd=repo_root, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = out.read_text().splitlines()
    assert lines[0].startswith("@HD")
    assert any(l.startswith("@SQ") and "SN:chr22" in l for l in lines)


def test_refflat_generation(run_sos, repo_root, tmp_path):
    """RefFlat_generation: gtfToGenePred (kent-tools) + awk reshape -> refFlat.
    Byte-exact vs the committed ERCC reference."""
    gtf = tmp_path / "ERCC92.gtf"
    gtf.write_text((repo_root / FIX / "ERCC92.gtf").read_text())
    p = run_sos(repo_root / NB, "RefFlat_generation",
                {"hg_gtf": gtf, "cwd": tmp_path}, cwd=repo_root, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    out = tmp_path / "ERCC92.ref.flat"
    assert out.read_text() == (repo_root / FIX / "expected_ERCC92.ref.flat").read_text()
