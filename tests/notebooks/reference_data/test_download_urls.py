"""Notebook tier: the download steps of reference_data_preparation.ipynb still point at files
that exist.

download_ercc_reference (29 KB) runs for real in test_builder_examples.py. The other three
download several GB each (GRCh38 FASTA, Ensembl GTF, dbSNP VCF), so they stay out of that test;
here each URL in a ``[download_*]`` step is asked for its headers only (``curl -I``), which
fails if the file has moved or been removed. No file is downloaded.

A machine that cannot reach the host at all (no network, a proxy that refuses it) skips the
URL with that reason rather than failing.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

NB = "code/SoS/reference_data/reference_data_preparation.ipynb"
# curl exit codes that mean "could not reach the host", not "the file is gone":
# 5/6 cannot resolve proxy/host, 7 cannot connect, 28 timeout, 35 TLS connect, 56 receive failure
UNREACHABLE = {5, 6, 7, 28, 35, 56}


def _download_urls(repo_root):
    nb = json.loads((repo_root / NB).read_text(encoding="utf-8"))
    urls = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", ""))
        m = re.match(r"\s*\[(download_\w+)\]", src)
        if cell.get("cell_type") != "code" or not m:
            continue
        body = src.split("download:", 1)[1] if "download:" in src else ""
        urls += [(m.group(1), u) for u in re.findall(r"(?:https?|ftp)://\S+", body)]
    return urls


def test_download_steps_list_urls(repo_root):
    steps = {s for s, _ in _download_urls(repo_root)}
    assert steps == {"download_hg_reference", "download_gene_annotation",
                     "download_ercc_reference", "download_dbsnp"}, steps


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_download_urls_exist(repo_root):
    gone, unreachable = [], []
    for step, url in _download_urls(repo_root):
        for _ in range(2):                                   # one retry for a slow server
            p = subprocess.run(["curl", "-sSIL", "--max-time", "60", url],
                               capture_output=True, text=True)
            if p.returncode not in UNREACHABLE:
                break
        if p.returncode in UNREACHABLE:
            unreachable.append(f"{step}: {url} ({p.stderr.strip()[:120]})")
            continue
        heads = p.stdout.replace("\r", "")
        codes = re.findall(r"^HTTP/\S+\s+(\d{3})", heads, re.M)
        size = re.findall(r"^content-length:\s*(\d+)", heads, re.M | re.I)
        ok = p.returncode == 0 and (not codes or codes[-1] == "200") and size and int(size[-1]) > 0
        if not ok:
            gone.append(f"{step}: {url} (curl exit {p.returncode}, status {codes[-1:] or '-'}, "
                        f"size {size[-1:] or '-'}) {p.stderr.strip()[:120]}")
    assert not gone, "download URLs that no longer answer:\n" + "\n".join(gone)
    if unreachable:
        pytest.skip("could not reach: " + "; ".join(unreachable))
