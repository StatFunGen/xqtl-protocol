# LeafCutter2 reference fixture

chr22 GTF + FASTA pair for `test_cluster_and_classify` in
`tests/notebooks/molecular_phenotypes/calling/test_splicing_calling_leafcutter2.py`.
LeafCutter2 classifies only junctions inside a gene with CDS / start_codon /
stop_codon records and reads those genes' sequence, so the pair covers the three
clusters the `SAMPLE_00{1,2}.junc.gz` fixtures keep at `--minclureads 3`
(two intergenic, one inside TTC28). Do **not** remove without migrating that test.

| File | Derivation |
|---|---|
| `Homo_sapiens.GRCh38.103.chr22.leafcutter2.reformatted.gtf` | Every Ensembl 103 record of the 21 chr22 genes overlapping those clusters, then `hg_gtf_1` (chr prefix, `transcript_biotype` -> `transcript_type`) |
| `chr22.leafcutter2.fa.gz` (+ `.fai`, `.gzi`) | Full-length chr22 from the protocol FASTA, N everywhere except those genes and clusters (+/- 1 kb); bgzipped |

## Rebuild

```bash
# Ensembl release 103 GTF (checksum in the same directory's CHECKSUMS)
curl -O https://ftp.ensembl.org/pub/release-103/gtf/homo_sapiens/Homo_sapiens.GRCh38.103.chr.gtf.gz
D=tests/fixtures/splicing_calling/leafcutter/reference
Rscript $D/build_reference.R \
  --ensembl-gtf Homo_sapiens.GRCh38.103.chr.gtf.gz \
  --fasta <reference_data>/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta \
  --outdir $D
zcat $D/chr22.leafcutter2.fa.gz | head -n1 > hdr.fa
Rscript code/script/reference_data/reference_data_preparation.R --step hg_gtf_1 \
  --hg-reference hdr.fa \
  --hg-gtf $D/Homo_sapiens.GRCh38.103.chr22.leafcutter2.gtf \
  --output $D/Homo_sapiens.GRCh38.103.chr22.leafcutter2.reformatted.gtf
rm hdr.fa $D/Homo_sapiens.GRCh38.103.chr22.leafcutter2.gtf
```
