# Torino fixture

Inputs for `isoform_calling_torino.ipynb` and
`tests/notebooks/molecular_phenotypes/calling/test_isoform_calling_torino.py`.
Torino learns coverage components from variation in read depth across samples, so
it needs a cohort rather than the single-sample alignments other fixtures ship.
These 24 samples are simulated over one gene, so the truth is known.

| File | Content |
|---|---|
| `bam/SAMPLE_0{01..24}.bam` (+ `.bai`) | Single-end, unstranded 100-base reads over AP000547.3 (ENSG00000283633, chr22:16601911-16615111, +), about 1,000 to 2,000 reads per sample, with reference sequence from `../leafcutter/reference/chr22.leafcutter2.fa.gz` |
| `protocol_example.sample_list.txt` | The manifest naming them, `strand_missing` throughout |
| `protocol_example.genes.bed` | AP000547.3 and AP000547.4; the second has no reads and is dropped by the coverage filter |
| `simulated_isoform_proportions.tsv` | Per sample: group, read count, and the proportion of reads drawn from each transcript |
| `build_fixture.R` | Regenerates all of the above |

Reads are drawn uniformly along three Ensembl 103 transcripts of AP000547.3, with
exon coordinates from `../leafcutter/reference/Homo_sapiens.GRCh38.103.chr22.leafcutter2.reformatted.gtf`:

- `spliced`: ENST00000592107, four exons, every intron spliced
- `intron_retention`: ENST00000592918, which keeps the last intron of ENST00000592107 (16614179-16614698) in its final exon
- `alt_last_exon`: ENST00000591299, ending in an alternative last exon that starts at 16613985

Samples alternate between group A (intron retention about 0.15) and group B (about
0.55); the alternative last exon is spread from 0.05 to 0.40 independently of group.

## Rebuild

```bash
D=tests/fixtures/splicing_calling/torino
Rscript $D/build_fixture.R \
  --fasta tests/fixtures/splicing_calling/leafcutter/reference/chr22.leafcutter2.fa.gz \
  --outdir $D
```

Needs R with optparse, and samtools on `PATH`. The output is deterministic for a
given `--seed`.
