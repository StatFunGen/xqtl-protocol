# methylation_calling fixtures

## Input

No IDATs are committed. The notebook's original `data/MWE/` was never provided, so
`tests/notebooks/molecular_phenotypes/calling/test_methylation_calling.py` sources
IDATs from the installed **minfiData** R package (450K, matching the installed
manifest). The `sesame` path additionally needs a one-time sesameData cache, which
is fetched from ExperimentHub on first run, so this test is not fully offline.

## Expected output

`expected/` holds a **chr22 subset** of a full SeSAMe run on the six minfiData
samples, produced with:

```bash
zcat <full>.sesame.{beta,M}.bed.gz | awk 'NR==1 || $1=="chr22"' | bgzip -c > <out>
tabix -f -p bed <out>
awk 'NR==1 || $2=="chr22"' <full>.sesame.gene_id.annot.tsv > <out>
```

| file | contents |
|---|---|
| `*.sesame.beta.bed.gz` (+ `.tbi`) | beta values, 15,807 chr22 probes x 13 samples |
| `*.sesame.M.bed.gz` (+ `.tbi`) | M values, same probes and samples |
| `*.sesame.gene_id.annot.tsv` | probe-to-gene annotation, chr22 |
| `*.sample_qcs.sesame.tsv` | per-sample QC metrics, genome-wide (unsubset, 10 KB) |

The full genome-wide run is 658 MB, which is why only chr22 is committed here --
the same convention the other calling fixtures use. Regenerate by rerunning
`methylation_calling.ipynb sesame` and reapplying the commands above.
