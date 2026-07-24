# phenotype_formatting test fixtures

Committed chr22 fixtures for the `phenotype_formatting` notebook tests, derived (no synthesis)
from the protocol-example MWE. Do **not** remove without migrating the tests.

| File | Purpose | Derivation |
|---|---|---|
| `protocol_example.rnaseq.bed.bed.gz` (+`.tbi`) | `phenotype_by_chrom` / `by_region` / `by_chrom_gct` input | copied as-is (chr22, tabix-indexed) |
| `regions.txt` | `phenotype_by_region` region list | two windows derived from the bed's own coordinate span |
| `protocol_example.tpm.gct.gz` | `gct_extract_samples` input | copied as-is |
| `keep_samples.txt` | `gct_extract_samples` keep-list | subset of the gct's own sample columns |
| `tad_list.txt` | `phenotype_annotate_by_tad` input | chr22 TADs from the MWE generalized TAD list + a header/`index` column |
| `protocol_example.chr22_16M_17M.bam` (+`.bai`) | `bam_subsetting` input | MWE rnaseq BAM subset to chr22:16–17M (44 KB) + index |
