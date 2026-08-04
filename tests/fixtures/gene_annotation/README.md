# gene_annotation test fixtures

Committed chr22 fixtures for the `gene_annotation` notebook tests, derived (no synthesis)
from the protocol-example MWE. Do **not** remove without migrating
`tests/notebooks/data_preprocessing/phenotype/`.

| File | Purpose | Derivation |
|---|---|---|
| `Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz` | `annotate_coord` (gene/protein) coordinate source | collapsed gene-model GTF subset to `chr22`, gzipped |
| `Homo_sapiens.GRCh38.103.chr22.exon.gtf.gz` | leafcutter/psichomics exon source | full chr22 GTF subset to the 109 genes the toy introns overlap ∪ psichomics genes |
| `protocol_example.rnaseq.bed.gz` | `annotate_coord` gene matrix | copied as-is (chr22) |
| `protocol_example.protein.no_coord.tsv` | `annotate_coord` protein matrix | protein `no_coord` subset to chr22 genes (19 rows) |
| `protocol_example.leafcutter.intron_count.tsv` | `map_leafcutter_cluster_to_gene` | copied as-is |
| `protocol_example.leafcutter.phenotype.bed.gz` | `annotate_leafcutter_isoforms` | copied as-is |
| `protocol_example.psichomics.phenotype.tsv` | `annotate_psichomics_isoforms` | copied as-is |
| `protocol_example.rnaseq.gene_ID.tsv` | `annotate_coord_biomart` (e2e; needs network) | copied as-is |
| `protocol_example.atac.tsv` | `annotate_coord` (atac) matrix | 10 chr22 peaks × 10 samples, from the MWE `proteomics/peaks_split` beds |
| `protocol_example.atac.coordinate_index.tsv` | `annotate_coord` (atac) coordinate index | `ID/#chr/start/end` for those 10 peaks |
