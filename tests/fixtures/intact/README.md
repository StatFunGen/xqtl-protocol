# INTACT integration fixture

Tiny hand-authored inputs for the `intact` step of
`code/SoS/pecotmr_integration/intact.ipynb`, which joins PTWAS + fastenloc output
and runs the Bioconductor **INTACT** package.

There is no upstream PTWAS or fastenloc data in this repo to derive from (the
fastenloc notebooks are in `code/SoS/graveyard/`), so these are synthesized. They
are the minimal tables INTACT needs and are shaped to exercise the wrapper's
logic:

- `protocol_example.ptwas.output` — PTWAS output, TSV with `GENE`, `STAT` (TWAS
  z-score), `SUBCLASS`. Seven rows: five distinct genes, one duplicate
  `(GENE, SUBCLASS)` row (collapsed by the notebook's `distinct()`), and one gene
  (`ENSG00000099968`) absent from fastenloc (dropped by the `inner_join`).
- `protocol_example.fastenloc.gene.out` — fastenloc output, whitespace table with
  `Gene`, `GLCP` (gene-level colocalization probability in [0, 1]).

The two strong genes (high `|STAT|` + high `GLCP`) come back FDR-significant and
the three weak ones do not, so `fdr_sig` carries both TRUE and FALSE — a guard on
the `alpha` argument reaching `fdr_rst` as a number rather than a string.
