# mnm_regression expected outputs

| file | produced by |
|---|---|
| `univariate_bvsr.rds` | `susie_twas` — per-context SuSiE fine-mapping (`method = "susie"`) |
| `univariate_twas_weights.rds` | `susie_twas` — TWAS weights from the same run |
| `multicontext_bvsr.rds` | `mnm` — joint multi-context fit (`method = "mvsusie"`, carries a `jointContexts` column) |

All three are keyed to study `test_study`, gene `ENSG00000283047`, contexts
`context1`/`context2`, and are generated with `--seed 1`.

## Regenerating

`mnm` reads the QtlDataset that `qtl_dataset_construct` writes, so run both steps
together — `mnm` cannot run alone. `--transpose-covariates True` is required for the
QTLtools-format toy covariates; without it the run fails with "No shared samples between
phenotype and covariate file", which reads like a data problem but is an orientation one.

```bash
sos run pipeline/mnm_regression.ipynb qtl_dataset_construct+mnm \
    --name test_study --cwd <out> \
    --genoFile tests/fixtures/qtl_mini/protocol_example.genotype.chr22.bed \
    --phenoFile tests/fixtures/qtl_mini/protocol_example.pheno_manifest_context.tsv \
    --covFile tests/fixtures/qtl_mini/example_covariates.tsv \
    --transpose-covariates True \
    --customized-association-windows tests/fixtures/qtl_mini/association_windows.bed \
    --region-name ENSG00000283047 --seed 1 \
    --modular_script_dir code/script -j1
```

Then copy `<out>/multivariate_fine_mapping/test_study.ENSG00000283047.multicontext_bvsr.rds`
over `multicontext_bvsr.rds`.

The `mnm` step takes no `--prior`: `fine_mapping.R` uses the canonical mixture prior unless
`--prior-twas-weights` supplies a TwasWeights RDS from a preceding `mrmash` run, from which
it builds the data-driven prior. `--prior` belongs to the `fsusie`/`mvfsusie` cells, and SoS
pools every cell's parameters, so passing it here is accepted and then silently ignored.

Two seeded runs on one machine are byte-identical. Cross-platform behaviour is untested;
if the x86/ARM CI matrix disagrees, loosen the manifest tolerance rather than dropping the
comparison.
