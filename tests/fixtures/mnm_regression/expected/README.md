# mnm_regression expected outputs

| file | produced by |
|---|---|
| `univariate_bvsr.rds` | `susie_twas` — per-context SuSiE fine-mapping (`method = "susie"`) |
| `univariate_twas_weights.rds` | `susie_twas` — TWAS weights from the same run |
| `protocol_example.ENSG00000283047.multicontext_bvsr.rds` | `mnm` — joint multi-context fit (`method = "mvsusie"`, carries a `jointContexts` column) |

## Regenerating the multicontext result

`mnm` needs the dataset built first, so run both steps together. Note two things the
module documentation currently gets wrong: the parameter is `--prior`, not
`--mixture-prior`, and `--transpose-covariates True` is required — without it the run
fails with "No shared samples between phenotype and covariate file", which looks like
a data problem but is an orientation one.

```bash
sos run pipeline/mnm_regression.ipynb qtl_dataset_construct+mnm \
    --name protocol_example --cwd <out> \
    --genoFile tests/fixtures/qtl_mini/protocol_example.genotype.chr22.bed \
    --phenoFile tests/fixtures/qtl_mini/protocol_example.pheno_manifest_context.tsv \
    --covFile tests/fixtures/qtl_mini/example_covariates.tsv \
    --transpose-covariates True \
    --customized-association-windows tests/fixtures/qtl_mini/association_windows.bed \
    --region-name ENSG00000283047 --save-data --no-skip-twas-weights \
    --phenotype-names mv_pheno \
    --prior tests/fixtures/mash/expected/mixture_prior.EE.prior.rds \
    --ld-reference-meta-file tests/fixtures/ld_reference/ld_meta_file.tsv \
    --modular_script_dir code/script -j1
```

The mixture prior is a choice: `tests/fixtures/mash/expected/` also holds
`prior.cov_ed.EE.rds`. This fixture was generated with `mixture_prior.EE.prior.rds`.
If that is not the intended prior for the toy data, regenerate and replace.

No test currently drives the `mnm` step — `test_mnm_regression` runs
`qtl_dataset_construct+susie_twas` only — so this file is documentation of the
expected shape rather than an asserted comparison.
