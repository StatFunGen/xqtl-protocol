# xqtl-protocol wrapper tests

`pytest` is the single orchestrator for the pipeline's thin CLI wrapper scripts.
Each test drives a wrapper exactly as the pipeline does — `Rscript <script>
--args` → output file — and asserts on the result. R-side structure is read back
through `helpers/rds_probe.R` (an `readRDS` → JSON summary), so no R test package
is needed.

## Running

From the repo root (uses the pixi env's `Rscript` + `pecotmr`):

```bash
pixi run pytest tests -m unit          # Tier A: fast CLI-contract (--help) tests
pixi run pytest tests -m integration   # Tier B: end-to-end wrapper runs on fixtures
pixi run pytest tests                  # everything
pixi run test                          # alias for `pytest tests`
```

Point at a specific R with `XQTL_RSCRIPT=/path/to/Rscript` (defaults to the one
on `PATH`).

## Layout

```
tests/
├── conftest.py                 # shared fixtures: run_r, read_rds, inventory hook
├── pytest.ini                  # markers (unit/integration/notebook)
├── helpers/
│   ├── r_runner.py             # run_r() subprocess runner + read_rds()
│   ├── rds_probe.R             # readRDS -> jsonlite structural summary
│   └── inventory.py            # session-end tested-vs-untested script report
├── fixtures/qtl_mini/          # committed chr22 MWE (genotypes, expr, manifests)
└── pecotmr_integration/        # mirrors code/script/pecotmr_integration/
    ├── conftest.py             # qtl_dataset fixture (built once per session)
    ├── test_cli_contract.py    # Tier A: --help for every non-legacy wrapper
    ├── test_fine_mapping.py    # Tier B: qtl_dataset -> fine_mapping / twas_weights
    └── test_manifests.py       # Tier B: region_manifest
```

The test tree mirrors `code/script/<domain>/`; split fast vs. slow by **marker**,
not directory. New domains (`mnm_analysis/`, `association_scan/`) get a sibling
`tests/<domain>/` dir.

## Markers

- `unit` — fast CLI-contract tests (arg parsing / `--help`); gate every PR.
- `integration` — end-to-end wrapper runs on the committed fixtures (slower).
- `notebook` — SoS notebook-step tests (slowest; opt-in).

## Notes

- The `qtl_mini` covariates are QTLtools-format (covariate rows, sample columns),
  so `qtl_dataset_construct` runs with `--transpose-covariates`.
- `twas_weights` needs a **predictive** method (e.g. `--methods enet`); the
  fine-mapping methods (`susie`, …) are rejected there by design.
- The cTWAS wrappers need a newer pecotmr than the pinned env exposes and are not
  yet covered end-to-end (their `--help` contract is).
