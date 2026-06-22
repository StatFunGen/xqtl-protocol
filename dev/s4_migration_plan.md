# S4 migration plan: pecotmr ↔ xqtl-protocol

Working document. Captures the design conversation from 2026-06-13/15 covering the
S4 migration of pecotmr, the conversion of xqtl-protocol to thin wrappers, and
the architectural constraints we agreed on.

## Constraints

1. **xqtl-protocol R code must never use `@` or `slot()`** on a pecotmr object. Only
   accessor functions (`getX()`) are allowed.
2. **xqtl-protocol R code should rarely use any accessor either.** A per-region
   script should be: parse args → loader → pipeline → save. If a script needs to
   call more than a couple of accessors, pecotmr should encapsulate that work.
3. **Every loader returns an S4** of an appropriate class.
4. **Every per-region pipeline accepts that S4 directly** via S4 method dispatch.
   The per-condition loop lives **inside pecotmr**, never in xqtl-protocol.
5. **pecotmr's own code follows the Bioconductor rule**: `@slot` is allowed only
   inside accessor method bodies (`setMethod("getX", "Class", ...)`), in
   `new(...)` / constructor bodies, in `slot<-` assignment, and in validity
   functions. Everywhere else, slot access must go through accessors.
6. Genome-aggregating analyses (mash, S-LDSC) stay two-stage. That is a method-
   level fact, not a smell.

## Where we are today

### Snake → camel rename (done)

68 mechanical function-name renames applied across `code/SoS/` notebooks and
`code/script/` R files. Scoped to `name(` and `pecotmr::name` patterns inside R
contexts only; skipped Python contexts in SoS notebooks; skipped local-collision
cases (`filter_relatedness` in GWAS_QC.R, `meta_analysis_per_cell` /
`update_mash_model_cov` in mash_posterior.ipynb).

### Mechanical S4 access conversion (done)

63 substitutions across 7 files (`fsusie.R`, `susie_twas.R`, `mnm.R`,
`mnm_regression.ipynb`, `qr_and_twas.ipynb`, `twas_ctwas.ipynb`,
`rss_ld_sketch.ipynb`):

- `fdat$residual_X[[r]]` → `getResidualX(fdat, r)`
- `fdat$residual_Y[[r]]` → `getResidualY(fdat, r)`
- `fdat$residual_X_scalar[[r]]` → `getResidualXScalar(fdat, r)`
- `fdat$residual_Y_scalar[[r]]` → `getResidualYScalar(fdat, r)`
- `fdat$X_variance[[r]]` → `getXVariance(fdat, r)`
- `fdat$X` → `getGenotypeMatrix(fdat)` (both RegionalData and Multivariate)
- `fdat$Y[[r]]` on RegionalData → `getPhenotypes(fdat)[[r]]`
- `fdat$Y` on MultivariateRegionalData → `getY(fdat)`
- `fdat$maf[[r]]` → `getMaf(fdat)[[r]]` (will work once getMaf for RegionalData lands)
- `fdat$covar[[r]]` → `getCovariates(fdat)[[r]]`
- `length/names/seq_along(fdat$residual_Y)` → `length/names/seq_along(getPhenotypes(fdat))`
  (count- and name-equivalent by construction; not "iterate residuals" semantically,
  but functionally correct)
- LdData: `$LD_variants` → `getVariantIds`, `$ref_panel` → `getRefPanel`,
  `$LD_matrix` → `getCorrelation` or `getGenotypes` (depending on `returnGenotype`
  flag at the call site)

**Reassignments skipped** (would have produced invalid R syntax):
- `fdat$residual_X[[r]] <- NA` and `fdat$residual_Y[[r]] <- NA` (intent was
  memory-clearing; moot under lazy residual computation — should be deleted in
  cleanup pass).
- `LD_list$LD_matrix <- LD_list$LD_matrix[-dup_idx, -dup_idx]` (the whole
  dedup block should be deleted — pecotmr deduplicates internally now).

**Zero `@` introduced** in any of the 63 substitutions.

### Pending S4 access points

30 remaining; after subtracting false positives (`opt$maf` is optparse, not
pecotmr; `qr_results$maf` is a local list), the real remaining work is ~22 sites
in four categories:

1. **Slot renames** (~12 sites): `fdat$dropped_samples` → `@droppedSamples`;
   `fdat$Y_coordinates` / `phenotype_coordiates` → `@coordinates`. These cannot
   be mechanically rewritten without accessors — see Phase 0 below.
2. **Reassignments** (~5 sites): delete the lines.
3. **Wrong access on Multivariate** (~2 sites): `$residual_Y` on
   `MultivariateRegionalData` → `getY()` (no residualization on that class).
4. **Judgment calls** (~3 sites): `$X_data`, `$X_variance` without index,
   `$residual_Y` matrix indexing in `mnm_regression.ipynb`. Need surrounding code
   review.

### Architectural realization

`fdat` is purely a variable name in xqtl-protocol — it doesn't appear in
pecotmr. With S4, the type is exposed by the loader name and the class name; the
variable name is doing zero work.

More importantly, **xqtl-protocol is reaching into low-level pecotmr internals**.
The current shape of `fsusie.R` / `susie_twas.R` / `mnm.R` is a per-condition
loop that unpacks `fdat` into pieces and feeds them to pipeline functions that
were designed for the old list-based API. Two smoking guns:

1. `susie_post_processor` (called from fsusie.R) was **removed from pecotmr in
   March 2024** (commit d85aca2). It doesn't exist anywhere — the script would
   fail at runtime. The replacement (`postprocessFinemappingFits`) is meant to be
   called *inside* `univariateAnalysisPipeline`, not by external callers.
2. `univariateAnalysisPipeline`, `multivariateAnalysisPipeline`,
   `twasWeightsPipeline`, `fsusieWrapper` all still take loose pieces (X, Y,
   maf, ...). With S4, this signature is a smell: the caller is forced to ask
   the object for its slots and feed them back.

## Target architecture

### pecotmr public API: S4 in, structured out

Each per-region pipeline accepts the loader's S4 directly. No loose-piece public
signatures. Loose-piece logic survives as private helpers.

| Pipeline | Public signature | Returns |
|---|---|---|
| `univariateAnalysisPipeline` | `(regional = RegionalData, ...)` | named list per condition |
| `multivariateAnalysisPipeline` | `(regional = MultivariateRegionalData, ...)` | single result |
| `twasWeightsPipeline` | `(regional = RegionalData, ...)` | named list of per-condition `TwasWeights` |
| `fsusieWrapper` | `(regional = RegionalData, ...)` | named list of per-condition fSuSiE results |
| `functionalAnalysisPipeline` *(new)* | `(regional = RegionalData, ...)` | named list of per-condition composites (top-PC + SuSiE-on-PC + TWAS + fSuSiE) |
| `quantileAnalysisPipeline` *(new)* | `(regional = RegionalData, ...)` | named list of per-condition results |
| `colocboostPipeline` | `(regional = MultitaskRegionalData, focalTrait, ...)` | single result |
| `rssAnalysisPipeline` | unchanged — already takes `LdData` | unchanged |
| `susieRssPipeline` | `(qcResult = QcResult, ldData = LdData, ...)` (tighten from loose) | single result |
| `twasPipeline` (cTWAS) | unchanged — per-LD-block by design | unchanged |
| `mashPipeline`, `sldscPostprocessingPipeline` | unchanged — genome-scope by design | unchanged |

### Per-gene SoS parallelization compatibility

Confirmed: pecotmr's per-region pipelines are scoped exactly right for "one SoS
job per gene". Three caveats:

- **cTWAS** is per-LD-block (`twasPipeline(twasWeightsData, ldMetaFilePath,
  gwasMetaFile, regionBlock, ...)`) — joint inference across genes sharing an LD
  region. `twas_ctwas.ipynb` already drives this correctly.
- **mash / S-LDSC** are genome-aggregating; the SoS pipeline is two-stage
  (per-gene fan-out → genome gather). `mash_preprocessing` → `mash_fit` →
  `mash_posterior` already has this shape.
- **`loadMultitaskRegionalData` currently returns a plain `list`**, not S4.
  Needs S4-ification along with a `MultitaskRegionalData` class.

### xqtl-protocol target shape

Each per-region R script collapses to ~10 lines:

```r
opt <- parse_args(...)
regional <- loadRegional*Data(
  genotype = opt$genotype, phenotype = phenotype_files,
  covariate = covariate_files, region = opt$region,
  conditions = conditions, ...
)
result <- xxxAnalysisPipeline(regional, ...)
result$region_info <- list(
  region_coord = parseRegion(opt$region),
  grange       = parseRegion(opt$window),
  region_name  = opt[["region-name"]]
)
saveRDS(result, opt$output)
```

No `fdat$...`. No `@`. No accessors in the success path. No per-condition loop.

## Plan

### Phase 0 — pecotmr accessor foundation

**Must come first** because every later phase needs internal pecotmr code to
read slots via accessors (Bioconductor rule).

#### Phase 0A — Accessor completion (~25-30 new accessors)

Add missing `setGeneric` + `setMethod("get*", ...)` for every slot read anywhere
in pecotmr code outside of accessor bodies / constructors / validity.

**`RegionalData`** (4 new):
- `getScaleResiduals(x)` → `x@scaleResiduals`
- `getMaf(x)` for `RegionalData` (mirror of the `MultivariateRegionalData` method)
- `getRegion(x)` → `x@region` (the `GRanges` directly; complementary to existing
  derived `getChrom`/`getGrange`)
- `getDroppedSamples(x)` → `x@droppedSamples`
- `getCoordinates(x)` → `x@coordinates`

**`MultivariateRegionalData`** (3 new, same generics):
- `getDroppedSamples`, `getRegion`, `getCoordinates`

**`LdData`** (2 new):
- `getSnpIdx(x)` → `x@snpIdx`
- `getNRef(x)` → `x@nRef`

**`GenotypeHandle`** (5 new):
- `getPath`, `getFormat`, `getSnpInfo`, `getNSamples`, `getSampleIds`
- (`getPgenPtr` only if any non-method code reads it)

**`GwasSumStats`** (3 new):
- `getSumstats`, `getGenome`, `getTraitName`

**`FineMappingResult`** (2 new):
- `getMethod`, `getSumstats`

**`H2Estimate`, `LdBlocks`, `LdEigen`, `LdScore`, `AnnotationMatrix`** (~10
more across the heritability / LD-statistic infrastructure, exact set after
internal audit)

All exported. Each gets a roxygen page and a test against a fixture object.

#### Phase 0B — Internal `@` → accessor refactor

Scripted pass across all of `R/`. For every `<expr>@<slot>` read outside of:

- `setMethod("get*", "Class", ...)` bodies
- `new("Class", ...)` and constructor function bodies
- `validity = function(object) { ... }` bodies (Bioconductor permits `@` here
  because the object is being constructed)
- `slot(x, "name") <- value` assignment

…replace with the corresponding accessor call. No behavior change. All tests
must pass. Baseline for `R CMD BiocCheck` compliance.

Existing scope: ~357 `@` access sites across 43 files. Bulk in `file_utils.R`,
`twas.R`, `univariate_pipeline.R`, `multivariate_pipeline.R`, `LD.R`,
`ld_loader.R`, `mash_wrapper.R`, `sumstats_qc.R`.

### Phase 1 — Pipelines accept only S4

**pecotmr PR A**: Refactor `univariateAnalysisPipeline`. Rename current body to
private `.univariateAnalysisOneCondition` (still uses accessors internally — no
`@`). New `univariateAnalysisPipeline(regional, ...)` requires `RegionalData`,
iterates conditions via `seq_along(getPhenotypes(regional))`, calls the private
helper. Update tests + roxygen examples to construct `RegionalData`.

**pecotmr PR B**: Same refactor for `multivariateAnalysisPipeline`,
`twasWeightsPipeline`, `fsusieWrapper`.

**pecotmr PR C**: New `MultitaskRegionalData` S4 class. Update
`loadMultitaskRegionalData` to return it. Update `regionDataToIndInput`,
`regionDataToRssInput`, `regionDataToColocboostInput` to accept it.
`colocboostPipeline(regional = MultitaskRegionalData, ...)`.

**pecotmr PR D**: New `functionalAnalysisPipeline(regional, ...)` — absorbs
the per-condition composite from `fsusie.R` (top-PC + SuSiE-on-PC + TWAS +
fSuSiE).

**pecotmr PR E**: New `quantileAnalysisPipeline(regional, ...)` — absorbs the
quantile-regression logic from `qr_and_twas.ipynb`.

**pecotmr PR F**: Tighten `susieRssPipeline` to take `QcResult` + `LdData`.

**pecotmr PR G**: Documentation pass. `NEWS.md`: "Per-region pipelines accept
S4 inputs only. Loose-piece signatures are no longer public. Construct a
`RegionalData` via the loader or the constructor." pkgdown index updated.

#### Open call: hard vs. soft deprecation

If the only real consumer of pecotmr is xqtl-protocol + this team's local
scripts, **hard break** is fine and saves a release cycle. If there are
external workflows, **soft path** with `.Deprecated()` for a release is safer.
Decide before Phase 1.

### Phase 2 — xqtl-protocol thin wrappers

Each per-region script becomes ~10 lines (template above). Per file:

- `code/script/mnm_analysis/mnm_methods/susie_twas.R` (~440 lines → ~15)
- `code/script/mnm_analysis/mnm_methods/fsusie.R` (~250 lines → ~15)
- `code/script/mnm_analysis/mnm_methods/mnm.R` (~180 lines → ~15)
- `code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb` (R-in-SoS blocks)
- `code/SoS/association_scan/quantile_models/qr_and_twas.ipynb`
- `code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb` (R-block only — the
  Python side stays for now)

One file at a time. Toy-dataset end-to-end verification between each.

### Phase 3 — Cleanup

- Delete `filter_fdat_except_specific_names` from fsusie.R (replaced by
  `minMarkers` arg in `loadRegionalFunctionalData`).
- Delete the cTWAS dedup block in `twas_ctwas.ipynb` (pecotmr's `loadLdMatrix`
  already deduplicates internally).
- Delete `fdat$residual_X[[r]] <- NA` and `fdat$residual_Y[[r]] <- NA`
  reassignments (memory-clearing is moot under lazy compute).
- Audit thin-wrapper outputs vs. pre-refactor outputs to confirm no behavioral
  drift.

## Sequencing summary

| Order | Work | Why |
|---|---|---|
| 1 | Phase 0A — add missing accessors | Pure addition, no risk. Unblocks 0B and Phase 1. |
| 2 | Phase 0B — internal `@` → accessor refactor | Establishes Bioconductor compliance. No behavior change. |
| 3 | Phase 1 PRs A–G — pipelines accept S4 only | Sequentially; each PR is one pipeline family. |
| 4 | Phase 2 — xqtl-protocol thin wrappers | One file at a time, verified against toy. |
| 5 | Phase 3 — cleanup | After thin wrappers are confirmed working. |

## Verification strategy

For each Phase 1 pipeline PR:

```r
test_that("univariateAnalysisPipeline.RegionalData matches one-condition path", {
  regional <- makeToyRegionalData()
  out_via_helper <- .univariateAnalysisOneCondition(
    X = getResidualX(regional, 1),
    Y = getResidualY(regional, 1),
    maf = getMaf(regional)[[1]],
    ...
  )
  out_via_public <- univariateAnalysisPipeline(regional, ...)
  expect_equal(out_via_public[[1]], out_via_helper)
})
```

For each xqtl-protocol thin wrapper:

1. Run pre-refactor script on toy dataset; capture output RDS as fixture.
2. Refactor to thin wrapper.
3. Run new wrapper on same toy dataset.
4. Diff RDS structures and key numerics (PIPs, z-scores, weights).

A wrapper is "done" only when its output matches the fixture, not when it
compiles.

## Per-condition iteration semantic note

For non-indexed `length()` / `names()` / `seq_along()` wrappers around
`fdat$residual_Y`, the mechanical conversion used `getPhenotypes(fdat)` rather
than materializing a residuals list. This is functionally equivalent because
`getResidualY(fdat, r)` returns a residual for each phenotype condition with the
same name. The conversion is correct for count- and name-based iteration; a
purist would precompute the residual list once outside the loop and index it
inside, which is both clearer and faster but a structural change beyond the
mechanical scope.

## Files modified by the mechanical passes (committed state TBD)

- `code/SoS/association_scan/quantile_models/qr_and_twas.ipynb`
- `code/SoS/enrichment/sldsc_enrichment.ipynb`
- `code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb`
- `code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb`
- `code/SoS/mnm_analysis/mnm_postprocessing.ipynb`
- `code/SoS/multivariate_genome/MASH/mash_preprocessing.ipynb`
- `code/SoS/pecotmr_integration/SuSiE_enloc.ipynb`
- `code/SoS/pecotmr_integration/twas_ctwas.ipynb`
- `code/SoS/reference_data/rss_ld_sketch.ipynb`
- `code/script/mnm_analysis/mnm_methods/fsusie.R`
- `code/script/mnm_analysis/mnm_methods/mnm.R`
- `code/script/mnm_analysis/mnm_methods/susie_twas.R`

`pipeline/*.ipynb` are symlinks into `code/SoS/`, so they pick up the changes
transparently.
