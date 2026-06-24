# Deferred work — pecotmr_integration migration

## ctwas `merge_regions` (boundary-gene region merging) — NOT YET PORTED

The legacy `twas_ctwas.ipynb [ctwas_3]` cell had an optional, **default-off**
(`merge_regions=False`) step: for high-PIP genes straddling an LD-block
boundary (`susie_pip > 0.5 & !is.na(cs)`), it ran
`ctwas::merge_region_data(...)` then re-`ctwas::finemap_regions(...)` on the
merged regions, writing `*_merged` outputs.

**Status in the new pipeline:** `pecotmr::estCtwasParam` already computes
`boundary_genes` (`ctwas::get_boundary_genes`, pecotmr `R/ctwasPipeline.R:363`)
and threads it through `finemapCtwasRegions`'s output (`R/ctwasPipeline.R:506`).
The **merge + re-finemap second pass is NOT implemented**. (Note: the
`ctwas::expand_region_data` call at `ctwasPipeline.R:439` is the
`thin < 1 -> thin = 1` pre-screen expansion, *not* boundary merging — don't
conflate them.)

**When revisited:** add an optional merge+re-finemap stage inside
`pecotmr::finemapCtwasRegions` (it already holds the needed internal state:
`boundary_genes`, `region_data`, `LD_map`, `snp_map`, `z_snp`, `z_gene`),
reusing the computed `boundary_genes` + `ctwas::merge_region_data`; then
`ctwas_finemap.R` just exposes a `--merge-regions` flag. The logic belongs in
pecotmr — keep the wrapper thin.

**Context:** the other legacy `ctwas_3` extras — `diagnose_LD_mismatch_susie` /
`get_problematic_genes` / `finemap_regions_noLD` — are **redundant**: the
LD-mismatch concern is handled upstream by the mandatory `summaryStatsQc()`
DENTIST/SLALoM gate, so they are not ported.
