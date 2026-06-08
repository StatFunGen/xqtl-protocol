# xQTL protocol core modules

## Basic QTL analysis

1. `molecular_phenotypes`: quantification and QC of molecular phenotypes
2. `data_preprocessing`: prepare (including QC) $X$, $Y$, $Z$ for association analysis
3. `association_scan`: xQTL calling

## Advanced QTL analysis and integration

1. `mnm_analysis`: Multivariate and multiple regression analysis, including SuSiE, fSuSiE, mvSuSiE, ColocBoost (for mvSuSiE and ColocBoost input can be QTL individual level data, xQTL summary statistics data and GWAS summary statistics), univariate TWAS weights, mr.mash, mmQTL (individual level data), SuSiE_RSS (xQTL summary statistics data and GWAS summary statistics)
2. `multivariate_genome`: MASH, METAL (genome-wide multivariate analysis)
3. `enrichment`: GREGOR, sLDSC, and a customized implementation using VEP
4. `pecotmr_integration`: pair-wise enrichment, colocalization, TWAS and MR using the pecotmr framework; cTWAS and other analysis that uses wrappers from `pecotmr` library
5. `rare_xqtl`: watershed pipeline
6. `quantile_qtl`: an experimental workflow of applying quantile regression to xQTL studies
7. `prototype_obsolete`: either obsolete pipelines, or unpolished and unused drafts including polyfun, MRAID, CAFEH, COLOC, fastenloc that we once tried but decided to not adopt in this project because of more suitable alternatives.

## Utilities

1. `command_generator`: an attempt to make a "one-click analysis" possible
2. `post_processing`: formatting of analysis results
3. `misc`: various scripts for maintenance and management


\textbf{Currently, what's completely missing are:}

- Rare xQTL related
