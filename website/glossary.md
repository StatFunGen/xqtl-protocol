---
title: Glossary
---

Terms used across the protocol, in plain language. Most entries link to a module that produces or uses the quantity.

## Genetics and study design

xQTL
: A genetic variant associated with a molecular trait. The *x* stands for the trait type: *eQTL* for gene expression, *sQTL* for splicing, *pQTL* for protein level, *mQTL* for DNA methylation, *haQTL* for histone acetylation, *apaQTL* for alternative polyadenylation.

Molecular phenotype
: A measured molecular trait per sample, such as a gene's expression level or an intron's splicing ratio. Modules store them as a matrix with one row per feature and one column per sample ([Phenotype preprocessing](../code/SoS/data_preprocessing/phenotype_preprocessing.ipynb)).

*cis* and *trans*
: A *cis*-QTL lies near the gene or feature it affects, by default within 1 Mb; a *trans*-QTL lies far away or on another chromosome ([TensorQTL](../code/SoS/association_scan/TensorQTL/TensorQTL.ipynb)).

Context
: The tissue, cell type or condition in which a molecular trait is measured. *Multi-context* analyses model several contexts jointly ([MASH](../code/SoS/multivariate_genome/MASH/mash_fit.ipynb)).

Covariates
: Variables other than genotype that affect the trait, such as sex, age, batch and genotype principal components. Association tests adjust for them ([Covariate preprocessing](../code/SoS/data_preprocessing/covariate_preprocessing.ipynb)).

Hidden factors
: Unmeasured sources of variation, such as batch effects or cell-type composition, estimated from the phenotype matrix and added to the covariates ([Hidden factor analysis](../code/SoS/data_preprocessing/covariate/covariate_hidden_factor.ipynb)).

Linkage disequilibrium (LD)
: Correlation between nearby variants, because they are inherited together. LD is why many variants near a causal one also show association.

LD reference
: A table of LD between variants, computed from a reference population and split into blocks. Methods that start from summary statistics need one ([Where to get reference data](reference_data_sources.md)).

Summary statistics
: Per-variant association results (effect size, standard error, *p*-value) without individual genotypes. GWAS results are usually shared this way.

## Fine-mapping

Fine-mapping
: Identifying which variants in an associated region are likely causal, given the LD among them ([Fine-mapping and TWAS](../code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb)).

Posterior inclusion probability (PIP)
: The probability that a variant is causal for the trait, from 0 to 1. Fine-mapping reports one PIP per variant.

95% credible set (CS)
: A small set of variants that contains a causal variant with 95% probability. A region can have several credible sets, one per independent signal.

Joint fine-mapping
: Fine-mapping several traits or contexts in one model, so a variant shared across them is identified once (mvSuSiE in [Fine-mapping and TWAS](../code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb)).

## GWAS integration

Colocalization
: Testing whether a molecular trait and a disease share the same causal variant in a region, rather than having separate causal variants in LD ([ColocBoost](../code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb), [SuSiE-enloc](../code/SoS/pecotmr_integration/SuSiE_enloc.ipynb)).

95% colocalization confidence set (CoS)
: The ColocBoost counterpart of a credible set: variants that contain a variant shared by the colocalized traits with 95% probability.

Variant colocalization probability (VCP)
: The probability that a variant is the shared causal variant of a colocalization signal, reported per variant by ColocBoost.

TWAS
: Transcriptome-wide association study. Predicts each gene's expression from genotype, using weights learned from xQTL data, and tests the prediction for association with a disease ([TWAS, cTWAS and MR](../code/SoS/pecotmr_integration/twas_ctwas.ipynb)).

cTWAS
: Causal TWAS. Fine-mapping genes and variants together, so a gene is credited only for the signal that nearby variants and genes do not explain.

Mendelian randomization (MR)
: Using genetic variants as instruments to estimate the causal effect of a molecular trait on a disease.

## Enrichment

Enrichment
: Whether a set of variants, such as fine-mapped xQTLs, overlaps disease signals or functional annotations more than expected by chance ([S-LDSC enrichment](../code/SoS/enrichment/sldsc_enrichment.ipynb)).

Heritability enrichment
: The share of a disease's heritability explained by variants in an annotation, divided by the share of variants in it. S-LDSC estimates it from GWAS summary statistics.
