---
title: Methods and references
---

The protocol runs published methods. When you report results, cite the papers for the methods you used, listed here by analysis stage, and the protocol repository (https://github.com/StatFunGen/xqtl-protocol). Preprints are marked; check for a journal version before citing.

## Workflow system

| Method | What it does in the protocol | Reference |
|---|---|---|
| SoS (Script of Scripts) | Runs every pipeline, locally or on a cluster | Wang and Peng 2019, *PLoS Comput Biol* ([doi](https://doi.org/10.1371/journal.pcbi.1006843)) |

## Molecular phenotypes

| Method | Module | Reference |
|---|---|---|
| STAR | [RNA-seq expression calling](../code/SoS/molecular_phenotypes/calling/RNA_calling.ipynb) | Dobin et al. 2013, *Bioinformatics* ([doi](https://doi.org/10.1093/bioinformatics/bts635)) |
| RSEM | RNA-seq expression calling | Li and Dewey 2011, *BMC Bioinformatics* ([doi](https://doi.org/10.1186/1471-2105-12-323)) |
| RNA-SeQC | RNA-seq expression calling | DeLuca et al. 2012, *Bioinformatics* ([doi](https://doi.org/10.1093/bioinformatics/bts196)) |
| LeafCutter | [Splicing calling](../code/SoS/molecular_phenotypes/calling/splicing_calling.ipynb) | Li et al. 2018, *Nat Genet* ([doi](https://doi.org/10.1038/s41588-017-0004-9)) |
| LeafCutter2 | [LeafCutter2](../code/SoS/molecular_phenotypes/calling/splicing_calling_leafcutter2.ipynb) | Buen Abad Najar et al. 2025, *bioRxiv* preprint ([doi](https://doi.org/10.1101/2025.04.06.646893)) |
| Torino | [Torino isoform usage](../code/SoS/molecular_phenotypes/calling/isoform_calling_torino.ipynb) | Buen Abad Najar et al. 2026, *bioRxiv* preprint ([doi](https://doi.org/10.64898/2026.10.21.760529)) |

## Covariates and association testing

| Method | Module | Reference |
|---|---|---|
| PEER | [Hidden factor analysis](../code/SoS/data_preprocessing/covariate/covariate_hidden_factor.ipynb) | Stegle et al. 2012, *Nat Protoc* ([doi](https://doi.org/10.1038/nprot.2011.457)) |
| TensorQTL | [TensorQTL](../code/SoS/association_scan/TensorQTL/TensorQTL.ipynb) | Taylor-Weiner et al. 2019, *Genome Biol* ([doi](https://doi.org/10.1186/s13059-019-1836-7)) |
| Hierarchical multiple testing | [QTL post-processing](../code/SoS/association_scan/qtl_association_postprocessing.ipynb) | Huang et al. 2018, *Nucleic Acids Res* ([doi](https://doi.org/10.1093/nar/gky780)) |
| MASH | [MASH](../code/SoS/multivariate_genome/MASH/mash_fit.ipynb) | Urbut et al. 2019, *Nat Genet* ([doi](https://doi.org/10.1038/s41588-018-0268-8)) |

## Fine-mapping

| Method | Module | Reference |
|---|---|---|
| SuSiE | [Fine-mapping and TWAS](../code/SoS/mnm_analysis/mnm_methods/mnm_regression.ipynb) | Wang et al. 2020, *J R Stat Soc B* ([doi](https://doi.org/10.1111/rssb.12388)) |
| SuSiE-RSS | [RSS fine-mapping](../code/SoS/mnm_analysis/mnm_methods/rss_analysis.ipynb) | Zou et al. 2022, *PLoS Genet* ([doi](https://doi.org/10.1371/journal.pgen.1010299)) |
| mvSuSiE | Fine-mapping and TWAS | Zou et al. 2026, *Nat Genet* ([doi](https://doi.org/10.1038/s41588-025-02486-7)) |
| fSuSiE | Fine-mapping and TWAS | Denault et al. 2025, *bioRxiv* preprint ([doi](https://doi.org/10.1101/2025.08.17.670732)) |

## GWAS integration

| Method | Module | Reference |
|---|---|---|
| ColocBoost | [ColocBoost](../code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb) | Cao et al. 2025, *medRxiv* preprint ([doi](https://doi.org/10.1101/2025.04.17.25326042)) |
| enloc | [SuSiE-enloc](../code/SoS/pecotmr_integration/SuSiE_enloc.ipynb) | Wen et al. 2017, *PLoS Genet* ([doi](https://doi.org/10.1371/journal.pgen.1006646)) |
| cTWAS | [TWAS, cTWAS and MR](../code/SoS/pecotmr_integration/twas_ctwas.ipynb) | Zhao et al. 2024, *Nat Genet* ([doi](https://doi.org/10.1038/s41588-023-01648-9)) |
| INTACT | [INTACT](../code/SoS/pecotmr_integration/intact.ipynb) | Okamoto et al. 2023, *Am J Hum Genet* 110(1):44–57 |

## Enrichment and validation

| Method | Module | Reference |
|---|---|---|
| S-LDSC | [S-LDSC enrichment](../code/SoS/enrichment/sldsc_enrichment.ipynb) | Finucane et al. 2015, *Nat Genet* ([doi](https://doi.org/10.1038/ng.3404)) |
| LD score regression | S-LDSC enrichment | Bulik-Sullivan et al. 2015, *Nat Genet* ([doi](https://doi.org/10.1038/ng.3211)) |
| GREGOR | [GREGOR](../code/SoS/enrichment/gregor.ipynb) | Schmidt et al. 2015, *Bioinformatics* ([PubMed](https://pubmed.ncbi.nlm.nih.gov/25886982/)) |
| Generalized TAD boundaries | [Generalized TADB](../code/SoS/reference_data/generalized_TADB.ipynb) | McArthur et al. 2021, *Am J Hum Genet* ([doi](https://doi.org/10.1016/j.ajhg.2021.01.001)) |
