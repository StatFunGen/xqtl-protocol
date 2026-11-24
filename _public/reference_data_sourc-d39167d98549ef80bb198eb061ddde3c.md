---
title: Where to get reference data
---

The example data in `tests/fixtures/` is enough to run every Minimal Working Example. A full analysis also needs genome-wide reference files, which are too large to keep in the repository. This page lists each one, where it comes from, and which modules use it. All files are GRCh38.

## Download, then format with the protocol

[Reference Data Standardization](../code/SoS/reference_data/reference_data_preparation.ipynb) downloads these files and formats them for the protocol (chromosome names, removed alternate contigs, added ERCC spike-ins). Run its steps rather than using the raw downloads directly.

| File | Source | Used by |
|---|---|---|
| Genome FASTA, `GRCh38_full_analysis_set_plus_decoy_hla.fa` | [1000 Genomes FTP](ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/) | [RNA-seq expression calling](../code/SoS/molecular_phenotypes/calling/RNA_calling.ipynb), [Genotype VCF QC](../code/SoS/data_preprocessing/genotype/VCF_QC.ipynb) |
| Gene annotation, Ensembl release 103 | [Ensembl FTP](http://ftp.ensembl.org/pub/release-103/gtf/homo_sapiens/Homo_sapiens.GRCh38.103.chr.gtf.gz) | [Gene coordinate annotation](../code/SoS/data_preprocessing/phenotype/gene_annotation.ipynb), RNA-seq calling, splicing and APA calling |
| ERCC spike-in sequences, `ERCC92` | [Thermo Fisher](https://assets.thermofisher.com/TFS-Assets/LSG/manuals/ERCC92.zip) | Added to the genome and annotation for RNA-seq |
| dbSNP build 150, `00-All.vcf.gz` | [NCBI FTP](https://ftp.ncbi.nlm.nih.gov/snp/organisms/human_9606_b150_GRCh38p7/VCF/) | [Genotype VCF QC](../code/SoS/data_preprocessing/genotype/VCF_QC.ipynb), only to add rsIDs |

## Built by the protocol

These files are not downloaded; [Reference Data Standardization](../code/SoS/reference_data/reference_data_preparation.ipynb) builds them from the files above. Build them once and reuse them across studies.

| File | Step | Used by |
|---|---|---|
| STAR genome index | 8 | RNA-seq alignment in [RNA-seq expression calling](../code/SoS/molecular_phenotypes/calling/RNA_calling.ipynb) |
| RSEM index | 9 | Transcript quantification in RNA-seq expression calling |
| RefFlat annotation | 10 | Picard RNA-seq QC in RNA-seq expression calling |
| Collapsed gene model, `*.collapse_only.gene.gtf` | 6 | Gene coordinate annotation, RNA-seq calling, expression normalization |

## Standardized copies on the AD Knowledge Portal

The ADSP Functional Genomics Consortium keeps the standardized reference files, with their specifications, on the [AD Knowledge Portal (syn36416587)](https://www.synapse.org/#!Synapse:syn36416587). Use these to match the consortium's analyses exactly. The AD Knowledge Portal needs a free account.

## LD reference

Fine-mapping from summary statistics ([RSS fine-mapping](../code/SoS/mnm_analysis/mnm_methods/rss_analysis.ipynb)), [TWAS, cTWAS and MR](../code/SoS/pecotmr_integration/twas_ctwas.ipynb), [ColocBoost](../code/SoS/mnm_analysis/mnm_methods/colocboost.ipynb) with summary statistics and [SuSiE-enloc](../code/SoS/pecotmr_integration/SuSiE_enloc.ipynb) need a linkage disequilibrium (LD) reference: correlations between variants, split into LD blocks. Modules read it through an LD meta file (`--ld-meta-data`), a table of blocks (`#chr`, `start`, `end`, `path`).

- **ADSP LD reference panel.** LD matrices from whole-genome sequencing of 16,571 non-Hispanic white individuals (Genome Center for Alzheimer's Disease), computed within 1,361 LD blocks defined from 1000 Genomes EUR samples. Each block is a `.cor.xz` matrix with a `.cor.xz.bim` variant list. Controlled access: request ADSP access on the AD Knowledge Portal, then `synapse get -r syn53171227`. Details on the [xQTL Resources LD reference page](https://adsp-fgc.niagads.org/xqtl-resources/xqtl-data/reference_data/ld_reference/).
- **Your own panel.** Build LD sketches from genotypes with [RSS LD sketch](../code/SoS/reference_data/rss_ld_sketch.ipynb). Use this when your GWAS ancestry differs from the ADSP panel.

The example LD meta file is `tests/fixtures/ld_reference/ld_meta_file.tsv`.

## Optional inputs

| File | Source | Used by |
|---|---|---|
| TAD calls for cortex and hippocampus (Schmitt et al. 2016) | See [Generalized TADB](../code/SoS/reference_data/generalized_TADB.ipynb) | TAD-based association windows instead of a fixed 1 Mb cis window |
| Baseline annotations and LD scores for S-LDSC | [ldsc](https://github.com/bulik/ldsc/wiki/Partitioned-Heritability-from-Continuous-Annotations) and [PolyFun](https://github.com/omerwe/polyfun) | [S-LDSC enrichment](../code/SoS/enrichment/sldsc_enrichment.ipynb) |
| GWAS summary statistics | Your study or a public release, GRCh38 | GWAS integration modules; see [Using your own data](own_data.md) for the required columns |
