## Title



## Authors




## Abstract




## Introduction




### Development of the protocol




### Overview of the procedure




### Applications of the method




### Comparison with other methods




### Experimental Design




#### Reference data (Step 1)
##### A.  Reference data preparation

#### Molecular Phenotypes (Step 2)
##### A.  RNA-seq expression

##### B.  Alternative splicing from RNA-seq data

#### Data Preprocessing (Step 3)
##### A.  Genotype preprocessing

##### B.  Phenotype preprocessing

##### C.  Covariate data preprocessing

#### QTL Association Testing (Step 4)
##### A.  QTL association testing

#### Multivariate Mixture Models (Step 5)
#### Multiomics Regression Models (Step 6)
##### A.  Fine-mapping and TWAS analysis

#### GWAS Integration (Step 7)
#### Enrichment and Validation (Step 8)

## Materials



### Software



### Hardware



## Procedure




### 1. Reference data

#### Reference data preparation
Timing: TBD
##### 1. Download the human genome

**What it does:** Download the GRCh38 reference FASTA used by sequence-aware tools.


```python

sos run pipeline/reference_data_preparation.ipynb download_hg_reference --cwd output/reference_data

```



##### 2. Download the gene annotation

**What it does:** Download the Ensembl gene annotation used to define genes and transcripts.


```python

sos run pipeline/reference_data_preparation.ipynb download_gene_annotation --cwd output/reference_data

```



##### 3. Download the ERCC reference

**What it does:** Download ERCC spike-in sequences and annotations for RNA-seq reference construction.


```python

sos run pipeline/reference_data_preparation.ipynb download_ercc_reference --cwd output/reference_data

```



##### 4. Download dbSNP

**What it does:** Download the dbSNP variant resource used when rsID annotation is required.


```python

sos run pipeline/reference_data_preparation.ipynb download_dbsnp --cwd output/reference_data

```



##### 5. Format and index the genome

**What it does:** Remove unsupported alternate sequences, append ERCC sequences and create FASTA indices.


```python

sos run pipeline/reference_data_preparation.ipynb hg_reference --cwd output/reference_data --ercc-reference output/reference_data/ERCC92.fa --hg-reference output/reference_data/GRCh38_full_analysis_set_plus_decoy_hla.fa

```



##### 6. Format the gene annotation

**What it does:** Add chromosome prefixes and construct the protocol-compatible gene models.


```python

sos run pipeline/reference_data_preparation.ipynb hg_gtf --cwd output/reference_data --hg-gtf output/reference_data/Homo_sapiens.GRCh38.103.chr.gtf --hg-reference output/reference_data/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy.fasta --stranded

```



##### 7. Combine gene and ERCC annotations

**What it does:** Combine the processed human and ERCC annotations into the GTF used downstream.


```python

sos run pipeline/reference_data_preparation.ipynb gene_annotation --cwd output/reference_data --ercc-gtf tests/fixtures/reference_data_preparation/ERCC92.gtf --hg-gtf output/reference_data/Homo_sapiens.GRCh38.103.chr.gtf --hg-reference output/reference_data/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy.fasta --stranded

```



##### 8. Build the STAR index

**What it does:** Build the STAR genome index used for RNA-seq alignment.


```python

sos run pipeline/reference_data_preparation.ipynb STAR_index --cwd output/reference_data --hg-reference output/reference_data/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta --numThreads 10 --mem 40G

```



##### 9. Build the RSEM index

**What it does:** Build the RSEM reference used for transcript-level expression quantification.


```python

sos run pipeline/reference_data_preparation.ipynb RSEM_index --cwd output/reference_data --hg-reference output/reference_data/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta --hg-gtf output/reference_data/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.gtf

```



##### 10. Generate RefFlat annotation

**What it does:** Convert the processed GTF into the RefFlat annotation used by Picard RNA-seq QC.


```python

sos run pipeline/reference_data_preparation.ipynb RefFlat_generation --cwd output/reference_data --hg-gtf output/reference_data/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.gtf

```



##### 11. Annotate a study VCF with dbSNP identifiers (optional)

**What it does:** Add rsIDs to a study VCF when downstream tools require named variants.


```python

sos run pipeline/VCF_QC.ipynb dbsnp_annotate \
    --genoFile tests/fixtures/vcf_qc/protocol_example.genotype.chr22.vcf.gz \
    --cwd output/vcf_qc

```



##### 12. Construct generalized TAD boundaries (optional)

**What it does:** Combine tissue-specific TAD calls into customized cis-association windows.


```python

sos run pipeline/generalized_TADB.ipynb default \
    --tad-input tests/fixtures/generalized_TADB/protocol_example.brain_TADs.txt \
    --gene-coords tests/fixtures/generalized_TADB/protocol_example.gene_start_end.tsv \
    --cwd output/tadb

```



##### 13. Construct an LD sketch for RSS analysis (optional)

**What it does:** Generate, process and merge LD-sketch products for summary-statistics regression.


```python

sos run pipeline/rss_ld_sketch.ipynb generate_W --n-samples 60 --output-dir output/rss_ld_sketch --B 50 --seed 123 --cwd output/rss_ld_sketch

sos run pipeline/rss_ld_sketch.ipynb process_block --ld-block-file tests/fixtures/rss_ld_sketch/protocol_example.ld_blocks.bed --chrom 22 --vcf-base output/rss_ld_sketch/rss_ld_sketch --vcf-prefix protocol_example.genotype. --output-dir output/rss_ld_sketch --W-matrix output/rss_ld_sketch/W_B50.rds --B 50 --cohort-id protocol_example --cwd output/rss_ld_sketch

sos run pipeline/rss_ld_sketch.ipynb merge_chrom --output-dir output/rss_ld_sketch --cohort-id protocol_example --chrom 22 --cwd output/rss_ld_sketch

```



### 2. Molecular Phenotypes

#### RNA-seq expression
Timing: TBD
##### 1. Assess FASTQ read quality

**What it does:** Generate per-sample FastQC reports before alignment.


```python

sos run pipeline/RNA_calling.ipynb fastqc \
    --cwd output/rnaseq/fastqc \
    --sample-list tests/fixtures/rna_calling/protocol_example.rnaseq.fastq.list.txt \
    --data-dir tests/fixtures/rna_calling/fastq

```



##### 2. Trim sequencing adapters

**What it does:** Remove configured adapter sequences and write trimmed FASTQ files for alignment.


```python

sos run pipeline/RNA_calling.ipynb fastp_trim_adaptor \
    --cwd output/rnaseq --sample-list tests/fixtures/rna_calling/protocol_example.rnaseq.fastq.list.txt \
    --data-dir tests/fixtures/rna_calling/fastq --STAR-index <path/to/STAR_Index> \
    --gtf <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.gtf> \
    --reference-fasta <path/to/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta> \
    --ref-flat <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.ref.flat>

```



##### 3. Align reads with STAR and run Picard QC

**What it does:** Map reads to the reference genome and calculate alignment-level RNA-seq quality metrics.


```python

sos run pipeline/RNA_calling.ipynb STAR_align \
    --cwd output/rnaseq/bam --sample-list tests/fixtures/rna_calling/protocol_example.rnaseq.fastq.list.txt \
    --data-dir tests/fixtures/rna_calling/fastq --STAR-index <path/to/STAR_Index> \
    --gtf <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.gtf> \
    --reference-fasta <path/to/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta> \
    --ref-flat <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.ref.flat> \
    --chimSegmentMin 0 \
    -J 50 --mem 200G --numThreads 8

```



##### 4. Quantify gene-level expression with RNA-SeQC

**What it does:** Summarize aligned reads into gene-level expression measurements.


```python

sos run pipeline/RNA_calling.ipynb rnaseqc_call \
    --cwd output/rnaseq/bam \
    --sample-list tests/fixtures/rna_calling/protocol_example.rnaseq.fastq.list.txt \
    --data-dir tests/fixtures/rna_calling/fastq \
    --gtf <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.collapse_only.gene.gtf> \
    --reference-fasta <path/to/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta> \
    --bam_list output/rnaseq/fastqc/sample_bam_list.txt

```



##### 5. Quantify transcript-level expression with RSEM

**What it does:** Estimate transcript- and gene-level abundance using the RSEM reference index.


```python

sos run pipeline/RNA_calling.ipynb rsem_call \
    --cwd output/rnaseq/bam \
    --sample-list tests/fixtures/rna_calling/protocol_example.rnaseq.fastq.list.txt \
    --data-dir tests/fixtures/rna_calling/fastq \
    --STAR-index <path/to/STAR_Index> \
    --gtf <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.gtf> \
    --reference-fasta <path/to/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta> \
    --ref-flat <path/to/Homo_sapiens.GRCh38.103.chr.reformatted.ERCC.ref.flat> \
    --bam_list output/rnaseq/fastqc/sample_bam_list.txt \
    --RSEM-index <path/to/RSEM_Index>

```



##### 6. Perform cohort-level expression QC

**What it does:** Filter low-expression features and identify expression outlier samples across the cohort.


```python

sos run pipeline/bulk_expression_QC.ipynb qc \
    --cwd output/rnaseq \
    --tpm-gct tests/fixtures/bulk_expression_normalization/protocol_example.rnaseq.tpm.gct.gz \
    --counts-gct tests/fixtures/bulk_expression_normalization/protocol_example.rnaseq.geneCount.gct.gz

```



##### 7. Normalize the QC-passed expression matrices

**What it does:** Normalize the retained count and TPM matrices and write association-ready expression phenotypes.


```python

sos run pipeline/bulk_expression_normalization.ipynb normalize \
    --cwd output/rnaseq \
    --tpm-gct output/rnaseq/protocol_example.low_expression_filtered.outlier_removed.tpm.gct.gz \
    --counts-gct output/rnaseq/protocol_example.low_expression_filtered.outlier_removed.geneCount.gct.gz \
    --annotation-gtf tests/fixtures/gene_annotation/Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz  \
    --count-threshold 1 --sample_participant_lookup tests/fixtures/bulk_expression_normalization/protocol_example.rnaseq.sample_participant_lookup.txt

```



#### Command Interface

List the workflows and parameters available in each module used by this mini-protocol.


```python

sos run pipeline/RNA_calling.ipynb -h
sos run pipeline/bulk_expression_QC.ipynb -h
sos run pipeline/bulk_expression_normalization.ipynb -h

```



#### Alternative splicing from RNA-seq data
Timing: TBD
##### 1. Quantify intron usage with LeafCutter

**What it does:** `leafcutter` extracts splice junctions and clusters introns to calculate per-sample intron excision ratios.


```python

sos run pipeline/splicing_calling.ipynb leafcutter   --cwd output/splicing/leafcutter   --samples output/rnaseq/protocol_example.rnaseq.bam.list.txt   --data-dir output/rnaseq/star_output_wasp

```



##### 2. Normalize LeafCutter ratios

**What it does:** `leafcutter_norm` filters introns and clusters, mean-imputes retained missing values, and quantile-normalizes the ratio matrix.


```python

sos run pipeline/splicing_normalization.ipynb leafcutter_norm   --cwd output/splicing/leafcutter   --ratios output/leafcutter/normalize/protocol_example.leafcutter.intron_usage_perind.counts.gz   --mean-impute

```



##### 3. Annotate LeafCutter phenotypes

**What it does:** `annotate_leafcutter_isoforms` maps introns to genes and writes the coordinate-sorted phenotype matrix and phenotype-group file used by TensorQTL.


```python

sos run pipeline/gene_annotation.ipynb annotate_leafcutter_isoforms   --cwd output/splicing/leafcutter   --phenoFile output/gene_annotation/protocol_example.leafcutter.intron_usage_perind.counts.gz_raw_data.qqnorm.txt   --intron-count tests/fixtures/gene_annotation/protocol_example.leafcutter.intron_count.tsv   --coordinate-annotation tests/fixtures/gene_annotation/Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz   --map-stra site

```



### 3. Data Preprocessing

#### Genotype preprocessing
Timing: ~3-5 min (on the toy dataset)
##### 1. Quality-control the input VCF

**What it does:** Normalize and filter the toy VCF against dbSNP and GRCh38.


```python

sos run pipeline/VCF_QC.ipynb qc \
    --genoFile tests/fixtures/vcf_qc/protocol_example.genotype.chr22.vcf.gz \
    --dbsnp-variants <path/to/00-All.add_chr.variants.gz> \
    --reference-genome <path/to/GRCh38_full_analysis_set_plus_decoy_hla.noALT_noHLA_noDecoy_ERCC.fasta> \
    --cwd output/vcf_qc \
    --skip-vcf-header-filtering True

```



##### 2. Convert the QC-passed VCF to PLINK and merge chromosomes

**What it does:** Convert the output of step 1 to PLINK. The merge command generalizes to multiple chromosome-level files.


```python

sos run pipeline/genotype_formatting.ipynb vcf_to_plink \
    --genoFile output/vcf_qc/protocol_example.genotype.chr22.leftnorm.vcf.gz \
    --cwd output/genotype_formatting/plink \
    --name protocol_example \
    -j 4

sos run pipeline/genotype_formatting.ipynb merge_plink \
    --genoFile `ls output/genotype_formatting/plink/protocol_example.genotype.chr*.bed` \
    --name protocol_example.genotype.merged \
    --cwd output/genotype_formatting/plink \
    -j 2

```



##### 3. Apply PLINK-level quality control

**What it does:** Apply genotype-, sample- and Hardy-Weinberg-equilibrium filters to the merged PLINK dataset.


```python

sos run pipeline/GWAS_QC.ipynb qc_no_prune \
    --cwd output/gwas_qc/plink \
    --genoFile output/genotype_formatting/plink/protocol_example.genotype.merged.bed \
    --geno-filter 0.1 \
    --mind-filter 0.1 \
    --hwe-filter 1e-08 \
    --mac-filter 0

```



##### 4. Partition the QC-passed genotype data by chromosome

**What it does:** Create chromosome-specific PLINK files required by chromosome-oriented downstream workflows.


```python

sos run pipeline/genotype_formatting.ipynb genotype_by_chrom \
    --genoFile output/gwas_qc/plink/protocol_example.genotype.merged.plink_qc.bed \
    --cwd output/genotype_by_chrom \
    --chrom `cut -f 1 output/gwas_qc/plink/protocol_example.genotype.merged.plink_qc.bim | uniq | sed "s/chr//g"` \
    -j 4

```



##### 5. Match genotype and molecular-phenotype samples

**What it does:** Retain the sample intersection between the QC-passed genotype data and molecular phenotype.


```python

sos run pipeline/GWAS_QC.ipynb genotype_phenotype_sample_overlap \
    --cwd output/gwas_qc/genotype \
    --genoFile output/gwas_qc/plink/protocol_example.genotype.merged.plink_qc.fam \
    --phenoFile tests/fixtures/gene_annotation/protocol_example.rnaseq.bed.gz

```



##### 6. Estimate kinship and separate related individuals

**What it does:** Use KING to identify related pairs and produce related and unrelated subsets.


```python

sos run pipeline/GWAS_QC.ipynb king \
    --cwd output/gwas_qc/kinship \
    --genoFile output/gwas_qc/plink/protocol_example.genotype.merged.plink_qc.bed \
    --name protocol_example.king \
    --keep-samples output/gwas_qc/genotype/protocol_example.rnaseq.bed.sample_genotypes.txt

```



##### 7. Prepare the unrelated, LD-pruned PCA subset

**What it does:** Apply the minor-allele-count filter and LD pruning used to estimate ancestry axes.


```python

sos run pipeline/GWAS_QC.ipynb qc \
    --cwd output/gwas_qc/genotype \
    --genoFile output/gwas_qc/kinship/protocol_example.genotype.merged.plink_qc.protocol_example.king.unrelated.bed \
    --mac-filter 5

```



##### 8. Estimate principal components in unrelated individuals

**What it does:** Estimate the PCA model and scores in unrelated individuals.


```python

sos run pipeline/PCA.ipynb flashpca \
    --cwd output/pca_uf \
    --genoFile output/gwas_qc/genotype/protocol_example.genotype.merged.plink_qc.protocol_example.king.unrelated.plink_qc.prune.bed \
    --name protocol_example

```



##### 9. Extract the related samples at the PCA variants

**What it does:** Restrict the related subset to the variants used by the unrelated-sample PCA model.


```python

sos run pipeline/GWAS_QC.ipynb qc_no_prune \
    --cwd output/pca_related \
    --genoFile output/gwas_qc/kinship/protocol_example.genotype.merged.plink_qc.protocol_example.king.related.bed \
    --geno-filter 0 --mind-filter 0.1 --maf-filter 0 \
    --keep-variants output/gwas_qc/genotype/protocol_example.genotype.merged.plink_qc.protocol_example.king.unrelated.plink_qc.prune.in \
    --name for_pca

```



##### 10. Project related samples and detect PCA outliers

**What it does:** Project related individuals into the PCA space and identify ancestry-space outliers.


```python

sos run pipeline/PCA.ipynb project_samples \
    --cwd output/pca_uf \
    --genoFile output/pca_related/protocol_example.genotype.merged.plink_qc.protocol_example.king.related.for_pca.plink_qc.extracted.bed \
    --phenoFile tests/fixtures/pca/protocol_example.pca_pheno.txt \
    --pca-model output/pca_uf/protocol_example.genotype.merged.plink_qc.protocol_example.king.unrelated.plink_qc.prune.protocol_example.pca.rds \
    --label-col race --pop-col race --name protocol_example --maha-k 2

```



##### 11. Remove projected PCA outliers

**What it does:** Remove projected outliers before recombining samples.


```python

sos run pipeline/GWAS_QC.ipynb qc_no_prune \
    --cwd output/pca_related \
    --genoFile output/pca_related/protocol_example.genotype.merged.plink_qc.protocol_example.king.related.for_pca.plink_qc.extracted.bed \
    --remove-samples output/pca_uf/protocol_example.pca_pheno.pca.projected.outliers \
    --name no_outlier

```



##### 12. Recombine unrelated and projected related samples

**What it does:** Merge the unrelated PCA subset with the retained projected related samples for downstream analysis.


```python

sos run pipeline/genotype_formatting.ipynb merge_plink \
    --genoFile output/gwas_qc/genotype/protocol_example.genotype.merged.plink_qc.protocol_example.king.unrelated.plink_qc.prune.bed \
               output/pca_related/protocol_example.genotype.merged.plink_qc.protocol_example.king.related.for_pca.plink_qc.extracted.no_outlier.plink_qc.bed \
    --cwd output/genotype_final \
    --name protocol_example.qced

```



#### Command Interface

List the workflows and parameters available in each module used by this mini-protocol.


```python

sos run pipeline/VCF_QC.ipynb -h
sos run pipeline/genotype_formatting.ipynb -h
sos run pipeline/GWAS_QC.ipynb -h
sos run pipeline/PCA.ipynb -h

```



#### Phenotype preprocessing
Timing: <12 min (on the toy dataset)
##### 1. Impute missing phenotype values

**What it does:** Use generalized empirical Bayes matrix factorization to complete the example protein matrix.


```python

sos run pipeline/phenotype_imputation.ipynb gEBMF \
    --phenoFile tests/fixtures/phenotype_imputation/protocol_example.protein.missing.bed.gz \
    --cwd output/phenotype_imputation_uf \
    --num_factor 30

```



##### 2. Add genomic coordinates to gene or protein phenotypes

**What it does:** Join phenotype identifiers to a supplied coordinate annotation and write a coordinate-aware BED matrix.


```python

sos run pipeline/gene_annotation.ipynb annotate_coord \
    --cwd output/gene_annotation \
    --phenoFile tests/fixtures/gene_annotation/protocol_example.rnaseq.bed.gz \
    --coordinate-annotation tests/fixtures/gene_annotation/Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz \
    --phenotype-id-column gene_id

```



##### 3. Retrieve gene coordinates from Ensembl BioMart

**What it does:** Query the selected Ensembl release when a local coordinate annotation is unavailable.


```python

sos run pipeline/gene_annotation.ipynb annotate_coord_biomart \
    --cwd output/gene_annotation \
    --phenoFile tests/fixtures/gene_annotation/protocol_example.rnaseq.gene_ID.tsv \
    --ensembl-version 115

```



##### 4. Map LeafCutter clusters to genes

**What it does:** Assign LeafCutter clusters to genes using splice-site overlap with the gene annotation.


```python

sos run pipeline/gene_annotation.ipynb map_leafcutter_cluster_to_gene \
    --cwd output/gene_annotation \
    --phenoFile tests/fixtures/gene_annotation/protocol_example.leafcutter.phenotype.bed.gz \
    --intron-count tests/fixtures/gene_annotation/protocol_example.leafcutter.intron_count.tsv \
    --coordinate-annotation <path/to/Homo_sapiens.GRCh38.103.chr.gtf> \
    --map-stra site

```



##### 5. Annotate LeafCutter isoforms

**What it does:** Convert LeafCutter intron-cluster phenotypes into annotated isoform features for QTL analysis.


```python

sos run pipeline/gene_annotation.ipynb annotate_leafcutter_isoforms \
    --cwd output/gene_annotation \
    --phenoFile tests/fixtures/gene_annotation/protocol_example.leafcutter.phenotype.bed.gz \
    --intron-count tests/fixtures/gene_annotation/protocol_example.leafcutter.intron_count.tsv \
    --coordinate-annotation <path/to/Homo_sapiens.GRCh38.103.chr.gtf> \
    --map-stra site

```



##### 6. Partition a BED phenotype by chromosome

**What it does:** Split a coordinate-annotated BED phenotype into chromosome-specific files.


```python

sos run pipeline/phenotype_formatting.ipynb phenotype_by_chrom \
    --cwd output/phenotype_uf \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --name protocol_example \
    --chrom chr22

```



##### 7. Partition a GCT phenotype by chromosome

**What it does:** Split a coordinate-aware GCT matrix into chromosome-specific GCT files.


```python

sos run pipeline/phenotype_formatting.ipynb phenotype_by_chrom_gct \
    --cwd output/phenotype_gct \
    --phenoFile output/phenotype/phenotype_by_chrom_for_cis/protocol_example.rnaseq.gene_tpm.gct.gz \
    --chrom chr21 chr22

```



##### 8. Partition a phenotype by predefined regions

**What it does:** Extract phenotype features falling within each region in a supplied region list.


```python

sos run pipeline/phenotype_formatting.ipynb phenotype_by_region \
    --cwd output/phenotype_by_region \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --region-list output/phenotype/phenotype_by_chrom_for_cis/protocol_example_protein.enhanced_cis_chr22.bed

```



##### 9. Define TAD-based phenotype regions

**What it does:** Assign phenotype features to TAD windows and generate a region list for downstream analysis.


```python

sos run pipeline/phenotype_formatting.ipynb phenotype_annotate_by_tad \
    --cwd output/phenotype_by_region \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --TAD-list tests/fixtures/generalized_TADB/expected/TADB_enhanced_cis.bed \
    --phenotype-per-tad 2

```



##### 10. Extract selected samples from a GCT matrix

**What it does:** Retain only samples listed in a supplied keep file.


```python

sos run pipeline/phenotype_formatting.ipynb gct_extract_samples \
    --cwd output/phenotype_gct \
    --phenoFile output/phenotype/phenotype_by_chrom_for_cis/protocol_example.rnaseq.gene_tpm.gct.gz \
    --keep-samples tests/fixtures/phenotype_formatting/keep_samples.txt

```



##### 11. Subset BAM files by genomic region

**What it does:** Extract selected chromosomes or regions from every BAM listed in the input manifest.


```python

sos run pipeline/phenotype_formatting.ipynb bam_subsetting \
    --cwd output/bam_subset \
    --phenoFile output/phenotype/phenotype_by_chrom_for_cis/bam_file_list.txt \
    --region chr21 chr22

```



#### Command Interface

List the workflows and parameters available in each module used by this mini-protocol.


```python

sos run pipeline/phenotype_imputation.ipynb -h
sos run pipeline/gene_annotation.ipynb -h
sos run pipeline/phenotype_formatting.ipynb -h

```



#### Covariate data preprocessing
Timing: <3 min (on the toy dataset)
##### 1. Merge observed covariates and genotype PCs

**What it does:** Combine the base covariate table with the selected genotype principal components.


```python

sos run pipeline/covariate_formatting.ipynb merge_genotype_pc \
    --cwd output/covariate \
    --pcaFile output/genotype/genotype_pca/protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.rds \
    --covFile tests/fixtures/covariate_formatting/covariates.base.tsv \
    --name protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca \
    --tol-cov 0.4 \
    --k `awk '$3 < 0.8' output/genotype/genotype_pca/protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.scree.txt | tail -1 | cut -f 1`

```



##### 2. Infer PCA factors with Marchenko–Pastur selection

**What it does:** Residualize the phenotype and automatically retain PCA factors above the Marchenko–Pastur noise threshold.


```python

sos run pipeline/covariate_hidden_factor.ipynb Marchenko_PC \
    --cwd output/covariate \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --covFile output/covariate/protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.gz \
    --mean-impute-missing

```



##### 3. Infer PEER factors

**What it does:** Residualize the phenotype and estimate the requested number of probabilistic PEER factors.


```python

sos run pipeline/covariate_hidden_factor.ipynb PEER \
    --cwd output/covariate \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --covFile output/covariate/protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.gz \
    --N 3

```



##### 4. Infer configurable PCA factors

**What it does:** Residualize the phenotype and estimate PCA factors using the selected dimension rule.


```python

sos run pipeline/covariate_hidden_factor.ipynb PCA \
    --cwd output/covariate \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --covFile output/covariate/protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.gz \
    --choose_k_method Marchenko \
    --mean-impute-missing

```



##### 5. Infer factors with bi-cross-validation

**What it does:** Residualize the phenotype and select latent structure using the BiCV workflow.


```python

sos run pipeline/covariate_hidden_factor.ipynb BiCV \
    --cwd output/covariate \
    --phenoFile tests/fixtures/phenotype_formatting/protocol_example.rnaseq.bed.bed.gz \
    --covFile output/covariate/protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.gz \
    --N 3

```



#### Command Interface

List the workflows and parameters available in each module used by this mini-protocol.


```python

sos run pipeline/covariate_formatting.ipynb -h
sos run pipeline/covariate_hidden_factor.ipynb -h

```



### 4. QTL Association Testing

#### QTL association testing
Timing: TBD
##### 1. cis-QTL scan

**What it does:** Tests each molecular trait against variants within its cis window, using `--MAC 5` for the small chromosome 22 example.


```python

sos run pipeline/TensorQTL.ipynb cis \
    --genotype-file output/genotype_by_chrom/protocol_example.genotype.merged.plink_qc.genotype_by_chrom_files.txt \
    --phenotype-file output/phenotype/phenotype_by_chrom_for_cis/bulk_rnaseq.phenotype_by_chrom_files.txt \
    --covariate-file output/covariate/protocol_example.rnaseq.bed.protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.Marchenko_PC.gz \
    --cwd output/tensorqtl_cis --name protocol_example --MAC 5 --numThreads 2

```



##### 2. trans-QTL scan

**What it does:** Tests the selected traits against variants on chromosome 22, restricting traits to the identifiers listed in `data/combined_AD_genes.csv`.


```python

sos run pipeline/TensorQTL.ipynb trans \
    --genotype-file output/genotype_by_chrom/protocol_example.genotype.merged.plink_qc.genotype_by_chrom_files.txt \
    --phenotype-file output/phenotype/phenotype_by_chrom_for_cis/bulk_rnaseq.phenotype_by_chrom_files.txt \
    --covariate-file output/covariate/protocol_example.rnaseq.bed.protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.Marchenko_PC.gz \
    --cwd output/tensorqtl_trans --name protocol_example --MAC 5 --numThreads 2 \
    --trans-geno-chromosome 22 --region-list data/combined_AD_genes.csv --region-list-phenotype-column 4

```



##### 3. interaction-QTL scan

**What it does:** Runs the cis model with a genotype-by-`msex` interaction term and reports evidence that the genotype effect changes with this covariate.


```python

sos run pipeline/TensorQTL.ipynb cis \
    --genotype-file output/genotype_by_chrom/protocol_example.genotype.merged.plink_qc.genotype_by_chrom_files.txt \
    --phenotype-file output/phenotype/phenotype_by_chrom_for_cis/bulk_rnaseq.phenotype_by_chrom_files.txt \
    --covariate-file output/covariate/protocol_example.rnaseq.bed.protocol_example.covariates.protocol_example.genotype.merged.plink_qc.plink_qc.prune.pca.Marchenko_PC.gz \
    --cwd output/tensorqtl_int --name protocol_example --MAC 5 --numThreads 2 \
    --interaction msex --maf-threshold 0.05 --no-permutation

```



### 5. Multivariate Mixture Models

### 6. Multiomics Regression Models

#### Fine-mapping and TWAS analysis
Timing: TBD
##### 1. Univariate fine-mapping and TWAS

**What it does:** `qtl_dataset_construct+susie_twas` builds the regional dataset, fits SuSiE and saves fine-mapping results and cross-validated TWAS weights.


```python

sos run pipeline/mnm_regression.ipynb qtl_dataset_construct+susie_twas   --name protocol_example --cwd output/mnm/univariate   --genoFile tests/fixtures/qtl_mini/protocol_example.genotype.chr22.bed   --phenoFile tests/fixtures/qtl_mini/protocol_example.pheno_manifest_context.tsv   --covFile tests/fixtures/qtl_mini/example_covariates.tsv   --customized-association-windows tests/fixtures/qtl_mini/association_windows.bed   --region-name ENSG00000130538 --transpose-covariates --save-data -j1

```



##### 2. Multivariate fine-mapping

**What it does:** `mnm` jointly fine-maps multiple molecular traits using the analyses and prior settings specified by the fine-mapping metadata.


```python

sos run pipeline/mnm_regression.ipynb mnm   --name protocol_example --cwd output/mnm/multivariate   --genoFile tests/fixtures/qtl_mini/protocol_example.genotype.chr22.bed   --phenoFile tests/fixtures/qtl_mini/protocol_example.pheno_manifest_context.tsv   --covFile tests/fixtures/qtl_mini/example_covariates.tsv   --customized-association-windows tests/fixtures/qtl_mini/association_windows.bed   --fine-mapping-meta tests/fixtures/qtl_mini/fine_mapping_meta.tsv   --transpose-covariates --save-data -j1

```



##### 3. Multigene multivariate fine-mapping

**What it does:** `mnm_genes` coordinates multivariate fine-mapping across genes while preserving phenotype identifiers and a common retained-sample set.


```python

sos run pipeline/mnm_regression.ipynb mnm_genes   --name protocol_example --cwd output/mnm/multigene   --genoFile tests/fixtures/qtl_mini/protocol_example.genotype.chr22.bed   --phenoFile tests/fixtures/qtl_mini/protocol_example.pheno_manifest_context.tsv   --covFile tests/fixtures/qtl_mini/example_covariates.tsv   --customized-association-windows tests/fixtures/qtl_mini/association_windows.bed   --pheno-id-map-file tests/fixtures/qtl_mini/pheno_id_map.tsv   --fine-mapping-meta tests/fixtures/qtl_mini/fine_mapping_meta.tsv   --keep-samples tests/fixtures/qtl_mini/keep_samples.txt --save-data -j1

```



##### 4. Functional fine-mapping

**What it does:** `fsusie` applies functional SuSiE to epigenomic or other functional phenotypes and saves posterior fine-mapping results and residual data.


```python

sos run pipeline/mnm_regression.ipynb fsusie   --name protocol_example --cwd output/mnm/fsusie   --genoFile tests/fixtures/qtl_mini/protocol_example.genotype.chr22.bed   --phenoFile tests/fixtures/qtl_mini/pheno_manifest.tsv   --covFile tests/fixtures/covariate_hidden_factor/covariates.tsv   --customized-association-windows tests/fixtures/qtl_mini/association_windows.bed   --cis-window 0 --max-cv-variants 5000 --save-data -j1

```



##### 5. Summary-statistic fine-mapping

**What it does:** the RSS chain harmonizes regional GWAS summary statistics with the LD reference, runs SuSiE-RSS fine-mapping and produces a regional diagnostic plot.


```python

sos run pipeline/rss_analysis.ipynb   generate_manifest+generate_gwas_sumstats+gwas_fine_mapping+gwas_rss_plot   --cwd output/mnm/rss --modular-script-dir code/script   --gwas-meta tests/fixtures/rss_analysis/protocol_example.rss_mwe.gwas_meta.tsv   --regions chr22:49355984-50799822   --ld-meta tests/fixtures/ld_reference/ld_meta_file.tsv

```



### 7. GWAS Integration

### 8. Enrichment and Validation


## Timing




| Step | Time|
|------|-----|
|Reference data|X minutes|
|Molecular Phenotypes|X minutes|
|Data Preprocessing|X minutes|
|QTL Association Testing|X minutes|
|Multivariate Mixture Models|X minutes|
|Multiomics Regression Models|X minutes|
|GWAS Integration|X minutes|
|Enrichment and Validation|X minutes|

## Troubleshooting




## Anticipated Results




####  Reference data preparation

## Command Interface

####  RNA-seq expression

<div style="text-align: center;">

####  Alternative splicing from RNA-seq data

## Command Interface
####  Genotype preprocessing

## Command Interface

####  Phenotype preprocessing

## Command Interface

####  Covariate data preprocessing

## Command Interface

####  QTL association testing

## Command Interface

####  Fine-mapping and TWAS analysis

## Command Interface

## Figures




## Tables




## Supplementary Information




## Author Contributions Statements



## Acknowledgements



## Competing Interests



## References





## Keywords




