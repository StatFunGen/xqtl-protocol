---
title: Using your own data
---

Every command on this site runs on the example data in `tests/fixtures/`. To run it on your study, you swap those example files for yours. This page lists what your files must look like so the swap works the first time.

## The general rule

1. Run the module's Minimal Working Example on the example data first, so you know the command works on your machine or cluster.
2. Open the example input files (`zcat file.gz | head`) and make your files match them: same columns, same header line, same compression and index files.
3. Replace each `tests/fixtures/...` path in the command with your file, and change `--cwd` to a new output folder.
4. Run with `-n` first. SoS checks the inputs and prints what it would run, without running it.
5. Run it for real, on a small piece first (one chromosome or a few regions), then on everything.

## Rules that apply to every input

- **Genome build: GRCh38 (hg38).** Reference files, annotations and LD panels in the protocol are all GRCh38. Lift over data on another build first.
- **Chromosome names start with `chr`** (`chr22`, not `22`) in every file: VCF, BED, GTF and FASTA. If your VCFs use `22`, run [`VCF_QC.ipynb rename_chrs`](../code/SoS/data_preprocessing/genotype/VCF_QC.ipynb) first.
- **Sample IDs must match across files.** The column names of your phenotype matrix, the sample IDs in your genotypes and the columns of your covariate file must be the same IDs. If your phenotype and genotype data name samples differently, make a two-column lookup table (`sample_id`, `participant_id`) and pass it with `--sample-participant-lookup` where a module offers it.
- **Compressed files are bgzipped and indexed.** `.vcf.gz` and `.bed.gz` files need `bgzip` (not plain `gzip`) and a `.tbi` index from `tabix`.

## Checklist by input

### Genotypes

- [ ] One VCF per chromosome (or one VCF for all), bgzipped, with `.tbi` index
- [ ] Chromosome names start with `chr`; build is GRCh38
- [ ] Sample IDs are the IDs your phenotype and covariate files use, or you have a lookup table
- [ ] If your VCF has only `GT` in FORMAT (no `DP`, `GQ`, `AD`), add `--gt-only-vcf-qc` to VCF QC

Starts at: [Genotype VCF File Quality Control](../code/SoS/data_preprocessing/genotype/VCF_QC.ipynb). Example: `tests/fixtures/vcf_qc/protocol_example.genotype.chr22.vcf.gz`.

### Molecular phenotype matrix (expression, splicing, methylation, protein)

- [ ] One row per feature (gene, intron cluster, probe), one column per sample
- [ ] BED layout: the first four columns are `#chr`, `start`, `end`, `ID`, then one column per sample
- [ ] If your matrix has only feature IDs and no coordinates, add them with [Gene Coordinate Annotation](../code/SoS/data_preprocessing/phenotype/gene_annotation.ipynb)
- [ ] bgzipped and tabix-indexed; sorted by chromosome and start

```
#chr   start     end       ID               SAMPLE_001  SAMPLE_002
chr22  10939387  10961337  ENSG00000283047  37.077      0.0
```

Starts at: [Phenotype preprocessing](../code/SoS/data_preprocessing/phenotype_preprocessing.ipynb). If you start from raw reads, begin with your data type under Molecular Phenotypes instead.

### Raw RNA-seq reads

- [ ] A tab-separated sample list with a header: `ID`, `fq1`, and `fq2` for paired-end; `strand` and `read_length` are optional
- [ ] `--data-dir` points at the folder holding the FASTQ files named in that list
- [ ] STAR and RSEM indices built from the same genome FASTA and GTF ([Reference Data Standardization](../code/SoS/reference_data/reference_data_preparation.ipynb))

Starts at: [Quantifying expression from RNA-seq data](../code/SoS/molecular_phenotypes/calling/RNA_calling.ipynb).

### Covariates

- [ ] One row per covariate, one column per sample, first header cell `#id`
- [ ] Same sample IDs as the phenotype matrix
- [ ] Numeric values (for example sex coded 1/2, each batch as its own 0/1 row)

```
#id  SAMPLE_001  SAMPLE_002  SAMPLE_003
sex  1           1           2
age  90.97       80.24       83.9
```

Starts at: [Covariate data preprocessing](../code/SoS/data_preprocessing/covariate_preprocessing.ipynb). Genotype PCs and hidden factors are added for you there.

### GWAS summary statistics (for colocalization, TWAS, MR)

- [ ] GRCh38 positions
- [ ] Effect size, standard error and both alleles for each variant
- [ ] An LD reference panel that matches your GWAS ancestry

Starts at: [GWAS and xQTL integration mini-protocol](../code/SoS/pecotmr_integration/gwas_integration.ipynb).

## Check before the long run

- **Do the samples overlap?** [`GWAS_QC.ipynb genotype_phenotype_sample_overlap`](../code/SoS/data_preprocessing/genotype/GWAS_QC.ipynb) lists the samples present in both your genotypes and your phenotype file. If the list is short or empty, the IDs do not match.
- **Do chromosome names match?** `zcat your.vcf.gz | grep -v '^#' | cut -f1 | uniq` and `zcat your.bed.gz | cut -f1 | uniq` should both print `chr1`, `chr2`, ...
- **Does it run on a small piece?** Many modules take a per-chromosome file list or a region list; try `chr22` alone before the whole genome.

When the inputs are ready, run the full analysis on your cluster: see [Running on a cluster](running_on_a_cluster.md).
