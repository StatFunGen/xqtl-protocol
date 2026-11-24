# FunGen-xQTL Computational Protocol

[![Tests](https://github.com/StatFunGen/xqtl-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/StatFunGen/xqtl-protocol/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/StatFunGen/xqtl-protocol/blob/main/LICENSE)
<!-- end-badges -->

Reproducible, reusable pipelines for molecular QTL analysis, developed for the NIH/NIA
Alzheimer's Disease Sequencing Project (ADSP) Functional Genomics xQTL (FunGen-xQTL)
project. A molecular QTL (xQTL) is a genetic variant linked to a change in a molecular
trait, such as gene expression, splicing, protein level or DNA methylation.
<!-- end-intro -->

## Getting started

| I want to... | Go to |
|---|---|
| **Set up my computing environment** | [Environment Setup](code/SoS/xqtl_protocol_demo.ipynb) |
| **Work out which pipelines I need** | [xQTL Analysis Workflow Builder](https://statfungen.github.io/xqtl-protocol/xqtl_protocol_workflow_builder.html): answer a few questions, get the modules and commands for your route |

[Where to get reference data](https://statfungen.github.io/xqtl-protocol/reference-data-sources/) is under Reference data in the sidebar; [Methods and references](https://statfungen.github.io/xqtl-protocol/methods-and-references/) and the [Glossary](https://statfungen.github.io/xqtl-protocol/glossary/) are under Reference.

<!-- site-home-continues -->
## What the protocol covers

| Stage | What it does |
|---|---|
| **Reference data** | Prepare the genome, gene annotation, alignment indices, TAD-based association windows and LD reference |
| **Molecular phenotypes** | Quantify expression, splicing, methylation and polyadenylation from bulk RNA-seq, single-nuclei or array data |
| **Data preprocessing** | Genotype QC and formatting, phenotype imputation and formatting, covariate and hidden-factor estimation |
| **QTL association testing** | *cis* and *trans* scans with TensorQTL, quantile-regression QTL, hierarchical multiple testing |
| **Multivariate modeling** | MASH, to borrow information across tissues and cell types |
| **Fine-mapping and TWAS** | SuSiE, mvSuSiE, fSuSiE and SuSiE-RSS, from individual-level data or summary statistics, with TWAS weights |
| **GWAS integration** | Colocalization (SuSiE-enloc, ColocBoost), TWAS and cTWAS, INTACT |
| **Enrichment and validation** | Pathway analysis, GREGOR, stratified LD score regression |

Not every project needs every stage. The
[xQTL Analysis Workflow Builder](https://statfungen.github.io/xqtl-protocol/xqtl_protocol_workflow_builder.html)
asks what you measured and what you want to find out, then shows the modules on
that route with the commands to run them.

## Overview of the protocol
<!-- site-overview -->

### Standardized reference data

The ADSP FGC Standardization Workgroup curates the reference data with NIAGADS.
The standardized files and their specifications are on
[the AD Knowledge Portal (syn36416587)](https://www.synapse.org/Synapse:syn36416587).
[Where to get reference data](https://statfungen.github.io/xqtl-protocol/reference-data-sources/)
lists each file, its source and the modules that use it.

### Software environment

We use a set of packages from the Conda ecosystem to deploy our software. Most
packages are from [conda-forge](https://conda-forge.org) and
[bioconda](https://bioconda.github.io), along with a
[custom channel](https://anaconda.org/dnachun) for software unavailable from those
repositories. Installation is managed through
[pixi-setup](https://github.com/StatFunGen/pixi-setup); native support is provided
for Linux and macOS (Intel and Apple Silicon), and Windows users will need Windows
Subsystem for Linux (WSL).

### Pipeline execution

Pipelines in this repository are written in the
[Script of Scripts (SoS) workflow language](https://vatlab.github.io/sos-docs/).
Each pipeline is a Jupyter notebook that holds both the explanation and the
pipeline code. The same notebook runs as a command-line script with
`sos run`, on a local computer or on a cluster, where SoS submits each task as a
job.

To run on your own data or on a SLURM or SGE cluster, use the
[Workflow Builder](https://statfungen.github.io/xqtl-protocol/xqtl-protocol-workflow-builder/)
to prepare the required inputs and generate the commands for your route.

### Source code

The source code of the pipelines is in the
[`code` folder on GitHub](https://github.com/statfungen/xqtl-protocol/tree/main/code).

<!-- site-overview-end -->
## How to use the resource
<!-- site-use -->

### Organization of the resource

This website is generated from files
under the `code` folder of the source code repository. The `pipeline` folder
contains symbolic links automatically generated for pipeline files under `code`, so
analyses are run from the root of the repository by typing
`sos run pipeline/<pipeline_file>.ipynb`.

The left sidebar (the menu button on a narrow screen) follows the order of an
xQTL analysis, one analysis stage per entry:

- **Mini-protocols** are the pages directly under each stage. Each walks through
  one analysis step by step, with the commands to run in a terminal or in Jupyter.
- **Pipeline modules** are listed under each mini-protocol (open it with the
  arrow). Each module is a self-contained command-line pipeline that can also be
  used outside xQTL analysis.

### Example data

Every module has example data committed under `tests/fixtures/`, subset to chr22 to
keep the repository small. These are the same files the test suite runs against, so
the commands shown in the documentation can be run directly after cloning, with no
download required. Every path shown resolves either to one of these fixtures or to
the output directory of the step that produces it.

Large reference files (genome FASTA, GTF annotations, aligner indices, external
LD panels) are not distributed with the repository. [Where to get reference data](https://statfungen.github.io/xqtl-protocol/reference-data-sources/)
lists their sources and the modules that use them.

### See Also

- [FunGen-xQTL analyses](https://github.com/cumc/xqtl-analysis) that use this protocol (lab access only)
- [FunGen-xQTL resource catalog](https://adsp-fgc.niagads.org/xqtl-resources/): the datasets and results, and where to get them
- [pecotmr](https://statfungen.github.io/pecotmr): the R package behind the fine-mapping, colocalization, TWAS and Mendelian randomization steps ([source](https://github.com/StatFunGen/pecotmr))
- [AD Loci Explorer](https://xqtl.niagads.org/AD-loci-explorer/): the AD GWAS loci linked to their target genes
- [About](https://statfungen.github.io/xqtl-protocol/about/): who develops the protocol, funding, license and contributors

<!-- site-use-end -->
## What's new

- **November 2026:** the Workflow Builder supports your own data and can export a route for a local terminal, SLURM or SGE.
- **November 2026:** [Torino](https://statfungen.github.io/xqtl-protocol/isoform-calling-torino/) module for isoform usage without a reference annotation, next to LeafCutter and LeafCutter2 for splicing.
- **November 2026:** the [xQTL Analysis Workflow Builder](https://statfungen.github.io/xqtl-protocol/xqtl_protocol_workflow_builder.html) asks one question at a time and gives the modules and example commands for your route.

All changes are in the [commit history](https://github.com/StatFunGen/xqtl-protocol/commits/main).

## How to cite

The FunGen-xQTL protocol paper is in preparation. Until it is out, please cite this
repository (https://github.com/StatFunGen/xqtl-protocol) and the papers for the
methods you use, such as SuSiE, ColocBoost or TensorQTL. [Methods and references](https://statfungen.github.io/xqtl-protocol/methods-and-references/)
lists the paper for each method by module.

## Get help

- Something not working, or a question about a module: [open an issue on GitHub](https://github.com/StatFunGen/xqtl-protocol/issues).
- Include the command you ran, the error message and the output of `pixi list` if it is about the software environment.
