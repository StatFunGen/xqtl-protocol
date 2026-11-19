---
title: FunGen-xQTL Computational Protocol
site:
  hide_title_block: true
---

% The site's home page. README.md is the GitHub front page; this page opens with the same
% hero banner as the Resources site and the AD Loci Explorer, then the start cards,
% which GitHub cannot render.

<div class="fx-hero">
<img class="fx-hero__mark fx-only-light" src="fungen_mark.svg" alt="" width="136" height="136">
<img class="fx-hero__mark fx-only-dark" src="fungen_mark_dark.svg" alt="" width="136" height="136">
<div class="fx-hero__text">
<p class="fx-hero__org">ADSP Functional Genomics Consortium (FunGen-AD)</p>
<h1 class="fx-hero__title">FunGen<span class="fx-hero__x">-xQTL</span> Protocol</h1>
<p class="fx-hero__lead">Reproducible, reusable pipelines for molecular QTL (xQTL) analysis, from reference data and molecular phenotypes to association scans, fine-mapping, colocalization and TWAS. A molecular QTL is a genetic variant linked to a change in a molecular trait, such as gene expression, splicing, protein level or DNA methylation.</p>
<p class="fx-hero__btns"><a class="fx-pill fx-pill--solid" href="https://statfungen.github.io/xqtl-protocol/xqtl-protocol-workflow-builder/">Find the pipelines you need</a><a class="fx-pill" href="https://statfungen.github.io/xqtl-protocol/xqtl-protocol-demo/">Try the example data</a></p>
</div>
</div>

## Where do you want to start?

::::{grid} 1 2 4 4
:class: fx-start

:::{card} Try the example data
:link: ../code/SoS/xqtl_protocol_demo.ipynb
Install the software, then run your first module on the small example data that comes with the repository.
:::

:::{card} Find the pipelines you need
:link: ../website/_generated/xqtl_protocol_workflow_builder.ipynb
Answer a few questions about your data and goals. Get the modules on your route and the commands to run them.
:::

:::{card} Use your own data
:link: own_data.md
File formats, sample IDs and checks for each input, so your files work the first time.
:::

:::{card} Run on a cluster
:link: running_on_a_cluster.md
Set up SLURM or SGE once, then send any command's jobs to your cluster.
:::
::::

```{include} ../README.md
:start-after: <!-- site-home-continues -->
```
