---
title: Running on a cluster
---

Every pipeline on this site can send its work to a SLURM or SGE cluster. You set it up once, then add two options to any command: `-c <host file> -q <queue>`.

## How it works

Each module splits its work into **tasks**: one per chromosome, per region, per sample batch, and so on. Each task already declares the memory, time and CPU cores it needs. When you add `-q`, SoS writes a small job script for each task, submits it to your scheduler, waits for it to finish, and then moves on to the next step. Without `-q`, the same tasks run one after another on the machine you are on.

You run `sos run` once; the cluster runs the tasks. The `sos run` command itself is light, but it must stay running until the last task finishes (see [Keep the pipeline running](#keep-running)).

## Set up once

**1. Install the scheduler plugin.** The pixi environment from [Environment Setup](../code/SoS/xqtl_protocol_demo.ipynb) includes SoS but not its SLURM/SGE plugin:

```bash
pixi global install -c conda-forge --environment python sos-pbs
```

This can take several minutes, because pixi re-checks the whole Python environment.

**2. Copy the host file for your scheduler** into your clone of the protocol:

| Scheduler | Template | Queue name to use |
|---|---|---|
| SLURM (`sbatch`, `squeue`) | [`code/cluster/slurm.yml`](https://github.com/StatFunGen/xqtl-protocol/blob/main/code/cluster/slurm.yml) | `slurm` |
| SGE (`qsub`, `qstat`) | [`code/cluster/sge.yml`](https://github.com/StatFunGen/xqtl-protocol/blob/main/code/cluster/sge.yml) | `sge` |
| SGE at Boston University (SCC), as a worked example | [`code/cluster/bu_scc.yml`](https://github.com/StatFunGen/xqtl-protocol/blob/main/code/cluster/bu_scc.yml) | `sge` |

```bash
cp code/cluster/slurm.yml my_cluster.yml   # or code/cluster/sge.yml
```

Not sure which scheduler you have? `which sbatch qsub` shows which one is installed.

**3. Edit the lines marked `EDIT`.** Usually that means:

- the largest memory, cores and time your cluster allows (`max_mem`, `max_cores`, `max_walltime`);
- an account, partition, project or queue line, if your cluster requires one (ask your HPC help desk, or look at a job script you already use);
- for SGE only: the name of the parallel environment and the memory resource, which differ between sites. `qconf -spl` lists parallel environments.

**4. Test it** with the toy workflow, which writes 10 small files using 10 jobs:

```bash
sos run code/SoS/misc/Job_Example.ipynb toy_example -c my_cluster.yml -q slurm   # or -q sge
```

While it runs, `squeue -u $USER` (SLURM) or `qstat -u $USER` (SGE) should list the jobs. When it finishes, `File_1.out` to `File_10.out` are in the current folder.

## Run a module on the cluster

Take any command from a module's Minimal Working Example and add the two options. For example, the gene annotation example from Environment Setup:

```bash
sos run pipeline/gene_annotation.ipynb annotate_coord \
    --phenoFile tests/fixtures/gene_annotation/protocol_example.rnaseq.bed.gz \
    --coordinate-annotation tests/fixtures/gene_annotation/Homo_sapiens.GRCh38.103.collapse_only.gene.chr22.gtf.gz \
    --phenotype-id-column gene_id \
    --modular-script-dir code/script --cwd output/gene_annotation \
    -c my_cluster.yml -q slurm
```

Three options set what each job asks for. They are the same in every module:

| Option | Meaning | Example |
|---|---|---|
| `--mem` | memory per task | `--mem 40G` |
| `--walltime` | time limit per task | `--walltime 24h` |
| `--numThreads` | CPU cores per task | `--numThreads 8` |
| `--job-size` | how many tasks share one job; raise it when a module makes thousands of short tasks | `--job-size 200` |

Each module's default values are in [Default resources by module](#default-resources) below, and `sos run pipeline/<module>.ipynb -h` prints them. The defaults suit the example data; for a full cohort, start from them and raise `--mem` or `--walltime` if jobs are killed.

Options with one dash (`-c`, `-q`, `-s`, `-j`) belong to SoS; options with two dashes belong to the module.

(keep-running)=
## Keep the pipeline running

`sos run` has to stay alive until its last job finishes, which for a full cohort can take hours or days. Two common ways:

**Submit `sos run` itself as a small job** (recommended; some clusters require it). The job needs little memory but a long time limit:

```bash
#!/bin/bash
#SBATCH --job-name=xqtl_driver
#SBATCH --time=72:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=xqtl_driver.%j.log

cd /path/to/xqtl-protocol
sos run pipeline/<module>.ipynb <workflow> <options> -c my_cluster.yml -q slurm
```

Save it as `driver.sh` and submit with `sbatch driver.sh`. On SGE, use the same body with `#$ -N xqtl_driver`, `#$ -l h_rt=72:00:00`, `#$ -cwd`, `#$ -j y` and submit with `qsub driver.sh`.

**Or run it inside `tmux` or `screen` on a login node**, if your cluster allows long-running processes there.

## Monitor, resume and rerun

Add the same `-c my_cluster.yml -q slurm` (or `-q sge`) to these so SoS looks at the right queue.

| To... | Run |
|---|---|
| See the state of every task | `sos status -c my_cluster.yml -q slurm -v 3` |
| Read a failed task's error | open `~/.sos/tasks/<task id>.err`, or run `sos status <task id> -c my_cluster.yml -q slurm -v 4` |
| Resume after a failure or a killed driver | run the same `sos run` command again: finished tasks are skipped |
| Redo steps whose outputs already exist | add `-s force` |
| Stop the pipeline | press Ctrl-C twice in the terminal running `sos run`; jobs already in the queue finish or can be removed with `scancel` / `qdel` |
| Clean up records of finished tasks | `sos purge -s completed -c my_cluster.yml -q slurm` |

(default-resources)=
## Default resources by module

What each task asks for unless you override it, as set in each module's code. Run times depend on your data; these defaults are starting points, not measurements.

| Module | `--mem` | `--walltime` | `--numThreads` | `--job-size` |
|---|---|---|---|---|
| Reference Data Standardization | 16G | 5h | 8 | 1 |
| Generalized TADB | 40G | 20h | 1 | 1 |
| LD pruning | 80G | 1h | 10 | 1 |
| RSS LD sketch | 32G | 24h | 8 | 1 |
| RNA-seq expression calling | 16G | 5h | 8 | 1 |
| Bulk RNA-seq QC | 16G | 5h | 8 | 1 |
| Bulk RNA-seq normalization | 16G | 5h | 20 | 1 |
| snRNA-seq preprocessing | 16G | 5h | 8 | 1 |
| Pseudobulk preprocessing | 16G | 5h | 8 | 1 |
| Methylation calling | 16G | 5h | 8 | 1 |
| Splicing calling (LeafCutter, LeafCutter2) | 16G | 5h | 8 | 1 |
| Isoform usage (Torino) | 16G | 5h | 1 | 1 |
| Splicing normalization | 16G | 5h | 8 | 1 |
| APA calling | 200G | 400h | 8 | 1 |
| APA imputation | 32G | 40h | 16 | 1 |
| Genotype VCF QC | 60G | 24h | 1 | 1 |
| Genotype formatting | 16G | 5h | 20 | 1 |
| Genotype PLINK QC | 16G | 5h | 20 | 1 |
| PCA | 16G | 5h | 10 | 1 |
| Gene coordinate annotation | 16G | 5h | 1 | 1 |
| Phenotype imputation | 16G | 72h | 20 | 1 |
| Phenotype formatting | 16G | 5h | 20 | 1 |
| Covariate formatting | 2G | 5h | 8 | 1 |
| Hidden factor analysis | 16G | 5h | 8 | 1 |
| TensorQTL | 16G | 12h | 8 | 1 |
| Quantile regression and TWAS | 20G | 1h | 1 | 200 |
| MASH preprocessing | 16G | 1h | 1 | 1 |
| Mixture prior | 16G | 5h | 1 | 1 |
| MASH fit | 16G | 5h | 1 | 1 |
| MASH posterior | 16G | 1h | 1 | 1 |
| Fine-mapping and TWAS (mnm_regression) | 20G | 1h | 1 | 200 |
| RSS fine-mapping (GWAS) | 16G | 2h | 1 | 1 |
| RSS fine-mapping (QTL summary statistics) | 16G | 5h | 1 | 1 |
| Fine-mapping post-processing | 6G | 96h | 2 | 1 |
| SuSiE-enloc | 16G | 5m | 1 | 200 |
| TWAS, cTWAS and MR | 8G | 5m | 1 | 100 |
| INTACT | 8G | 5h | 1 | 1 |
| ColocBoost | 20G | 1h | 1 | 200 |
| Block-jackknife enrichment | 16G | 12h | 8 | 1 |
| Pathway analysis | 16G | 12h | 8 | 1 |
| S-LDSC enrichment | 16G | 12h | 16 | 1 |

## Common problems

**`Failed to load task engine of type "pbs"`.** The `sos-pbs` plugin is missing; install it (step 1).

**`requested more mem ... than allowed max_mem`.** A task asked for more than the `max_mem` in your host file. Lower `--mem`, or raise `max_mem` if your cluster allows it.

**Every task fails with `does not match any existing task`.** The job started with a different home directory than your login shell, so SoS could not find its task files. The templates set `HOME` inside each job to prevent this; keep that line if you edit the template.

**Jobs sit in the queue and never start.** The request may exceed what any node offers, or your cluster needs an account or partition line. Check with `squeue -u $USER --start` (SLURM) or `qstat -j <job id>` (SGE).

**Jobs are killed for memory or time.** Rerun the same command with a higher `--mem` or `--walltime`; finished tasks are not repeated.

**Thousands of tiny jobs flood the queue.** Raise `--job-size` so several tasks share one job, and lower `max_running_jobs` in the host file.
