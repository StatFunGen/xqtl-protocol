# Module notebook format

Derived from `rss_ld_sketch`, `mash_fit`, `mnm_regression` and `mash_posterior`.
Applies to module notebooks under `code/SoS/`, not to mini-protocols or vignettes.

## Section order

These H2 headings, in this order, and no others:

```
# <Title>            + one-line subtitle
## Overview
## Input
## Output
## Minimal Working Example
## Command Interface
## Workflow implementation
### Troubleshooting   (optional, very last)
```

No `## Steps`, `## Aim`, `## Methods`, `## Prerequisites`, `## Description`,
`## Anticipated Results`, `## Other workflows`. Fold their content into the
section it belongs to; if a heading is worth keeping, demote it to `###`
inside the owning section.

## Title

H1 followed by a single sentence saying what the pipeline produces.

## Overview

Mechanism first, in prose. State the key idea as a complete sentence inside the prose --
do not use a `**Key idea:**` label. Where the maths needs its symbols defined,
`**Matrix dimensions:**` may introduce a list of symbol, shape and meaning.

Name the workflows in one sentence, in the order they run.

Do not add a note about the example being toy data, and strip any that exist. Every
notebook uses toy data; saying so each time is noise.

End with `**When to run it.**` - position in the pipeline, what comes before and after.

Cite where the content warrants it - a specific method paper behind the maths being
described. Not every notebook needs a citation.

Citations go inline at the end of the sentence they support, short form with
the link on the short form:

```
... shrunk towards those patterns rather than towards zero ([Wang et al., 2020](https://doi.org/10.1111/rssb.12388)).
```

Never a standalone `Method reference: [full citation](doi)` line.

## Input

One bullet per command-line flag. File-valued flags carry the example path, a
parenthesised description on its own line, then a fenced preview:

```
- `--ld-block-file` **`input/rss_ld_sketch/protocol_example.ld_blocks.bed`**
(regions to sketch, columns `chr`, `start`, `end` in 0-based half-open coordinates)

<fenced preview: header + first two rows>
```

Scalar flags collapse onto one line:

```
- `--n-samples 60` (number of individuals in the VCF. Must match the VCF sample count.)
```

Preview by file type: header plus first rows for text; an R `str()`-style dump
for `.rds`; nothing at all for binary formats (`.bed`, `.pgen`, `.bam`) - name
them and describe them in prose instead.

There is no separate `### Example input files` heading. Each preview sits
directly under the flag that consumes it.

Where a notebook has two input modes, head each with a bold label
(`**Gene-expression mode ...**`, `**Peak mode ...**`) and repeat the flag list.

## Output

One bullet per file the pipeline writes, followed by a structure dump.

No lead-in sentence under the heading - the first bullet follows it directly. In
particular do not restate that everything lands under `--cwd`; the paths show it.

Filenames come from the `output:` statement of the step that writes them, never
from prose inherited from an older version. Check them against the code.

Name the step that writes each file. Dumps must come from a real file on disk.
Where no example output exists, say so and label any stand-in with its
provenance, for example:

```
(No `mash_posterior` run ships with the repository - `output/mash_posterior/` is
empty. The object below was written by the optional posterior step of `mash_fit`;
it is the same kind of object, with the same fields.)
```

## Minimal Working Example

A one-line orientation sentence saying which workflow is the entry point and
what the others do.

Do not list the workflows or methods up front - the section itself already shows them.

Headings are descriptive titles, not workflow names. Write
`### Build the MASH input from fine-mapped SuSiE results`, not `### susie_to_mash`.
Name the workflow in the body text underneath.

**Every workflow that has a command gets its own `###` section**, in the order a reader
meets them. Do not bucket the workflows the example data cannot run into a single
`### Other workflows` section - give each of them its own heading like the rest.

Each such section is exactly three cells:

1. a markdown cell holding the `###` title and one line of prose naming the workflow
   and saying what it does
2. a markdown cell holding only the `**Timing**` line
3. a code cell holding the `sos run` command

Titles are short noun phrases in sentence case with no trailing full stop:
`### Collapse per-region tables`, `### VCF export`, `### QTL and GWAS overlap`.

The command always goes in a code cell, never a fenced block inside a markdown cell,
even when the example data cannot run it. Use angle-bracketed placeholders
(`<study_name>`, `<regions.bed>`) for the inputs the repository does not ship.

Where the notebook offers several methods, use `###` for each method and `####` for
that method's steps, numbering restarting at 1 within each method:

```
### <descriptive title of method>

#### Step 1. <what it does>
#### Step 2. <what it does>

### <descriptive title of another method>

#### Step 1. <what it does>
```

Where the notebook has a single method whose commands chain - each consuming the output
of the one before it - give that method one `###` title and number its steps `#### Step
1.` to `#### Step N.` underneath, the same as any other method. Say in the body text that
they chain and must be run in order.

Where there is one command, there is no `Step` heading at all - fold the description
into the surrounding prose and show the command.

Every `sos run` command is preceded by its timing, in exactly this form, in a markdown
cell immediately above the command cell - not as a list item:

```
**Timing**: ~2 min (on toy dataset)
```

Where the runtime is not known, use `**Timing**: TBD (on toy dataset)`. Never invent a
number; timings cannot be measured in this environment.

Check whether workflows chain or are alternatives before writing the section. Read each
step's `input:` and `output:`: if two workflows write the same output file and neither
reads the other's, they are alternative entry points, not sequential stages. Say so
explicitly; a reader will otherwise assume top-to-bottom order.

Commands use the `input/` and `output/` path convention. Do not restate that as
boilerplate - the paths show it.

A second set of commands with different options becomes its own `###` section like any
other, rather than being deleted.

## Command Interface

Heading, then a code cell and its captured output. No lead-in sentence under the
heading - the code cell follows it directly.

```
## Command Interface
```

- code cell: `sos run pipeline/<notebook>.ipynb -h`
- markdown cell: the captured help text in a fenced block

Strip the `pkg_resources is deprecated` warning sos prints before the usage
text - it carries an absolute path.

## Workflow implementation

The `[step]` definitions. Do not edit these when reformatting; illustrative
command and `readRDS` cells may be corrected, workflow definitions may not.

Section dividers that sit inside this section (for example `## Posterior contrast`)
are demoted to `###` so they do not appear in the page table of contents beside the
real sections.

## Rules that apply throughout

- Every filename, flag and default is checked against the code, not copied from
  older prose.
- No absolute paths. Mask anything under a user or project root as `<path>/`.
- No first person, no `FIXME`/`TODO`, no "please let me know otherwise".
- Previews and dumps come from real files; never invent a structure.
- One `-h` output per notebook, not two.
