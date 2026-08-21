# Vignette format

A vignette is a concept-first worked analysis. It explains why a method is used, how the analysis is designed, what the results mean, and where interpretation can fail. It should not duplicate the complete command reference from a module notebook or the route table from a mini-protocol.

## Required structure

1. `# Title` followed by one sentence stating the analytical task.
2. `## Learning goals` with two to four concrete outcomes.
3. `## Background and method` explaining the model, assumptions, and method choice, with citations for external claims.
4. `## Worked example`, beginning with the analysis unit and real input files, followed by `###` subsections that introduce each analytical action before executable code.
5. `## Results and interpretation`, with each result presented as one unit: output layout, focused inspection code, then its statistical or biological interpretation.
6. `## Limitations and common pitfalls` stating design constraints and failure modes.
7. `## Next steps` linking to the relevant mini-protocol and module notebook.

## Writing and content rules

- Keep one scientific question throughout the vignette.
- Explain method choice before showing commands.
- Keep example inputs inside the worked example rather than creating a separate top-level Example data section.
- Number subsections only when every item is part of one required sequential chain. Label alternatives as optional routes instead.
- Use real repository-relative example paths; do not expose project-specific absolute paths.
- Preserve exact method names and distinguish association, prediction, fine-mapping, colocalization, and causality.
- Show effect estimates, uncertainty, predictive performance, posterior probabilities, or credible sets as appropriate.
- Keep interpretation directly below the result it explains.
- Do not infer causality from association alone.
- Use figures only when they support an interpretive point; give every figure a caption and readable dimensions.
- Move exhaustive flags and workflow implementation details to the module notebook.
- Link workflow headings to the module page.
- End with the next analysis decision, not a generic conclusion.
