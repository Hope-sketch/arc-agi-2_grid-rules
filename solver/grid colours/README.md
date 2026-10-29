# Grid Colours

## Overview

The Grid Colours stage investigates how the set of colours used in an input grid compares with the set of colours used in its corresponding output grid.

The solver identifies the colours present in each grid, compares the input against the output, and records whether colours were added, removed, or maintained.

This stage also investigates whether the background colour remains the same between input and output grids.

**Current status:** Investigation only. The findings are not yet used to solve tasks.

## Objectives

The Grid Colours stage aims to:

1. Identify the unique colours present in each input and output grid.
2. Compare the input and output colour sets.
3. Classify colour changes across training examples.
4. Investigate whether the same colours are added or removed consistently across training pairs.
5. Investigate whether the background colour is maintained or changes.
6. Record results to determine whether these patterns can help the solver understand grid transformations.

## Section 1: Colour Set Comparison

The solver compares the unique colours in each training input grid against the unique colours in its corresponding output grid.

Let:

- `C_input` represent the set of colours in the input grid.
- `C_output` represent the set of colours in the output grid.

The comparison identifies:

- **Added colours:** Colours present in the output but absent from the input.
- **Removed colours:** Colours present in the input but absent from the output.
- **Maintained colours:** The input and output contain the same set of colours.
- **Added and removed:** Some colours are introduced while others disappear.

### Colour Change Categories

| Category | Condition |
|---|---|
| `added` | New colours appear, and no input colours disappear. |
| `removed` | Input colours disappear, and no new colours appear. |
| `maintained` | The input and output contain exactly the same colour set. |
| `added_and_removed` | New colours appear and existing colours disappear in the same pair. |

These categories describe changes to the set of colours, not the number of cells occupied by each colour.

For example, if an input contains colours `{0, 1}` and the output contains `{0, 2}`, one colour has been added and another removed. The total number of unique colours remains the same, but the colour set has changed.

### Classifying the Entire Task

After comparing each training pair, the solver investigates whether the same category applies throughout the task.

- If every training pair is classified as `added`, the task is grouped as `added`.
- If every training pair is classified as `removed`, the task is grouped as `removed`.
- If every training pair is classified as `maintained`, the task is grouped as `maintained`.
- If the training pairs disagree, the task is grouped as `other`.
- If every training pair has both added and removed colours, the task is also grouped as `other` in the current implementation.

For tasks classified as `added` or `removed`, the solver also checks whether the same specific colour or colours are consistently added or removed across all training pairs.

## Section 2: Background Colour

The second investigation examines whether the background colour remains consistent between input and output grids.

For this investigation, the background colour is defined as the **most frequent colour in a grid**.

The solver identifies the most frequent colour in each input and output and compares them.

### Background Categories

| Category | Condition |
|---|---|
| `maintained` | The input and output have the same identified background colour. |
| `vanishes` | The identified input and output background colours differ. |
| `other` | Training pairs disagree about whether the background colour is maintained or changes. |
| `unknown` | Too few training pairs can be classified reliably. |

### Handling Ties

Sometimes two or more colours occur equally frequently in a grid. In this situation, the solver does not guess which colour is the background.

Instead, that training pair is marked `unknown` for the background investigation.

The task can still be classified using the other training pairs, provided at least half of the pairs can be decided. If fewer than half can be decided, the task remains `unknown`.

This prevents an uncertain background identification from automatically determining the result of the entire task.

## Section 3: Testing the Observed Patterns

The solver uses the training pairs to establish each task's colour and background categories.

Where evaluation outputs are available for research, the script checks whether the observed training pattern also holds for the corresponding test pair.

Possible outcomes are:

- **`holds`:** The test pair follows the observed training category.
- **`breaks`:** The test pair does not follow the observed training category.
- **`n/a`:** The training task is classified as `other`, `unknown`, or another category that cannot be checked meaningfully.

These checks help measure whether a pattern observed in the training examples also appears in the test examples.

A pattern that holds across the examples is evidence of consistency, but it does not by itself establish the complete transformation rule.

## Research Methodology

The current investigation follows these steps:

1. Load the available training and evaluation task data.
2. Identify the unique colours in each grid.
3. Compare input and output colour sets.
4. Record added and removed colours for each training pair.
5. Group each task according to the consistency of its training pairs.
6. Identify the most frequent colour in each input and output grid.
7. Compare the background colours, accounting for ties.
8. Check the observed categories against available test outputs.
9. Save the results for further investigation.

The analysis treats colour-set comparison and background-colour comparison as separate investigations because they examine different properties of a grid.

## Results

The script generates a CSV file named `grid_colour_results.csv`, containing one row per task.

The recorded information includes:

- Dataset and task identifier.
- Colour-change category and classification reason.
- Whether the same specific colours are added or removed consistently.
- Whether the colour pattern holds on the test pair.
- Background-colour category and classification reason.
- Whether the background pattern holds on the test pair.
- Per-pair colour additions and removals.
- Per-pair input and output background colours.

The script also prints summary tables and checks selected examples that were previously classified by hand.

These outputs allow the results of the investigation to be inspected, compared, and verified.

## Current Limitations

The Grid Colours stage examines colour presence and background frequency. It does not yet determine which cells changed, whether objects moved, or what transformation caused the observed colour differences.

A colour appearing or disappearing does not necessarily reveal the complete rule governing the task.

The background investigation also assumes that the most frequent colour represents the background. When the most frequent colours are tied, the current implementation abstains rather than making an unsupported choice.

The findings remain experimental until they are evaluated sufficiently and incorporated into the solver.

## Future Work

Planned work includes:

- Investigating how colour-set patterns relate to other grid properties.
- Determining whether consistent colour additions or removals can support later solving stages.
- Comparing colour changes with shape, cell, row, column, and transformation findings.
- Evaluating which patterns provide useful evidence for the solver.
- Integrating only sufficiently supported findings into the overall solving process.

## Purpose Within the Solver

Grid Colours is one component of a modular ARC-AGI-2 research solver.

Its responsibility is to investigate which colours appear in input and output grids, how those sets change, and whether the identified background colour remains consistent.

The results can provide evidence to other stages, but colour-set comparison alone is not intended to solve the complete task.
