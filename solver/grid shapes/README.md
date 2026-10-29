# Grid Shapes

## Overview

The Grid Shapes stage investigates how shapes change between input and output grids in ARC-AGI-2 tasks.

A shape is defined as a group of connected cells of the same colour. A single cell can also form a shape if it has no connected neighbour of the same colour.

The background colour is excluded from shape detection. The remaining shapes are analysed by their count and form to investigate patterns across training examples.

**Current status:** Investigation only. The findings are not yet used to solve tasks.

## Objectives

The Grid Shapes stage aims to:

1. Identify connected regions of the same colour.
2. Distinguish individual shapes using a configurable connectivity setting.
3. Identify and exclude the background colour.
4. Compare the number of shapes in input and output grids.
5. Compare shape forms independently of their colours and positions.
6. Identify forms that appear in every training input and output.
7. Check whether observed training patterns also occur in available test examples.
8. Record the findings for further investigation and potential integration into the solver.

## 1. What Counts as a Shape?

A shape is a connected group of cells that share the same colour.

The rules are:

- Cells must have the same colour to belong to the same shape.
- Cells must be connected according to the selected connectivity setting.
- A single cell can be a shape of size 1.
- Two groups of the same colour that do not touch are treated as separate shapes.
- The identified background colour is excluded from the shape collection.

Each detected shape is represented by three properties:

| Property | Description |
|---|---|
| Colour | The colour shared by the cells in the shape. |
| Size | The number of cells belonging to the shape. |
| Form | The relative arrangement of the shape's cells, independent of its position. |

## 2. Connectivity

The solver supports two connectivity settings: 4-connectivity and 8-connectivity.

### 4-Connectivity

Cells are considered connected when they touch horizontally or vertically.

Diagonal contact does not connect two cells.

### 8-Connectivity

Cells are considered connected when they touch horizontally, vertically, or diagonally.

Two same-coloured groups that are separate under 4-connectivity may form a single shape under 8-connectivity if their cells touch diagonally.

### Running the Investigation

The connectivity setting can be selected from the command line.

```bash
python solver/grid_shapes.py 8
```

To investigate 4-connectivity:

```bash
python solver/grid_shapes.py 4
```

The default setting is 8-connectivity when no argument is supplied.

Both configurations can be run independently to compare how connectivity affects the detected shapes and experimental results.

## 3. Background Colour

The background colour is defined as the most frequent colour in a grid.

The solver identifies the background separately for each input and output grid. Any connected region whose colour matches the selected background colour is excluded from the shape collection.

### Handling Ties

If multiple colours occur equally frequently, the solver selects the tied colour whose largest connected region contains the most cells.

The connectivity setting affects how connected regions are identified and therefore can affect this tie-breaking process.

Whenever multiple colours tie for the highest frequency, the grid is recorded as having a background tie. The results report these cases so that the frequency of tied background identification can be examined.

The background remains an assumption within this investigation: the most frequent colour is treated as the background, with the largest connected region used to break frequency ties.

## Rule A: Shape Count

The first rule compares the number of detected shapes in each input and output grid.

Let:

- `shapes_in` represent the number of shapes in the input.
- `shapes_out` represent the number of shapes in the output.

### Shape-Count Categories

| Category | Condition |
|---|---|
| `added` | The output contains more shapes than the input. |
| `removed` | The output contains fewer shapes than the input. |
| `same` | The input and output contain the same number of shapes. |

These categories describe changes in the total number of detected shapes. They do not establish which individual shapes were added, removed, merged, split, or transformed.

### Task Classification

The solver compares the category assigned to each training pair.

- If every training pair has the same category, the task is assigned that category.
- If the training pairs have different categories, the task is classified as `other`.

For example, if every training pair contains fewer output shapes than input shapes, the task is classified as `removed`.

This classification is based on shape counts only.

## Rule B: Shape Forms

The second rule compares the forms of detected shapes between input and output grids.

A shape's form describes its cell arrangement after its position has been removed.

To calculate the form, the solver shifts the shape's coordinates so that its minimum row and minimum column become zero. The resulting relative coordinates are sorted and used to represent the shape's layout.

### What Form Represents

The form comparison deliberately ignores two properties:

- **Position:** Two shapes can have the same form even when located in different parts of a grid.
- **Colour:** Two shapes can have the same form even when their colours differ.

This allows the investigation to focus on the arrangement of cells rather than where a shape appears or which colour it uses.

### Shape-Form Categories

| Category | Condition |
|---|---|
| `same_forms` | The input and output contain exactly the same set of forms. |
| `kept_some` | The sets differ, but at least one form appears in both. |
| `none_shared` | The input and output share no forms. |

### Task Classification

The solver classifies the task according to the shape-form category observed in each training pair.

- If every training pair has the same category, the task is assigned that category.
- If the training pairs disagree, the task is classified as `other`.

The form comparison uses sets, so repeated instances of an identical form are treated as one form for this rule. The separate shape-count investigation records the total number of shapes.

## 4. Forms Shared Across All Training Grids

In addition to comparing input and output forms pair by pair, the solver investigates which forms occur in every training input and every training output.

A form is included in this shared collection only when it appears in the form sets of all training input and output grids.

The number of shared forms is recorded for each task.

When shared forms exist, the investigation checks whether at least one of these forms appears among the detected shapes in the test input and at least one appears among the detected shapes in the test output.

This is a separate observation from Rule B. It checks for the presence of training-shared forms rather than requiring the complete input and output form sets to match.

## 5. Testing the Observed Patterns

The solver establishes its classifications using training pairs only.

Where test outputs are available in the research data, the script checks whether the training pattern also holds for the test pair.

### Test-Check Categories

| Category | Meaning |
|---|---|
| `holds` | The test pair follows the observed training category. |
| `breaks` | The test pair does not follow the observed training category. |
| `n/a` | The training classification is `other` or `none`, or a meaningful check cannot be performed. |

The shape-count and shape-form rules are checked separately.

The shared-form investigation has its own check, based on whether the identified training-shared forms appear among the test input shapes and test output shapes.

These checks help measure the consistency of the observed patterns. A pattern that holds does not, by itself, establish the complete transformation rule for the task.

## Research Methodology

The investigation follows these steps:

1. Load task data from the available training and evaluation datasets.
2. Detect connected, same-coloured regions using the selected connectivity.
3. Identify the background colour and exclude its regions.
4. Compare the number of shapes in each input and output.
5. Compare the sets of shape forms in each input and output.
6. Classify tasks according to the consistency of their training pairs.
7. Identify forms shared by every training input and output.
8. Check the observations against available test outputs.
9. Record the results for further analysis.

The script can be run separately with both connectivity settings to investigate how the definition of connectedness affects the findings.

## Results

The investigation produces terminal summary tables and a CSV file for each connectivity setting.

The output files are:

- `results/grid_shapes_results_4.csv`
- `results/grid_shapes_results_8.csv`

Each CSV contains one row per task, with the dataset name and task identifier recorded alongside the results.

The recorded information includes:

- Shape-count category, classification reason, and test check.
- Shape-form category, classification reason, and test check.
- Number of forms shared across all training inputs and outputs.
- Shared-form test check.
- Number of grids with tied background colours across the training pairs.
- Per-pair input and output shape counts.

The terminal tables summarise the task classifications, test-check outcomes, shared-form findings, and background ties.

Comparing the two CSV files makes it possible to investigate whether 4-connectivity and 8-connectivity produce different results.

## Current Limitations

The Grid Shapes stage investigates shape counts and forms, but it does not yet identify the complete transformation responsible for a change.

The investigation has several important limitations:

- The selected background colour may not represent the true background in every task.
- Shape detection depends on the selected connectivity setting.
- Shape-count changes do not identify which individual shapes were altered.
- Shape-form comparison ignores colour, position, and the number of repeated instances of the same form.
- Shared-form detection records recurring forms but does not establish why they recur.
- A pattern observed in the training examples may not hold for the test pair.

These limitations are part of the investigation and should be considered when deciding which findings can be incorporated into later solving stages.

## Future Work

Planned work includes:

- Comparing 4-connectivity and 8-connectivity across the dataset.
- Investigating tasks where shape-count patterns are inconsistent.
- Examining how colour, size, and form interact.
- Investigating whether shared forms provide useful evidence about task transformations.
- Comparing shape findings with grid sizes, colours, cells, rows, columns, and transformations.
- Determining which findings are sufficiently supported for integration into the solver.
- Using verification to identify and analyse incorrect predictions.

## Purpose Within the Solver

Grid Shapes is one component of the modular ARC-AGI-2 research solver.

Its responsibility is to investigate connected shapes, shape counts, and shape forms. The findings may provide evidence to other stages, but shape analysis alone is not intended to solve the complete task.

The stage remains an investigation module until its findings are sufficiently evaluated and integrated into the solving pipeline.
