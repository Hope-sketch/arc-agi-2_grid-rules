# Grid Sizes

## Overview

The Grid Sizes stage investigates the relationship between the dimensions of input and output grids in ARC-AGI-2 tasks.

A grid's size is measured using its **height (h)** and **width (w)**:

- **Height (h):** The number of rows in a grid.
- **Width (w):** The number of columns in a grid.
- **Grid size:** Represented as `(h, w)`.

The purpose of this stage is to identify patterns in the training examples and use those patterns to predict the dimensions of a test output when sufficient evidence is available.

When the observed patterns do not support an exact size prediction, the solver leaves the problem unresolved for further investigation by other stages.

## Objectives

The Grid Sizes stage aims to:

1. Measure the height and width of each training input and output grid.
2. Compare input and output dimensions across training examples.
3. Classify each task according to five grid-size rules.
4. Predict test output dimensions when a rule supports a prediction.
5. Investigate the direction of grid-size changes when an exact prediction is not yet possible.
6. Record the findings for further analysis and integration into the solver.

## Grid-Size Rules

The solver classifies each task using its training examples. The test input is not used to decide which rule a task belongs to.

### Step 1: Input = Output

**Condition:** Every training pair has the same input and output dimensions.

For each training pair:

`Input size = Output size`

**Hypothesis:** If every training pair preserves its dimensions, the test output is likely to have the same dimensions as the test input.

**Solver action:** Predict the test output dimensions using the test input dimensions.

### Step 2: Input ≠ Output, Input = Input, Output = Output

**Conditions:**

- Each training input differs in size from its corresponding output.
- All training input grids have the same dimensions.
- All training output grids have the same dimensions.

**Hypothesis:** The test output is likely to have the same dimensions as the training outputs.

**Solver action:** Predict the test output dimensions using the dimensions consistently observed in the training outputs.

### Step 3: Input ≠ Output, Input ≠ Input, Output = Output

**Conditions:**

- Each training input differs in size from its corresponding output.
- Training input dimensions vary across the examples.
- Training output dimensions remain constant.

**Hypothesis:** Even when the input dimensions vary, the test output is likely to have the same dimensions as the training outputs.

**Solver action:** Predict the test output dimensions using the consistent training output dimensions.

### Step 4: Input ≠ Output, Input = Input, Output ≠ Output

**Conditions:**

- Each training input differs in size from its corresponding output.
- Training input dimensions remain constant.
- Training output dimensions vary across the examples.

**Current solver action:** The solver will not attempt to predict the exact output dimensions at this stage.

**Further investigation:** Compare the input and output heights and widths separately to determine whether consistent size-change patterns exist.

### Step 5: Input ≠ Output, Input ≠ Input, Output ≠ Output

**Conditions:**

- Training input dimensions vary across the examples.
- Training output dimensions also vary across the examples.
- No consistent output size has been established by the preceding rules.

**Current solver action:** The solver will not attempt to predict the exact output dimensions at this stage.

**Further investigation:** Compare the height and width changes of the training pairs to investigate whether the direction of change provides useful information.

The current implementation also places tasks with a mixture of size-preserving and size-changing training pairs into the unresolved category.

## Height and Width Direction Analysis

For tasks that cannot be resolved through an exact size prediction, the investigation compares the input and output dimensions independently.

Let:

- `input_h` represent the input height.
- `input_w` represent the input width.
- `output_h` represent the output height.
- `output_w` represent the output width.

The analysis compares these values to classify the direction of the size change.

| Direction label | Condition |
|---|---|
| `total_drop` | The output grid is `1 × 1`. |
| `absolutely_smaller` | The output height and width are both smaller than the input dimensions. |
| `absolutely_bigger` | The output height and width are both bigger than the input dimensions. |
| `mixed_1` | The output height increases while the output width decreases. |
| `mixed_2` | The output height decreases while the output width increases. |
| `mixed_3` | The output height stays the same while the width changes. |
| `mixed_4` | The output width stays the same while the height changes. |
| `same` | The input and output dimensions are identical. |

The `total_drop` condition takes priority over the other direction labels. The `same` label identifies size-preserving pairs.

### Comparing Training and Test Patterns

The solver records the direction of the size change for each training pair and determines whether the training examples share a single direction or show varying directions.

When the training examples have one consistent direction, the investigation checks whether the test pair follows that same direction.

This comparison is intended to establish whether direction patterns can help the solver reason about grid dimensions in tasks where a direct size prediction is not yet supported.

A matching direction is evidence of a repeated pattern, not proof that the complete task has been solved.

## Research Methodology

The investigation follows these steps:

1. Load task data from the available training and evaluation datasets.
2. Calculate the dimensions of input and output grids.
3. Classify each task using the training examples.
4. Apply the supported output-size prediction rule, if one is available.
5. Compare predicted dimensions with the available output dimensions for evaluation.
6. Investigate height and width direction patterns for the designated rules.
7. Record correct predictions, incorrect predictions, unresolved cases, and direction comparisons.

The task classification is based on training grids. Evaluation outputs, when available in the research data, are used to check the predictions rather than to decide the classification rule.

## Results

The investigation generates a CSV file containing task-level findings.

The results include:

- Task identifier and assigned rule.
- Reason for the classification.
- Number of training pairs.
- Predicted and actual output dimensions, when available.
- Prediction outcome.
- Training direction labels and direction summary.
- Test direction labels and direction comparison.

These results make it possible to inspect individual tasks, identify failures, and determine which cases require additional investigation.

## Current Limitations

The Grid Sizes stage does not attempt to solve every task.

When the training examples do not support a reliable exact-size prediction, the task remains unresolved at this stage. Other investigation stages may contribute additional information.

The proposed rules are hypotheses about recurring patterns. Their usefulness must be assessed against observed results, including incorrect predictions and cases where a rule is insufficient.

Direction agreement alone does not establish that a test output has been predicted correctly.

## Future Work

Planned improvements include:

- Further investigation of tasks classified under Steps 4 and 5.
- Testing additional relationships between input and output heights and widths.
- Identifying whether direction patterns can support more precise size predictions.
- Integrating supported findings into the overall solver.
- Using a verification stage to assess predictions and identify cases requiring further investigation.

## Purpose Within the Solver

Grid Sizes is one component of a modular ARC-AGI-2 solver. Its responsibility is to investigate grid dimensions, identify recurring size relationships, and provide evidence that can support later solving stages.

It does not independently determine the complete transformation of a grid. Its findings are intended to work alongside the other investigation stages.
