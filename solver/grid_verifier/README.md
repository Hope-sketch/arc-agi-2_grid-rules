# Grid Verifier

## Overview

Stage 8 checks the Grid Solver's two attempts against the public test outputs and shows **where** it fails. It never changes the solver or its predictions.

## What it measures

Per test input and per attempt:

| Measure | Meaning |
|---|---|
| `size` | Right shape (height by width) |
| `colours` | Same set of colours as the real output, and which are missing or extra |
| `shapes` | Same shapes (colour and form, position ignored) |
| `cells` | Exact match, cell accuracy (same-size grids), overlap accuracy |
| `lines` | Share of rows and columns that are exactly right |
| `changes` | Same-size tasks: of the cells that had to change, how many were changed to the right colour (recall), and how many of the changed cells were right (precision) |

## Diagnosis

The diagnosis is the **first** stage that fails, in this order:

`WRONG_SIZE` → `WRONG_COLOURS` → `WRONG_SHAPES` → `WRONG_CELLS` → `EXACT`

For example, `WRONG_COLOURS` means the size was right but the colours were wrong. The task diagnosis uses the better of the two attempts.

The verifier also checks the solver's own claims: whether the sizes stage's prediction is right, and whether a higher solver score means a better answer (calibration).

## Results (public evaluation set)

| Check | Result |
|---|---|
| Tasks solved | 0 of 120 |
| Sizes: predicted size right | 83 of 86 predictions |
| Most common diagnosis | `WRONG_SHAPES` (at least 64 of 120 tasks) |
| Second most common | `WRONG_SIZE` (at least 32 of 120 tasks) |

The diagnosis counts are lower bounds, because tasks with several test inputs can carry more than one diagnosis.

## Files

| File | Purpose |
|---|---|
| `grid_verifier.py` | The stage script |
| `results/grid_verifier_results.csv` | One row per task |
| `results/grid_verifier_receipts.json` | Per test input, no grids |

## Running

Run stages 1 to 7 first.

```bash
python solver/grid_verifier/grid_verifier.py
```

## Limitations

- It compares against the public test outputs, so it is a development diagnostic, not a source of hidden-test performance.
- "First failing stage" blames the earliest stage, even when a later stage would also have failed.
