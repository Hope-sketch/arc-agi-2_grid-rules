# Grid Transformations

## Overview

Stage 6 does two things for every task. It feeds the Grid Solver.

**Part 1: Collect.** Gathers the receipts of stages 1 to 5 (sizes, colours, shapes, cells, rows and columns) into one transformation receipt:

| Field | Meaning |
|---|---|
| `steps` | The flags of every stage |
| `flags` / `sometimes` | Flags true in every training pair / in some pairs |
| `combined` | Flags that only appear when stages are read together |
| `group`, `story` | A short label for the kind of task (size, palette, shape count) |

Combined flags: `COMBINED_NEAR_COPY`, `COMBINED_RECOLOUR_ONLY`, `COMBINED_MOVE_ONLY`, `COMBINED_ADD_COLOURED_SHAPES`, `COMBINED_REMOVE_SHAPES_AND_COLOURS`, `COMBINED_CROP`, `COMBINED_FILL_HOLES`, `COMBINED_RESIZE_SHAPES`, `COMBINED_ROTATE_OR_FLIP`, `COMBINED_REARRANGE_LINES`.

**Part 2: Learn more.** Tries a library of transformations on the training pairs. A candidate **fits** only if it reproduces **every** training output (after one learned colour map).

| Kind | What it covers |
|---|---|
| `whole_grid` | Flips, rotations, transpose, mirrors, tiling, scaling up and down, halves, crop to the non-background, crop to the box of a colour, shift by up to 3 cells, plain recolouring |
| `local_rule` | Same-size tasks: what a cell becomes, learned from its neighbourhood |

Candidates are listed simplest first, and `best` is the first one. If nothing fits, the three closest are kept as `partial`, with how many training pairs each reproduces exactly. `size_ratio` is stored when output size over input size is a constant ratio in every pair.

## Interface for the Grid Solver

```python
specs, partial = find_transformations(train_pairs)            # [(input, output), ...]
prediction, coverage = apply_transformation(specs[0], test_input)   # if specs is not empty
```

The receipt JSON holds a compact description of each candidate. The learned local rule itself lives in memory and is rebuilt by `find_transformations`.

## No test leakage

The receipt never contains the test output. The public test output is used only for the "reproduces test" checks in the CSV and tables. These are checks, not scores.

## Results (public evaluation set)

| Check | Result |
|---|---|
| A transformation fits all training pairs | 4 of 120 tasks (3.3%) |
| Kind of the 4 fits | All `local_rule`, none `whole_grid` |
| Correct on the test pair | 0 |

For comparison, on the training set a transformation fits 112 of 1000 tasks (11.2%): 83 `local_rule` and 29 `whole_grid`.

This is the bottleneck of the pipeline. The description layers are mostly reliable, but they rarely lead to a transformation that fits.

## Files

| File | Purpose |
|---|---|
| `grid_transformations.py` | The stage script |
| `whole_grid_library.py` | Whole-grid library file (`grid_transformations.py` does not need it) |
| `results/grid_transformations_results.csv` | One row per task |
| `results/grid_transformations_receipts.json` | What the Grid Solver loads |

## Running

Run stages 1 to 5 first. This stage reads the receipts they write.

```bash
python solver/grid_transformations/grid_transformations.py
```

## Limitations

- The library is fixed. A task whose rule is not in it cannot fit.
- A candidate must reproduce every training output, so a rule that is almost right is not used. It is kept only as `partial`.
- The local rule is learned from a cell's neighbourhood, so it cannot express rules that depend on object-level structure.
