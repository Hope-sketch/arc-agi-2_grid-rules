# Grid Cells

## Overview

Stage 4 asks: **how alike are the input and the output, cell by cell?**

Every training input is compared with its output (and the training outputs with each other). Height and width are the first clue:

- **Same size:** cells are compared position by position.
- **Different size:** the smaller grid is slid over the bigger one and the best fit is used, because the output can be a piece of the input, or the reverse.

The background colour comes from `find_background()` in [`grid_colours`](../grid_colours), so that stage's script must be present.

## Similarity

Each training pair gets a similarity word, based on the share of cells that are the same. The share uses the bigger grid's area, so a small piece of a big grid is not called "similar". The piece itself is reported separately.

| Word | Share of matching cells |
|---|---|
| `very_similar` | 60% or more |
| `somewhat_similar` | 50% to 59.9% |
| `not_similar` | under 50% |

## Flags

A flag is recorded in the receipt when it is true in **every** training pair. A flag that is true in only some pairs is listed under `sometimes`.

| Group | Flags |
|---|---|
| Similarity | `CELLS_VERY_SIMILAR`, `CELLS_SOMEWHAT_SIMILAR`, `CELLS_NOT_SIMILAR`, `CELLS_MIXED` |
| Size | `SAME_SIZE`, `OUTPUT_BIGGER`, `OUTPUT_SMALLER`, `MIXED_SIZE`, `SIZE_VARIES` |
| Events | `GRID_IDENTICAL`, `CELLS_ADDED`, `CELLS_ERASED`, `CELLS_RECOLOURED`, `OUTPUT_IS_PIECE_OF_INPUT`, `INPUT_IS_PIECE_OF_OUTPUT`, `SIMILAR_AFTER_FLIP_OR_ROTATION`, `SIMILARITY_FROM_BACKGROUND` |

`SIMILARITY_FROM_BACKGROUND` means the grids are 60% or more alike, but under 50% of the non-background cells match, so the background is doing the work.

The receipt also holds output-versus-output similarity when all training outputs share a size.

## No test leakage

The receipt never contains the test output. The public test output is used only for the "holds on test" checks in the CSV and the terminal tables.

## Results (public evaluation set)

| Check | Result |
|---|---|
| Similarity pattern holds on the test pair | 97.3% (110 checkable tasks) |
| Size pattern holds on the test pair | 98.2% (112 checkable tasks) |
| Event flags hold on the test pair | 96.3% |

## Files

| File | Purpose |
|---|---|
| `grid_cells.py` | The stage script |
| `results/grid_cells_results.csv` | One row per task |
| `results/grid_cells_details.csv` | Pair-level detail |
| `results/grid_cells_receipts.json` | What later stages load |

## Running

```bash
python solver/grid_cells/grid_cells.py
```

Later stages import the loaders from this stage:

```python
from grid_cells import load_cells_receipts, check_cells_receipts
```

## Limitations

- Similarity measures overlap, not meaning. Two grids can be 90% alike and differ in the one cell that matters.
- Best-fit sliding is only a clue for different-size pairs. It does not explain how the output was made.
- Flags that hold in every pair can still fail on the test pair, which is what the "holds on test" checks measure.
