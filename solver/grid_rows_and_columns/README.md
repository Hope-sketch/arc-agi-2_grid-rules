# Grid Rows and Columns

## Overview

Stage 5 describes how the **lines** of the output relate to the lines of the input: which rows and columns stayed, moved, were added, were removed, or were reordered.

The grid is read as a plane:

- **X axis = columns**, positive X is right.
- **Y axis = rows**, positive Y is down.

A move of `(dy, dx) = (+2, -1)` means two rows down and one column left.

Blank lines (only background) are ignored. A line pair is **similar** when 60% or more of its non-background cells match. Rows are comparable when input and output have the same width, and columns when they have the same height.

The background colour comes from `find_background()` in [`grid_colours`](../grid_colours), so that stage's script must be present.

## Events

Each event is true or false per training pair.

| Group | Events |
|---|---|
| Colour moves | `COLOUR_MOVED_X`, `COLOUR_MOVED_Y`, `COLOUR_MOVED_POSITIVE_X` / `_NEGATIVE_X`, `COLOUR_MOVED_POSITIVE_Y` / `_NEGATIVE_Y`, `COLOURS_SHIFT_TOGETHER` |
| Similarity | `ROW_SIMILAR`, `ROW_NOT_SIMILAR`, `COLUMN_SIMILAR`, `COLUMN_NOT_SIMILAR`, `ALL_ROWS_SIMILAR`, `ALL_COLUMNS_SIMILAR` |
| Added or removed | `ROW_ADDED`, `ROW_REMOVED`, `COLUMN_ADDED`, `COLUMN_REMOVED` |
| Moved lines | `ROW_MOVED_POSITIVE_Y` / `_NEGATIVE_Y`, `COLUMN_MOVED_POSITIVE_X` / `_NEGATIVE_X` |
| Reordered | `ROWS_REORDERED`, `COLUMNS_REORDERED` |
| Output vs output | `OUTPUT_ROW_SIMILAR`, `OUTPUT_ROW_NOT_SIMILAR`, `OUTPUT_COLUMN_SIMILAR`, `OUTPUT_COLUMN_NOT_SIMILAR` |

The receipt keeps `flags` (events true in every training pair) and `sometimes` (true in some). It also holds `consistent_shift = [dy, dx]` when every training pair moves its colours by the same shift, otherwise `null`.

## Notes

- Colour moves are only measured on same-size grids.
- A colour with a single cell is ambiguous (moved, or removed and added elsewhere) and is reported as moved.
- The receipt never contains the test output. The public test output is used only for the "holds on test" checks.

## Results (public evaluation set)

| Check | Result |
|---|---|
| Flag checks hold on the test pair | 87.6% |
| Tasks with at least one failing flag | 25.8% |

## Files

| File | Purpose |
|---|---|
| `grid_rows_and_columns.py` | The stage script |
| `hypothesis/grid_sorter.py` | Grouping experiment |
| `results/grid_rows_and_columns_results.csv` | One row per task |
| `results/grid_rows_columns_details.csv` | Pair-level detail |
| `results/grid_rows_and_columns_receipts.json` | What later stages load |

## Running

```bash
python solver/grid_rows_and_columns/grid_rows_and_columns.py
```

```python
from grid_rows_and_columns import load_rows_columns_receipts, check_rows_columns_receipts
```

## Limitations

- Line-level descriptions miss changes that cross lines, such as diagonal or object-level moves.
- Moves are not measured when input and output differ in size.
- About a quarter of evaluation tasks have a flag that holds in training but fails on the test pair.
