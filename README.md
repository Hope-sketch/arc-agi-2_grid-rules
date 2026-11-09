# ARC-AGI-2 Grid Rules

A rule-based, explainable solver for [ARC-AGI-2](https://arcprize.org/) that **describes a task before it tries to solve it**.

Each task passes through six investigation layers (grid sizes, colours, shapes, cells, rows and columns, transformations). The layers describe how the training inputs relate to the training outputs, and the tasks are sorted into named groups. A Grid Solver then always produces an answer from the group's evidence, and a Grid Verifier reports which layer was right or wrong on each task.

This is not a high-scoring solver. The contribution is the layered description of tasks and the per-layer verifier, which shows where an explainable pipeline fails and why.

- **Writeup (ARC Prize 2026, Paper Track):** [The Golden Grid Sets: Sorting ARC-AGI-2 Tasks by What a Human Sees First](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/the-golden-grid-sets)
- **Kaggle notebook:** [notebook7c6dc98add](https://www.kaggle.com/code/njabulohopemakana/notebook7c6dc98add)

## Results

Results on the 120 public evaluation tasks. Training-set results are not used for the claims below.

| Set | Result |
|---|---|
| Kaggle leaderboard | Rank 3211 |
| ARC-AGI-2 public evaluation | 0/120 |

Layer checks from the verifier (evaluation set):

| Layer | Check | Result |
|---|---|---|
| Sizes | Rules 1 to 3 predicted the output size | 86 tasks, 83 correct (96.5%) |
| Colours | Palette holds | 112 of 113 checkable tasks (99%) |
| Shapes | Every shape flag holds on the test pair | 99 of 120 tasks (82.5%) |
| Transformations | A transformation fits all training pairs | 4 of 120 tasks (3.3%) |
| Transformations | Correct on the test pair | 0 |

**What the numbers show.** The description layers are mostly reliable, but the transformation layer is the bottleneck. All four fits are local rules, and the whole-grid library (flip, rotate, tile) fitted none of the evaluation tasks.

The full per-layer table is in the writeup.

## How it works

| # | Stage | What it investigates | Folder |
|---|---|---|---|
| 1 | Grid Sizes | Input and output dimensions, sorted into five size rules | [`solver/grid_sizes`](solver/grid_sizes) |
| 2 | Grid Colours | Colours kept, added or removed, and background changes | [`solver/grid_colours`](solver/grid_colours) |
| 3 | Grid Shapes | Connected same-colour regions (4-connectivity) | [`solver/grid_shapes`](solver/grid_shapes) |
| 4 | Grid Cells | What differs cell by cell between input and output | [`solver/grid_cells`](solver/grid_cells) |
| 5 | Grid Rows and Columns | How output lines relate to input lines | [`solver/grid_rows_and_columns`](solver/grid_rows_and_columns) |
| 6 | Grid Transformations | Local rules, cropping, and a whole-grid library, chosen by the groups above | [`solver/grid_transformations`](solver/grid_transformations) |
| 7 | Grid Solver | Reads the group and evidence, always outputs a prediction | [`solver/grid_solver`](solver/grid_solver) |
| 8 | Grid Verifier | Compares predictions with answers, PASS or FAIL per layer | [`solver/grid_verifier`](solver/grid_verifier) |

Two design rules:

1. Grid size is the only hard-coded layer.
2. Colours and shapes are never hard-coded. They are discovered from each task's training pairs.

Tasks are grouped by size rule plus palette outcome, with a shape word (`MORE`, `FEWER`, `EVEN`, `VARY`) added to the group name. For example, `2|same|more` is a fixed output size, a different size from the input, more shapes, and colours kept.

## Repository layout

```
.
├── data/                      # ARC-AGI-2 tasks (not included, see below)
│   ├── training/
│   └── evaluation/
└── solver/
    └── <stage>/
        ├── <stage>.py         # the stage's script
        ├── hypothesis/        # grouping and sorting experiments
        └── results/           # CSV results and JSON "receipts"
```

- **CSV files** are per-task results, one row per task, so any single task can be inspected.
- **Receipts (`*_receipts.json`)** are the structured findings each stage passes on to the next, so later stages load earlier results instead of recomputing them.

## Running it

**Requirements:** Python 3 and `numpy` (the sizes, colours and shapes stages use only the standard library).

**Data:** the ARC-AGI-2 tasks are not redistributed here. Download them from the [ARC-AGI-2 repository](https://github.com/arcprize/ARC-AGI-2) and place the task files as:

```
data/training/<task_id>.json
data/evaluation/<task_id>.json
```

**Run the stages in order from the repository root.** Each script finds `data/` relative to its own location and writes into its own `results/` folder:

```bash
pip install numpy

python solver/grid_sizes/grid_sizes.py
python solver/grid_colours/grid_colours.py
python solver/grid_shapes/grid_shapes.py
python solver/grid_cells/grid_cells.py
python solver/grid_rows_and_columns/grid_rows_and_columns.py
python solver/grid_transformations/grid_transformations.py
python solver/grid_solver/grid_solver.py
python solver/grid_verifier/grid_verifier.py
```

The results already committed in each `results/` folder are the ones reported in the writeup.

## Limits

- The layers describe each step separately, but many tasks are only explained by combining them.
- The solver needs training examples to confirm each rule, and each new task type needs a new rule or probe.
- Tasks with no counting or size signal remain weak.
- Predicting the output size when training outputs vary is still unsolved.

## Future work

- Combine layers so that size, colour and shape evidence can decide an answer together.
- Carry the "describe, then act" idea to ARC-AGI-3.

## License

See [LICENSE](LICENSE).

## Author

Njabulo Hope Makana, developer, South Africa.
