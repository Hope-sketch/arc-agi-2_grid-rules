# Grid Solver

## Overview

Stage 7 reads the receipt of every task from [`grid_transformations`](../grid_transformations), which already holds the flags of stages 1 to 5, and acts on what the receipt states.

Every test input gets **two attempts** (`attempt_1`, `attempt_2`). The solver never leaves a task empty, even when its confidence is low.

## How it solves

**1. Strategies are switched on by the receipt.**

| Strategy | Switched on when |
|---|---|
| `transform:<name>` | Candidates fit every training pair (best first) |
| `partial:<name>` | Nothing fits, so the closest library functions are used |
| `shift_consistent` | Every pair moves its colours by the same shift |
| `fill_enclosed` | The receipt says holes are filled in |
| `colour_map_input` | Same-size tasks: the colour change learned from training |
| `soft_local_<kind>` | Same-size tasks: neighbourhood rule by majority vote |
| `majority_output` | All training outputs share a size: cell-by-cell majority |
| `resize_input` | The sizes stage predicted an output size |
| `copy_input` | Always on, the guaranteed answer |

**2. Every strategy is judged on the training pairs only.** Strategies that learn something are judged leave-one-out (learn on the other pairs, predict the one held back), so a rule that merely memorises scores badly.

**3. Score of a prediction:**

| Part | Weight |
|---|---|
| Pairs reproduced exactly | 0.55 |
| Share of cells right | 0.35 |
| Size check (1 = predicted size, 0 = different size, 0.5 = no size predicted) | 0.10 |
| Each other strategy that proposed the same grid | +0.03 (capped at 0.06) |

**4. Attempts.** `attempt_1` is the best scoring grid and `attempt_2` is the best scoring **different** grid. If only one grid exists, a fallback (majority output, blank grid, flipped input) fills in.

The test output is never looked at while solving. The public test outputs only fill the "correct" columns and tables, as a check.

## Outputs

| File | Purpose |
|---|---|
| `results/grid_solver_submission_<dataset>.json` | `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one per test input (the Kaggle submission layout) |
| `results/grid_solver_receipts.json` | Which strategy made which attempt, with its training evidence (no grids), for the Grid Verifier |
| `results/grid_solver_results.csv` | One row per task |
| `results/grid_solver_results_evaluation.csv` | Evaluation-only results |
| `results/grid_solver_predictions_evaluation.json` | Evaluation predictions |

## Results

| Set | Result |
|---|---|
| ARC-AGI-2 public evaluation | 0/120 |

The training set was used while writing the rules, so its results are not used for any claim.

## Running

Run stages 1 to 6 first.

```bash
python solver/grid_solver/grid_solver.py
```

## Limitations

- Most answers come from generic strategies (`colour_map_input`, `copy_input`, `resize_input`), not from a transformation that fits.
- The score is based on training evidence, so a task with few or unrepresentative examples gets an unreliable score.
- Each new task type needs a new strategy or probe.
