"""
Step 8: Grid verifier

Checks the Grid Solver's two attempts against the public test outputs and shows
where it fails. It never changes the solver or its predictions.

Per test input and per attempt it measures
  size      right shape (height x width)
  colours   the same set of colours as the real output (and which are missing / extra)
  shapes    the same shapes (colour + form, position ignored)
  cells     exact match, cell accuracy (same-size grids), overlap accuracy
  lines     share of rows / columns that are exactly right
  changes   (same-size tasks) of the cells that had to change, how many the solver
            changed to the right colour (recall), and how many of the cells it
            changed were right (precision)

Diagnosis = the FIRST stage that fails, in this order:
  WRONG_SIZE -> WRONG_COLOURS -> WRONG_SHAPES -> WRONG_CELLS -> EXACT
e.g. WRONG_COLOURS means the size was right but the colours were wrong.
The task diagnosis uses the better of the two attempts.

It also checks the solver's own claims: is the sizes step's prediction right, and does
a higher solver score mean a better answer (calibration).

Usage
  python solver/grid_verifier/grid_verifier.py
  (run steps 1-7 first, this step reads their files)

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_verifier/results/grid_verifier_results.csv    (one row per task)
  * solver/grid_verifier/results/grid_verifier_receipts.json (per test input, no grids)
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
SOLVER_RESULTS = ROOT / "solver" / "grid_solver" / "results"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_verifier_results.csv"
OUT_JSON = RESULTS_DIR / "grid_verifier_receipts.json"
DATASETS = ("training", "evaluation")

for step in ("grid_colours", "grid_shapes", "grid_transformations", "grid_solver"):
    sys.path.insert(0, str(ROOT / "solver" / step))
try:
    from grid_colours import find_background
    from grid_shapes import extract_shapes
    from grid_transformations import load_transformations_receipts, check_transformations_receipts
    from grid_solver import load_solver_receipts, check_solver_receipts
except (ImportError, SystemExit) as error:
    raise SystemExit(f"Could not import a step script ({error}).\nSteps 1-7 must be in their solver/ folders.")

STAGES = ["WRONG_SIZE", "WRONG_COLOURS", "WRONG_SHAPES", "WRONG_CELLS", "EXACT"]
STAGE_RANK = {name: i for i, name in enumerate(STAGES)}
CALIBRATION_BINS = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


def load_submission(dataset):
    path = SOLVER_RESULTS / f"grid_solver_submission_{dataset}.json"
    if not path.exists():
        raise SystemExit(f"Solver submission not found: {path}\nRun solver/grid_solver/grid_solver.py first.")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def to_np(grid):
    return np.array(grid, dtype=int)


def family(name):
    return name.split(":")[0] if ":" in name else name


# ----------------------------------------------------------------------
# Measuring one attempt
# ----------------------------------------------------------------------
def shapes_of(grid):
    bg = find_background(grid.tolist())
    return Counter((s.colour, s.form) for s in extract_shapes(grid.tolist(), bg)), bg


def measure_attempt(pred, actual, test_input):
    shape_ok = pred.shape == actual.shape
    exact = shape_ok and bool(np.array_equal(pred, actual))
    cell_accuracy = float((pred == actual).mean()) if shape_ok else 0.0
    h, w = min(pred.shape[0], actual.shape[0]), min(pred.shape[1], actual.shape[1])
    overlap = float((pred[:h, :w] == actual[:h, :w]).sum()) / max(pred.size, actual.size)

    palette_pred, palette_actual = set(np.unique(pred).tolist()), set(np.unique(actual).tolist())
    colours_ok = palette_pred == palette_actual
    shapes_pred, bg_pred = shapes_of(pred)
    shapes_actual, bg_actual = shapes_of(actual)
    shapes_ok = shapes_pred == shapes_actual

    result = {"shape_ok": shape_ok, "exact": exact,
              "cell_accuracy": round(cell_accuracy, 4), "overlap_accuracy": round(overlap, 4),
              "colours_ok": colours_ok,
              "missing_colours": sorted(palette_actual - palette_pred),
              "extra_colours": sorted(palette_pred - palette_actual),
              "background_ok": bg_pred == bg_actual,
              "shape_count_ok": sum(shapes_pred.values()) == sum(shapes_actual.values()),
              "shapes_ok": shapes_ok,
              "rows_right": None, "columns_right": None, "changes": None}

    if shape_ok:
        equal = pred == actual
        result["rows_right"] = round(float(equal.all(axis=1).mean()), 4)
        result["columns_right"] = round(float(equal.all(axis=0).mean()), 4)
        if test_input.shape == actual.shape:
            needed, made = test_input != actual, test_input != pred
            result["changes"] = {"needed": int(needed.sum()), "made": int(made.sum()),
                                 "correct": int((made & equal).sum())}

    if not shape_ok:
        diagnosis = "WRONG_SIZE"
    elif not colours_ok:
        diagnosis = "WRONG_COLOURS"
    elif not shapes_ok:
        diagnosis = "WRONG_SHAPES"
    elif not exact:
        diagnosis = "WRONG_CELLS"
    else:
        diagnosis = "EXACT"
    result["diagnosis"] = diagnosis
    return result


def better(a, b):
    """True if attempt a is at least as good as attempt b."""
    return (STAGE_RANK[a["diagnosis"]], a["cell_accuracy"]) >= (STAGE_RANK[b["diagnosis"]], b["cell_accuracy"])


# ----------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------
def verify_task(task, submission, solver, transformation):
    records = []
    for k, t in enumerate(task["test"]):
        if "output" not in t:
            return None
        actual, test_input = to_np(t["output"]), to_np(t["input"])
        attempts = [to_np(submission[k]["attempt_1"]), to_np(submission[k]["attempt_2"])]
        measured = [measure_attempt(a, actual, test_input) for a in attempts]
        claims = solver["test"][k]
        for m, claim in zip(measured, claims["attempts"]):
            m["strategy"], m["solver_score"] = claim["strategy"], claim["score"]

        predicted_size = claims["predicted_size"]
        size_receipt = ("none" if predicted_size is None
                        else "yes" if tuple(predicted_size) == actual.shape else "no")
        records.append({"attempts": measured, "best": 0 if better(measured[0], measured[1]) else 1,
                        "size_receipt": size_receipt, "predicted_size": predicted_size,
                        "actual_size": list(actual.shape), "input_size": list(test_input.shape)})
    return records


# ----------------------------------------------------------------------
# For the next step (improving the solver)
# ----------------------------------------------------------------------
def load_verifier_receipts(path=OUT_JSON):
    """{dataset: {task_id: [per test input record]}}."""
    if not Path(path).exists():
        raise SystemExit(f"Grid verifier receipts not found: {path}\nRun solver/grid_verifier/grid_verifier.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def csv_row(dataset, task_id, transformation, records):
    join = lambda values: ";".join(str(v) for v in values)
    a1 = [r["attempts"][0] for r in records]
    a2 = [r["attempts"][1] for r in records]
    best = [r["attempts"][r["best"]] for r in records]
    solved = [b["exact"] for b in best]

    def change(key):
        return join(a["changes"][key] if a["changes"] else "" for a in a1)

    def recall_precision(which):
        values = []
        for a in a1:
            c = a["changes"]
            if not c:
                values.append("")
            elif which == "recall":
                values.append(round(c["correct"] / c["needed"], 3) if c["needed"] else "")
            else:
                values.append(round(c["correct"] / c["made"], 3) if c["made"] else "")
        return join(values)

    return {
        "dataset": dataset, "task_id": task_id, "group": transformation["group"],
        "story": transformation["story"], "n_test_inputs": len(records),
        "size_category": transformation["steps"]["sizes"]["category"],
        "size_receipt_predicted": join("none" if r["predicted_size"] is None else f"{r['predicted_size'][0]}x{r['predicted_size'][1]}"
                                       for r in records),
        "size_receipt_right": join(r["size_receipt"] for r in records),
        "actual_size": join(f"{r['actual_size'][0]}x{r['actual_size'][1]}" for r in records),
        "attempt_1_strategy": join(a["strategy"] for a in a1),
        "attempt_1_solver_score": join(a["solver_score"] for a in a1),
        "attempt_1_diagnosis": join(a["diagnosis"] for a in a1),
        "attempt_2_diagnosis": join(a["diagnosis"] for a in a2),
        "best_diagnosis": join(b["diagnosis"] for b in best),
        "attempt_1_cell_accuracy": join(a["cell_accuracy"] for a in a1),
        "attempt_2_cell_accuracy": join(a["cell_accuracy"] for a in a2),
        "best_cell_accuracy": round(mean([b["cell_accuracy"] for b in best]), 4),
        "attempt_1_overlap_accuracy": join(a["overlap_accuracy"] for a in a1),
        "attempt_1_size_ok": join("yes" if a["shape_ok"] else "no" for a in a1),
        "attempt_1_colours_ok": join("yes" if a["colours_ok"] else "no" for a in a1),
        "attempt_1_shapes_ok": join("yes" if a["shapes_ok"] else "no" for a in a1),
        "attempt_1_missing_colours": join(",".join(map(str, a["missing_colours"])) or "-" for a in a1),
        "attempt_1_extra_colours": join(",".join(map(str, a["extra_colours"])) or "-" for a in a1),
        "attempt_1_background_ok": join("yes" if a["background_ok"] else "no" for a in a1),
        "attempt_1_rows_right": join("" if a["rows_right"] is None else a["rows_right"] for a in a1),
        "attempt_1_columns_right": join("" if a["columns_right"] is None else a["columns_right"] for a in a1),
        "changes_needed": change("needed"), "changes_made": change("made"),
        "changes_correct": change("correct"),
        "change_recall": recall_precision("recall"), "change_precision": recall_precision("precision"),
        "solved": "yes" if all(solved) else "partial" if any(solved) else "no",
        "task_score": round(sum(solved) / len(solved), 3),
    }


def print_summary(dataset, all_records, rows):
    flat = [(task_id, r) for task_id, records in all_records.items() for r in records]
    n_inputs, n_tasks = len(flat), len(rows)
    a1 = [r["attempts"][0] for _, r in flat]
    best = [r["attempts"][r["best"]] for _, r in flat]
    print(f"\n=== {dataset}: grid verifier ({n_tasks} tasks, {n_inputs} test inputs) ===")

    score = sum(row["task_score"] for row in rows) / n_tasks
    print(f"Score (either attempt)              : {pct(score)}   "
          f"({sum(row['solved'] == 'yes' for row in rows)} tasks fully solved)")
    print(f"Cell accuracy, attempt 1 / best     : {pct(mean([a['cell_accuracy'] for a in a1]))} / "
          f"{pct(mean([b['cell_accuracy'] for b in best]))}   (a wrong size counts as 0)")
    right_size = [a for a in a1 if a["shape_ok"]]
    print(f"Cell accuracy when the size is right: {pct(mean([a['cell_accuracy'] for a in right_size]))}   "
          f"({len(right_size)}/{n_inputs} test inputs have the right size)")
    only_second = sum(1 for _, r in flat if r["attempts"][1]["exact"] and not r["attempts"][0]["exact"])
    print(f"Solved only by attempt 2            : {only_second} test inputs")

    print("\nWhere attempt 1 fails (first failing stage)")
    header = f"{'Diagnosis':<16}{'Test inputs':>13}{'Share':>9}{'Cell acc.':>11}"
    print(header)
    print("-" * len(header))
    counts = Counter(a["diagnosis"] for a in a1)
    for name in STAGES:
        group = [a for a in a1 if a["diagnosis"] == name]
        print(f"{name:<16}{counts[name]:>13}{pct(counts[name] / n_inputs):>9}"
              f"{pct(mean([a['cell_accuracy'] for a in group])):>11}")

    print("\nStage pass rates, attempt 1 (each checked on its own)")
    for label, key in (("size right", "shape_ok"), ("colours right", "colours_ok"),
                       ("background right", "background_ok"), ("shape count right", "shape_count_ok"),
                       ("shapes right", "shapes_ok"), ("exact", "exact")):
        print(f"  {label:<18}{pct(sum(a[key] for a in a1) / n_inputs)}")

    print("\nThe sizes step's claim, by size category")
    header = f"{'Category':<10}{'Test inputs':>13}{'Predicted':>11}{'Prediction right':>18}{'Attempt 1 size right':>22}{'Cell acc.':>11}"
    print(header)
    print("-" * len(header))
    by_category = defaultdict(list)
    for task_id, r in flat:
        by_category[next(row["size_category"] for row in rows if row["task_id"] == task_id)].append(r)
    for category in ("1", "2", "3"):
        group = by_category.get(category, [])
        if not group:
            continue
        predicted = [r for r in group if r["size_receipt"] != "none"]
        right = sum(r["size_receipt"] == "yes" for r in predicted)
        a1_group = [r["attempts"][0] for r in group]
        print(f"{category:<10}{len(group):>13}{len(predicted):>11}"
              f"{(f'{right}/{len(predicted)}' if predicted else '-'):>18}"
              f"{pct(sum(a['shape_ok'] for a in a1_group) / len(group)):>22}"
              f"{pct(mean([a['cell_accuracy'] for a in a1_group])):>11}")

    print("\nBy the strategy that made attempt 1")
    header = f"{'Strategy':<20}{'Attempts':>10}{'Exact':>8}{'Size right':>12}{'Cell acc.':>11}"
    print(header)
    print("-" * len(header))
    by_strategy = defaultdict(list)
    for a in a1:
        by_strategy[family(a["strategy"])].append(a)
    for name, group in sorted(by_strategy.items(), key=lambda kv: -len(kv[1])):
        print(f"{name:<20}{len(group):>10}{sum(a['exact'] for a in group):>8}"
              f"{pct(sum(a['shape_ok'] for a in group) / len(group)):>12}"
              f"{pct(mean([a['cell_accuracy'] for a in group])):>11}")

    same_size = [a for a in a1 if a["changes"]]
    print(f"\nSame-size tasks: did attempt 1 make the changes that were needed? ({len(same_size)} test inputs)")
    needed = [a for a in same_size if a["changes"]["needed"]]
    recalls = [a["changes"]["correct"] / a["changes"]["needed"] for a in needed]
    made = [a for a in same_size if a["changes"]["made"]]
    precisions = [a["changes"]["correct"] / a["changes"]["made"] for a in made]
    print(f"  change needed but none made   : {sum(1 for a in needed if not a['changes']['made'])}/{len(needed)}")
    print(f"  mean recall (needed changes done right) : {pct(mean(recalls))}")
    print(f"  mean precision (changes made that were right): {pct(mean(precisions))}")

    print("\nCalibration: does a higher solver score mean a better attempt 1?")
    header = f"{'Solver score':<16}{'Attempts':>10}{'Exact':>8}{'Cell acc.':>11}"
    print(header)
    print("-" * len(header))
    for low, high in CALIBRATION_BINS:
        group = [a for a in a1 if low <= a["solver_score"] < high]
        if group:
            print(f"{low:.1f} to {min(high, 1.0):.1f}   {len(group):>10}{sum(a['exact'] for a in group):>8}"
                  f"{pct(mean([a['cell_accuracy'] for a in group])):>11}")

    print("\nBy task group (size category | palette | shape count), groups with 20+ tasks")
    header = f"{'Group':<26}{'Tasks':>7}{'Solved':>8}{'Cell acc.':>11}{'Size right':>12}{'Main failure':>16}"
    print(header)
    print("-" * len(header))
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    for group, items in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        if len(items) < 20:
            continue
        ids = {row["task_id"] for row in items}
        group_a1 = [r["attempts"][0] for task_id, r in flat if task_id in ids]
        failures = Counter(a["diagnosis"] for a in group_a1 if a["diagnosis"] != "EXACT")
        main = failures.most_common(1)[0][0] if failures else "-"
        print(f"{group:<26}{len(items):>7}{sum(row['solved'] == 'yes' for row in items):>8}"
              f"{pct(mean([a['cell_accuracy'] for a in group_a1])):>11}"
              f"{pct(sum(a['shape_ok'] for a in group_a1) / len(group_a1)):>12}{main:>16}")


def main():
    transformations_in = load_transformations_receipts()
    solver_in = load_solver_receipts()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        tasks = load_tasks(folder)
        check_transformations_receipts(transformations_in, dataset, list(tasks))
        check_solver_receipts(solver_in, dataset, list(tasks))
        submission = load_submission(dataset)

        rows, records_by_task, receipts[dataset] = [], {}, {}
        skipped = 0
        for task_id, task in tasks.items():
            records = verify_task(task, submission[task_id], solver_in[dataset][task_id],
                                  transformations_in[dataset][task_id])
            if records is None:
                skipped += 1
                continue
            records_by_task[task_id] = records
            receipts[dataset][task_id] = records
            rows.append(csv_row(dataset, task_id, transformations_in[dataset][task_id], records))
        if skipped:
            print(f"Note: {skipped} {dataset} tasks have no public test output and were not verified")
        if rows:
            print_summary(dataset, records_by_task, rows)
        all_rows += rows

    if not all_rows:
        raise SystemExit("No tasks with public test outputs were found, nothing to verify.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with OUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(receipts, file)
    print(f"\nPer-task results written to {OUT_CSV}")
    print(f"Verifier receipts written to {OUT_JSON}")


if __name__ == "__main__":
    main()