"""
Step 1: Grid sizes (investigation only)

Looks ONLY at the training pairs of each task (the test input is never used to
decide the category) and sorts the task into one of three categories:

  1  Input == Output        every training pair has the same size in and out
  2  Input != Output,       every training pair changes size and all training
     Output == Output       outputs share one size (inputs may differ)
  3  Nothing equal          everything else (outputs vary, or pairs are mixed)

Report line per category (the human-readable part of the receipt):
  1  Output grid size is the same as the input
  2  Output grid size is the same as the output
  3  Output grid size cannot be solved at this moment, more information required

The RECEIPT is what later steps read. Use the category code and predicted sizes,
not the report wording:
  {"category": "1", "detail": "...", "report": "...",
   "test_input_sizes": [[h, w], ...], "predicted_output_sizes": [[h, w] or null, ...]}
It never contains the test output, so nothing leaks into the solver.

Usage
  python solver/grid_sizes/grid_sizes.py

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_sizes/results/grid_sizes_results.csv   (one row per task)
  * solver/grid_sizes/results/grid_sizes_receipts.json (what later steps load)

From another step
  import sys; sys.path.insert(0, str(ROOT / "solver" / "grid_sizes"))
  from grid_sizes import load_receipts, check_receipts
"""

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_sizes_results.csv"
OUT_JSON = RESULTS_DIR / "grid_sizes_receipts.json"
DATASETS = ("training", "evaluation")

SAME_AS_INPUT, SAME_AS_OUTPUT, UNSOLVED = "1", "2", "3"

REPORT = {
    SAME_AS_INPUT: "Output grid size is the same as the input",
    SAME_AS_OUTPUT: "Output grid size is the same as the output",
    UNSOLVED: "Output grid size cannot be solved at this moment, more information required",
}


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


def size_of(grid):
    return (len(grid), len(grid[0]) if grid else 0)


def fmt(size):
    return f"{size[0]}x{size[1]}"


# ----------------------------------------------------------------------
# Receipt (training pairs only)
# ----------------------------------------------------------------------
def classify(train):
    in_sizes = [size_of(p["input"]) for p in train]
    out_sizes = [size_of(p["output"]) for p in train]
    pair_same = [i == o for i, o in zip(in_sizes, out_sizes)]

    if all(pair_same):
        return SAME_AS_INPUT, "every training input == its output"
    if not any(pair_same) and len(set(out_sizes)) == 1:
        if len(set(in_sizes)) == 1:
            return SAME_AS_OUTPUT, "inputs all same size, outputs all same size"
        return SAME_AS_OUTPUT, "inputs vary, outputs all same size"
    if any(pair_same):
        return UNSOLVED, "mixed pairs (some input == output, some not)"
    if len(set(in_sizes)) == 1:
        return UNSOLVED, "inputs all same size, outputs vary"
    return UNSOLVED, "inputs vary, outputs vary"


def get_sizes_receipt(task):
    """The receipt for one task. Uses the training pairs and the test INPUT sizes."""
    train = task["train"]
    category, detail = classify(train)
    test_inputs = [list(size_of(t["input"])) for t in task["test"]]

    if category == SAME_AS_INPUT:
        predicted = [list(s) for s in test_inputs]
    elif category == SAME_AS_OUTPUT:
        shared = list(size_of(train[0]["output"]))
        predicted = [shared for _ in test_inputs]
    else:
        predicted = [None for _ in test_inputs]

    return {"category": category, "detail": detail, "report": REPORT[category],
            "test_input_sizes": test_inputs, "predicted_output_sizes": predicted}


def check_against_test(receipt, task):
    """yes / no / n/a (category 3) / unknown (no public test output). Verification only."""
    if receipt["category"] == UNSOLVED:
        return "n/a", ""
    if not all("output" in t for t in task["test"]):
        return "unknown", ""
    actual = [list(size_of(t["output"])) for t in task["test"]]
    text = ";".join(fmt(s) for s in actual)
    return ("yes" if actual == receipt["predicted_output_sizes"] else "no"), text


# ----------------------------------------------------------------------
# For the other steps
# ----------------------------------------------------------------------
def load_receipts(path=OUT_JSON):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid sizes receipts not found: {path}\n"
                         f"Run solver/grid_sizes/grid_sizes.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_receipts(receipts, dataset, task_ids):
    """Stop if any task has no grid sizes receipt, and say which ones."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid sizes receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_sizes/grid_sizes.py.")
    print(f"Grid sizes receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def print_summary(dataset, rows):
    total = len(rows)
    print(f"\n=== {dataset}: grid sizes ({total} tasks) ===")
    header = f"{'Category':<10}{'Tasks':>7}{'Share':>8}{'Right':>8}{'Wrong':>8}{'Unknown':>9}"
    print(header)
    print("-" * len(header))
    for cat in (SAME_AS_INPUT, SAME_AS_OUTPUT, UNSOLVED):
        in_cat = [r for r in rows if r["category"] == cat]
        share = f"{100 * len(in_cat) / total:.1f}%" if total else "-"
        if cat == UNSOLVED:
            print(f"{cat:<10}{len(in_cat):>7}{share:>8}{'-':>8}{'-':>8}{'-':>9}")
        else:
            count = lambda word: sum(r["size_correct"] == word for r in in_cat)
            print(f"{cat:<10}{len(in_cat):>7}{share:>8}{count('yes'):>8}{count('no'):>8}{count('unknown'):>9}")

    print("Breakdown:")
    for (cat, detail), n in sorted(Counter((r["category"], r["detail"]) for r in rows).items()):
        print(f"  {cat} | {detail}: {n}")

    checked = [r for r in rows if r["size_correct"] in ("yes", "no")]
    if checked:
        hits = sum(r["size_correct"] == "yes" for r in checked)
        print(f"Size right on categories 1-2: {hits}/{len(checked)} ({100 * hits / len(checked):.1f}%)")


def main():
    columns = ["dataset", "task_id", "n_train_pairs", "train_input_sizes", "train_output_sizes",
               "category", "detail", "report", "test_input_sizes", "predicted_output_sizes",
               "actual_output_sizes", "size_correct"]
    all_rows, receipts = [], {}

    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        rows, receipts[dataset] = [], {}
        for task_id, task in load_tasks(folder).items():
            receipt = get_sizes_receipt(task)
            receipts[dataset][task_id] = receipt
            size_correct, actual = check_against_test(receipt, task)
            train = task["train"]
            rows.append({
                "dataset": dataset, "task_id": task_id, "n_train_pairs": len(train),
                "train_input_sizes": ";".join(fmt(size_of(p["input"])) for p in train),
                "train_output_sizes": ";".join(fmt(size_of(p["output"])) for p in train),
                "category": receipt["category"], "detail": receipt["detail"],
                "report": receipt["report"],
                "test_input_sizes": ";".join(fmt(s) for s in receipt["test_input_sizes"]),
                "predicted_output_sizes": ";".join(fmt(s) if s else "" for s in receipt["predicted_output_sizes"]),
                "actual_output_sizes": actual, "size_correct": size_correct,
            })
        print_summary(dataset, rows)
        all_rows += rows

    if not all_rows:
        raise SystemExit(f"No tasks found under {DATA_DIR}. Expected data/training and data/evaluation.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    with OUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(receipts, file)
    print(f"\nPer-task results written to {OUT_CSV}")
    print(f"Receipts for later steps written to {OUT_JSON}")


if __name__ == "__main__":
    main()