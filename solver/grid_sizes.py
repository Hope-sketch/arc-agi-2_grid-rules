"""
Step 1: Grid sizes (investigation + output-size prediction)

What this script does
  1. Loads every task JSON from data/training and data/evaluation.
  2. Measures each grid as (height, width) = (number of rows, number of columns).
  3. Classifies each TASK (not each pair) into rule 0-5.
  4. For rules 1-3 it predicts the test output size, and checks that against
     the real test output (the public data includes the answers).
  5. Prints a table of where the rules work and where they fail, and writes
     one row per task to grid_size_results.csv so you can inspect any task.

Rules (from your Grid_Sizes.pdf)
  0 : no training examples (safety case, should not happen in ARC-AGI-2)
  1 : every pair has input size == output size
  2 : inputs differ from outputs; inputs all one size; outputs all one size
  3 : inputs differ from outputs; inputs vary;         outputs all one size
  4 : inputs differ from outputs; inputs all one size; outputs vary
  5 : everything else (both vary, or a mix of same/different pairs)

Predictions
  rule 1   -> output size = size of the test input
  rule 2/3 -> output size = the one output size seen in training
  rule 4/5 -> no prediction yet ("unknown")
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_CSV = Path("grid_size_results.csv")


# ----------------------------------------------------------------------
# 1. Load files
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


# ----------------------------------------------------------------------
# 2. Height and width
#    ARC convention: height = rows = len(grid), width = columns = len(grid[0]).
#    So a size is always written (height, width), i.e. h x w.
# ----------------------------------------------------------------------
def grid_size(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


# ----------------------------------------------------------------------
# 3. Which rule does the task fall into?
# ----------------------------------------------------------------------
def classify_task(task):
    """Return (rule, reason, constant_output_size)."""
    train = task.get("train", [])
    test = task.get("test", [])

    if not train:
        return 0, "no_training_examples", None

    in_sizes = [grid_size(p["input"]) for p in train]
    out_sizes = [grid_size(p["output"]) for p in train]
    test_in_sizes = [grid_size(t["input"]) for t in test]

    same_flags = [i == o for i, o in zip(in_sizes, out_sizes)]

    # Rule 1: every pair keeps its size.
    if all(same_flags):
        return 1, "all_pairs_same_size", None

    # Mixed: some pairs keep their size, some do not -> undecided (rule 5).
    if any(same_flags):
        return 5, "mixed_same_and_different", None

    # From here on, every pair has input size != output size.
    # As in your PDF, the test input counts when asking "are the inputs the same?"
    inputs_constant = len(set(in_sizes + test_in_sizes)) == 1
    outputs_constant = len(set(out_sizes)) == 1

    if inputs_constant and outputs_constant:
        return 2, "inputs_constant_outputs_constant", out_sizes[0]
    if outputs_constant:
        return 3, "inputs_vary_outputs_constant", out_sizes[0]
    if inputs_constant:
        return 4, "inputs_constant_outputs_vary", None
    return 5, "both_vary", None


def predict_output_size(rule, constant_output_size, test_input):
    """Predicted (height, width) of the test output, or None if unknown."""
    if rule == 1:
        return grid_size(test_input)
    if rule in (2, 3):
        return constant_output_size
    return None


# ----------------------------------------------------------------------
# 4. Run on every task and compare with the real answer
# ----------------------------------------------------------------------
def solve_dataset(tasks):
    """One result dict per task."""
    results = []
    for task_id, task in tasks.items():
        rule, reason, const_out = classify_task(task)

        predicted = []
        actual = []
        for t in task.get("test", []):
            predicted.append(predict_output_size(rule, const_out, t["input"]))
            actual.append(grid_size(t["output"]) if "output" in t else None)

        if not predicted or any(p is None for p in predicted):
            outcome = "no_prediction"
        elif any(a is None for a in actual):
            outcome = "unchecked"
        elif predicted == actual:
            outcome = "correct"
        else:
            outcome = "wrong"

        results.append({
            "task_id": task_id,
            "rule": rule,
            "reason": reason,
            "predicted": predicted,
            "actual": actual,
            "outcome": outcome,
        })
    return results


# ----------------------------------------------------------------------
# 5. Tables
# ----------------------------------------------------------------------
def print_size_overview(name, tasks):
    """Counts of grid sizes, inputs and outputs counted separately."""
    in_counts, out_counts = Counter(), Counter()
    for task in tasks.values():
        for pair in task["train"]:
            in_counts[grid_size(pair["input"])] += 1
            out_counts[grid_size(pair["output"])] += 1
    print(f"\n--- {name}: size overview (training pairs only) ---")
    print(f"Unique input sizes:  {len(in_counts)}")
    print(f"Unique output sizes: {len(out_counts)}")
    print("Most common input sizes: ", in_counts.most_common(5))
    print("Most common output sizes:", out_counts.most_common(5))


def print_rule_table(name, results):
    print(f"\n=== {name}: where the size rules work and where they fail ===")
    header = f"{'Rule':<6}{'Tasks':>7}{'Correct':>9}{'Wrong':>7}{'No pred.':>10}{'Acc (when predicted)':>24}"
    print(header)
    print("-" * len(header))

    by_rule = defaultdict(list)
    for r in results:
        by_rule[r["rule"]].append(r)

    total = Counter()
    for rule in range(0, 6):
        rows = by_rule.get(rule, [])
        if not rows and rule == 0:
            continue
        c = sum(1 for r in rows if r["outcome"] == "correct")
        w = sum(1 for r in rows if r["outcome"] == "wrong")
        n = len(rows) - c - w
        acc = f"{c / (c + w):.1%}" if (c + w) else "-"
        print(f"{rule:<6}{len(rows):>7}{c:>9}{w:>7}{n:>10}{acc:>24}")
        total["tasks"] += len(rows)
        total["correct"] += c
        total["wrong"] += w
        total["none"] += n

    print("-" * len(header))
    c, w = total["correct"], total["wrong"]
    acc = f"{c / (c + w):.1%}" if (c + w) else "-"
    print(f"{'All':<6}{total['tasks']:>7}{c:>9}{w:>7}{total['none']:>10}{acc:>24}")
    print(f"Tasks the size step can decide: {c + w}/{total['tasks']} "
          f"({(c + w) / total['tasks']:.1%}); correct among those: {acc}")

    # Why tasks ended up in rule 5 (and 4), so later steps can use it.
    reasons = Counter(r["reason"] for r in results if r["rule"] in (4, 5))
    if reasons:
        print("\nWhy tasks are undecided (rules 4 and 5):")
        for reason, count in reasons.most_common():
            print(f"  {reason:<34}{count:>5}")

    # Every failure, so you can open those tasks.
    wrong = [r for r in results if r["outcome"] == "wrong"]
    print(f"\nWrong predictions ({len(wrong)}):")
    for r in wrong:
        print(f"  {r['task_id']}  rule {r['rule']}  predicted {r['predicted']}  actual {r['actual']}")
    if not wrong:
        print("  none")


def write_csv(all_results):
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "task_id", "rule", "reason", "predicted", "actual", "outcome"])
        for dataset, results in all_results.items():
            for r in results:
                writer.writerow([dataset, r["task_id"], r["rule"], r["reason"],
                                 r["predicted"], r["actual"], r["outcome"]])
    print(f"\nPer-task results written to {RESULTS_CSV}")


# ----------------------------------------------------------------------
def main():
    datasets = {
        "training": load_tasks(DATA_DIR / "training"),
        "evaluation": load_tasks(DATA_DIR / "evaluation"),
    }
    print("Training tasks:", len(datasets["training"]))
    print("Evaluation tasks:", len(datasets["evaluation"]))

    all_results = {}
    for name, tasks in datasets.items():
        print_size_overview(name, tasks)
        all_results[name] = solve_dataset(tasks)
        print_rule_table(name, all_results[name])

    write_csv(all_results)


if __name__ == "__main__":
    main()
