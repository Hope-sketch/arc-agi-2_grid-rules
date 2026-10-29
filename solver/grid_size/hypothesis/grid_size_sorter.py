"""
Step 1: Grid sizes (SORTER only, no size prediction)

This step does not predict anything. It looks at the TRAINING pairs of a task
(the test input is not used) and sorts the task into one size rule, so the
Grid Solver at the end knows which kind of task it is dealing with.

Rules (strict)
  1 : Input == output, in EVERY training pair. Nothing else.
  2 : Input != output in every pair, all inputs are the same size,
      all outputs are the same size.
  3 : Input != output in every pair, inputs vary, all outputs are the same size.
  4 : Input != output in every pair, all inputs are the same size, outputs vary.
  5 : Input != output in every pair, inputs vary AND outputs vary.
      Also here: tasks that MIX pairs (some pairs keep their size, some do not).
  0 : no training examples (safety case)

Size is always (height, width) = (rows, columns).

Direction labels (per pair, height and width compared separately)
  absolutely_smaller / absolutely_bigger / mixed_1 (h bigger, w smaller) /
  mixed_2 (h smaller, w bigger) / mixed_3 (h same, w changed) /
  mixed_4 (w same, h changed) / total_drop (output 1x1, takes priority) / same

Test check (the "where does it fail" part)
  The sorting comes from the training pairs only. Afterwards the public test
  answers are used to ask whether the same thing is also true on the test pair:
      rule 1  test input == test output
      rule 2  test input is the training input size AND test output is the training output size
      rule 3  test output is the training output size
      rule 4  test input is the training input size
      rule 5  n/a (nothing is claimed)
  holds / breaks / n/a, per task (all test pairs must hold).

Outputs
  * tables in the terminal
  * results/grid_size_results.csv (one row per task)

Run from the project root (the folder that holds data/):
  python solver/grid_sizes.py
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_DIR, results_dir  # noqa: E402
RESULTS_DIR = results_dir("grid_sizes")
RESULTS_CSV = RESULTS_DIR / "grid_sizes_sort_results.csv"

RULE_TEXT = {
    0: "no training examples",
    1: "input == output",
    2: "input != output; inputs all same; outputs all same",
    3: "input != output; inputs vary; outputs all same",
    4: "input != output; inputs all same; outputs vary",
    5: "inputs vary and outputs vary (or pairs mixed)",
}


# ----------------------------------------------------------------------
# Load / measure
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as f:
            tasks[file_path.stem] = json.load(f)
    return tasks


def grid_size(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


def fmt(size):
    return f"{size[0]}x{size[1]}"


def sign(a, b):
    return (b > a) - (b < a)


def direction_label(size_in, size_out):
    if size_out == (1, 1) and size_in != (1, 1):
        return "total_drop"
    dh, dw = sign(size_in[0], size_out[0]), sign(size_in[1], size_out[1])
    if dh == 0 and dw == 0:
        return "same"
    if dh < 0 and dw < 0:
        return "absolutely_smaller"
    if dh > 0 and dw > 0:
        return "absolutely_bigger"
    if dh > 0 and dw < 0:
        return "mixed_1"
    if dh < 0 and dw > 0:
        return "mixed_2"
    return "mixed_3" if dh == 0 else "mixed_4"


# ----------------------------------------------------------------------
# Sorting (training pairs only)
# ----------------------------------------------------------------------
def classify(in_sizes, out_sizes):
    """Return (rule, reason)."""
    if not in_sizes:
        return 0, "no_training_examples"

    same = [i == o for i, o in zip(in_sizes, out_sizes)]
    if all(same):
        return 1, "every_pair_input_equals_output"
    if any(same):
        return 5, "mixed_some_pairs_same_some_different"

    inputs_same = len(set(in_sizes)) == 1
    outputs_same = len(set(out_sizes)) == 1
    if inputs_same and outputs_same:
        return 2, "inputs_same_outputs_same"
    if outputs_same:
        return 3, "inputs_vary_outputs_same"
    if inputs_same:
        return 4, "inputs_same_outputs_vary"
    return 5, "inputs_vary_outputs_vary"


def test_check(rule, in_sizes, out_sizes, test_pairs):
    """Does the sorting also describe the public test pair(s)?"""
    if rule in (0, 5) or not test_pairs:
        return "n/a"
    ok = []
    for t_in, t_out in test_pairs:
        if rule == 1:
            ok.append(t_in == t_out)
        elif rule == 2:
            ok.append(t_in == in_sizes[0] and t_out == out_sizes[0])
        elif rule == 3:
            ok.append(t_out == out_sizes[0])
        elif rule == 4:
            ok.append(t_in == in_sizes[0])
    return "holds" if all(ok) else "breaks"


def analyse_task(task):
    train = task.get("train", [])
    test = [t for t in task.get("test", []) if "output" in t]

    in_sizes = [grid_size(p["input"]) for p in train]
    out_sizes = [grid_size(p["output"]) for p in train]
    test_pairs = [(grid_size(t["input"]), grid_size(t["output"])) for t in test]

    rule, reason = classify(in_sizes, out_sizes)
    return {
        "rule": rule,
        "reason": reason,
        "train_pairs": " | ".join(f"{fmt(i)}>{fmt(o)}" for i, o in zip(in_sizes, out_sizes)),
        "directions": "|".join(direction_label(i, o) for i, o in zip(in_sizes, out_sizes)),
        "test_pairs": " | ".join(f"{fmt(i)}>{fmt(o)}" for i, o in test_pairs),
        "test_check": test_check(rule, in_sizes, out_sizes, test_pairs),
    }


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def print_table(name, results):
    print(f"\n=== {name}: tasks sorted by size rule ({len(results)} tasks) ===")
    header = f"{'Rule':<6}{'Tasks':>7}{'Share':>8}{'Holds':>8}{'Breaks':>8}{'n/a':>6}  What it means"
    print(header)
    print("-" * 78)
    for rule in range(0, 6):
        rows = [r for r in results.values() if r["rule"] == rule]
        if not rows and rule == 0:
            continue
        h = sum(r["test_check"] == "holds" for r in rows)
        b = sum(r["test_check"] == "breaks" for r in rows)
        n = len(rows) - h - b
        print(f"{rule:<6}{len(rows):>7}{len(rows) / len(results):>8.1%}{h:>8}{b:>8}{n:>6}  {RULE_TEXT[rule]}")

    mixed = [t for t, r in results.items() if r["reason"].startswith("mixed")]
    print(f"\nRule 5 split: both vary {sum(r['reason'] == 'inputs_vary_outputs_vary' for r in results.values())}, "
          f"mixed pairs {len(mixed)}")

    print("\nDirection of change per training pair (rules 2-5):")
    counts = Counter()
    for r in results.values():
        if r["rule"] >= 2:
            counts.update(r["directions"].split("|"))
    for label, n in counts.most_common():
        print(f"  {label:<20}{n:>6}")

    broken = [(t, r) for t, r in results.items() if r["test_check"] == "breaks"]
    print(f"\nTasks where the sorting breaks on the test pair ({len(broken)}):")
    for t, r in broken:
        print(f"  {t}  rule {r['rule']}  train {r['train_pairs']}  test {r['test_pairs']}")
    if not broken:
        print("  none")


def write_csv(all_results):
    RESULTS_DIR.mkdir(exist_ok=True)
    cols = ["dataset", "task_id", "rule", "reason", "train_pairs", "directions",
            "test_pairs", "test_check"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ds, res in all_results.items():
            for tid, r in res.items():
                w.writerow([ds, tid] + [r[c] for c in cols[2:]])
    print(f"\nPer-task results written to {RESULTS_CSV}")


def main():
    all_results = {}
    for name in ("training", "evaluation"):
        tasks = load_tasks(DATA_DIR / name)
        all_results[name] = {tid: analyse_task(t) for tid, t in tasks.items()}
        print_table(name, all_results[name])
    write_csv(all_results)


if __name__ == "__main__":
    main()
