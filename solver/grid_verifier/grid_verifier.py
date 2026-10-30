"""
Step 8: the Grid Verifier

Checks the predictions written by grid_solver.py against the public test answers and
sorts the failures. The solver never sees these answers.

A test pair counts as solved when attempt_1 OR attempt_2 matches the answer exactly.
A task's score is the share of its test pairs solved, and the overall score is the
sum of task scores divided by the number of tasks (the Kaggle way).

Failure kinds (for a test pair no attempt solved)
  no_method    : no learned method fitted the training pairs, only fallbacks were offered
  wrong_size   : a learned method was offered, but the predicted grid has the wrong size
  wrong_cells  : right size, wrong cells (the cell accuracy is recorded)

Usage
  python solver/grid_verifier/grid_verifier.py evaluation     (or training, or a folder path)

Outputs
  * tables in the terminal
  * results/grid_verifier_results_<name>.csv   one row per task
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parent
ROOT = SOLVER.parent
DATA_DIR = ROOT / "data"
SOLVER_RESULTS = SOLVER / "grid_solver" / "results"

MIN_GROUP_TASKS = 10


def family_of(method):
    return method.split(":")[0]


def cell_accuracy(predicted, expected):
    if predicted.shape != expected.shape:
        return 0.0
    return float((predicted == expected).mean())


def read_solver_rows(dataset):
    path = SOLVER_RESULTS / f"grid_solver_results_{dataset}.csv"
    if not path.exists():
        raise SystemExit(f"Run the solver first: python solver/grid_solver/grid_solver.py {dataset}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return {r["task_id"]: r for r in csv.DictReader(file)}


def verify_task(task, attempts, solver_row):
    methods = [solver_row["attempt_1_method"], solver_row["attempt_2_method"]]
    solved, kinds, accuracies, winner = 0, [], [], ""
    for pair, tries in zip(task["test"], attempts):
        expected = np.array(pair["output"], dtype=int)
        grids = [np.array(tries["attempt_1"], dtype=int), np.array(tries["attempt_2"], dtype=int)]
        hit = next((k for k, g in enumerate(grids)
                    if g.shape == expected.shape and np.array_equal(g, expected)), None)
        accuracies.append(max(cell_accuracy(g, expected) for g in grids))
        if hit is not None:
            solved += 1
            winner = winner or methods[hit]
            continue
        learned = [g for g, m in zip(grids, methods) if family_of(m) != "fallback"]
        if not learned:
            kinds.append("no_method")
        elif all(g.shape != expected.shape for g in learned):
            kinds.append("wrong_size")
        else:
            kinds.append("wrong_cells")

    total = len(task["test"])
    if solved == total:
        failure = "solved"
    elif solved:
        failure = "partly"
    else:
        failure = Counter(kinds).most_common(1)[0][0]
    return {"solved_pairs": solved, "test_pairs": total, "score": solved / total,
            "winner_method": winner, "failure_kind": failure,
            "best_cell_accuracy": f"{sum(accuracies) / total:.3f}"}


def print_tables(dataset, rows):
    score = sum(r["score"] for r in rows)
    full = sum(1 for r in rows if r["failure_kind"] == "solved")
    partly = sum(1 for r in rows if r["failure_kind"] == "partly")
    print(f"\n=== {dataset}: verifier ({len(rows)} tasks) ===")
    print(f"Score: {score:.2f} of {len(rows)} ({100 * score / len(rows):.1f}%)   "
          f"fully solved {full}, partly solved {partly}")

    print("\nFailures")
    for kind, n in Counter(r["failure_kind"] for r in rows).most_common():
        print(f"  {kind:<12}{n:>6}")

    offered = Counter(family_of(r["attempt_1_method"]) for r in rows)
    won = Counter(family_of(r["winner_method"]) for r in rows if r["winner_method"])
    print(f"\nMethod of the first attempt: how often offered, and solved by either attempt")
    for family, n in offered.most_common():
        print(f"  {family:<14}{n:>6} offered{won.get(family, 0):>6} solved")
    print("  (a solve by attempt_2 is credited to the method that made attempt_2)")

    print(f"\nBy group (groups with {MIN_GROUP_TASKS}+ tasks here)")
    header = f"{'Group':<28}{'Tasks':>6}{'Solved':>8}{'Offered':>9}{'Mean cell acc.':>16}"
    print(header)
    print("-" * len(header))
    by_group = defaultdict(list)
    for r in rows:
        if r["group"]:
            by_group[r["group"]].append(r)
    for group, g_rows in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        if len(g_rows) < MIN_GROUP_TASKS:
            continue
        solved = sum(r["score"] for r in g_rows)
        offered_n = sum(1 for r in g_rows if int(r["learned_methods"]) > 0)
        acc = sum(float(r["best_cell_accuracy"]) for r in g_rows) / len(g_rows)
        print(f"{group:<28}{len(g_rows):>6}{solved:>8.1f}{offered_n:>9}{acc:>16.3f}")


def main():
    name_or_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    folder = Path(name_or_path) if Path(name_or_path).is_dir() else DATA_DIR / name_or_path
    dataset = folder.name

    solver_rows = read_solver_rows(dataset)
    with (SOLVER_RESULTS / f"grid_solver_predictions_{dataset}.json").open(encoding="utf-8") as file:
        predictions = json.load(file)

    rows = []
    for task_id, solver_row in solver_rows.items():
        with (folder / f"{task_id}.json").open("r", encoding="utf-8") as file:
            task = json.load(file)
        result = verify_task(task, predictions[task_id], solver_row)
        result.update({"dataset": dataset, "task_id": task_id, "group": solver_row["group"],
                       "learned_methods": solver_row["learned_methods"],
                       "attempt_1_method": solver_row["attempt_1_method"],
                       "attempt_2_method": solver_row["attempt_2_method"]})
        rows.append(result)

    print_tables(dataset, rows)

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grid_verifier_results_{dataset}.csv"
    columns = ["dataset", "task_id", "group", "solved_pairs", "test_pairs", "score",
               "winner_method", "attempt_1_method", "attempt_2_method", "learned_methods",
               "failure_kind", "best_cell_accuracy"]
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] for c in columns])
    print(f"\nPer-task results written to {out_path}")


if __name__ == "__main__":
    main()