"""
Step 1 - Grid Sizes (investigation step)

Looks ONLY at the training pairs of each task (the test input is never used to
decide the category) and sorts the task into one of three categories:

  1. Input == Output        every training pair has the same size in and out
  2. Input != Output,       every training pair changes size, and all training
     Output == Output       outputs share one size (inputs may differ)
  3. Nothing equal          everything else (outputs vary, or pairs are mixed)

Transformations report line per category:

  1 -> "Output grid size is the same as the input"
  2 -> "Output grid size is the same as the (training) outputs"
  3 -> "Output grid size cannot be solved at this moment, more information required"

Usage:
  python grid_sizes.py <challenges.json | folder_of_task_jsons> [solutions.json]

  Examples:
    python grid_sizes.py arc-agi_training_challenges.json arc-agi_training_solutions.json
    python grid_sizes.py data/training

Writes: results/grid_sizes_results.csv (one row per task)
"""

import csv
import json
import os
import sys
from collections import Counter

CAT_SAME_AS_INPUT = "1"
CAT_SAME_AS_OUTPUT = "2"
CAT_UNSOLVED = "3"

REPORT = {
    CAT_SAME_AS_INPUT: "Output grid size is the same as the input",
    CAT_SAME_AS_OUTPUT: "Output grid size is the same as the output",
    CAT_UNSOLVED: "Output grid size cannot be solved at this moment, more information required",
}


# ---------------------------------------------------------------- settings
# Option A: leave these empty and the script uses <repo>/data/<DEFAULT_DATASET>/
#           (a folder of task JSON files), or failing that searches for
#           *training_challenges.json / *training_solutions.json.
# Option B: paste the full paths here, e.g.
#   CHALLENGES_PATH = r"C:\Users\Makan\...\arc-agi_training_challenges.json"
CHALLENGES_PATH = ""
SOLUTIONS_PATH = ""

DEFAULT_DATASET = "training"   # change to "evaluation" to sort the evaluation tasks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")   # results/ next to this script


def find_file(suffix):
    """Search the repo (3 folders up from this script) for a file ending in suffix."""
    root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(suffix):
                return os.path.join(folder, name)
    return ""


# ----------------------------------------------------------------- loading
def load_tasks(path):
    """Return {task_id: task_dict}. Accepts one big JSON or a folder of JSONs."""
    if os.path.isdir(path):
        tasks = {}
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                with open(os.path.join(path, name)) as fh:
                    tasks[name[:-5]] = json.load(fh)
        return tasks
    with open(path) as fh:
        return json.load(fh)


def load_solutions(path):
    if not path:
        return {}
    with open(path) as fh:
        return json.load(fh)


def size_of(grid):
    return (len(grid), len(grid[0]) if grid else 0)


def fmt(sz):
    return f"{sz[0]}x{sz[1]}"


# ------------------------------------------------------------ classification
def classify(train_pairs):
    """Return (category, detail) using training pairs only."""
    in_sizes = [size_of(p["input"]) for p in train_pairs]
    out_sizes = [size_of(p["output"]) for p in train_pairs]
    pair_same = [i == o for i, o in zip(in_sizes, out_sizes)]

    # Category 1: every pair keeps its size
    if all(pair_same):
        return CAT_SAME_AS_INPUT, "every training input == its output"

    # Category 2: every pair changes size, and all outputs share one size
    if not any(pair_same) and len(set(out_sizes)) == 1:
        if len(set(in_sizes)) == 1:
            return CAT_SAME_AS_OUTPUT, "inputs all same size, outputs all same size"
        return CAT_SAME_AS_OUTPUT, "inputs vary, outputs all same size"

    # Category 3: nothing usable is equal
    if any(pair_same):
        detail = "mixed pairs (some input == output, some not)"
    elif len(set(in_sizes)) == 1:
        detail = "inputs all same size, outputs vary"
    else:
        detail = "inputs vary, outputs vary"
    return CAT_UNSOLVED, detail


def predict_sizes(category, train_pairs, test_inputs):
    """Predicted output size for each test input (None when unsolved)."""
    if category == CAT_SAME_AS_INPUT:
        return [size_of(t["input"]) for t in test_inputs]
    if category == CAT_SAME_AS_OUTPUT:
        shared = size_of(train_pairs[0]["output"])
        return [shared for _ in test_inputs]
    return [None for _ in test_inputs]


def actual_test_sizes(task_id, task, solutions):
    """Real test output sizes if we have them (solutions file or inline outputs)."""
    if task_id in solutions:
        return [size_of(g) for g in solutions[task_id]]
    if all("output" in t for t in task.get("test", [])) and task.get("test"):
        return [size_of(t["output"]) for t in task["test"]]
    return None


# ------------------------------------------------------------------- main
def main():
    challenges = sys.argv[1] if len(sys.argv) > 1 else CHALLENGES_PATH
    solutions_path = sys.argv[2] if len(sys.argv) > 2 else SOLUTIONS_PATH

    if not challenges:
        # your layout: <repo>/data/training/<task_id>.json (folder of task files)
        repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
        default_folder = os.path.join(repo_root, "data", DEFAULT_DATASET)
        if os.path.isdir(default_folder):
            challenges = default_folder
        else:
            challenges = find_file("training_challenges.json")
            if challenges and not solutions_path:
                solutions_path = find_file("training_solutions.json")

    if not challenges or not os.path.exists(challenges):
        print("Could not find the ARC data file.")
        print("Set CHALLENGES_PATH (and SOLUTIONS_PATH) at the top of grid_sizes.py,")
        print("or run:  python grid_sizes.py <challenges.json> [solutions.json]")
        sys.exit(1)

    print(f"Challenges: {challenges}")
    print(f"Solutions:  {solutions_path or '(none, sizes will not be checked)'}")

    tasks = load_tasks(challenges)
    solutions = load_solutions(solutions_path)

    rows = []
    for task_id, task in tasks.items():
        train = task["train"]
        test = task.get("test", [])
        category, detail = classify(train)
        predicted = predict_sizes(category, train, test)
        actual = actual_test_sizes(task_id, task, solutions)

        if category == CAT_UNSOLVED:
            size_correct = "n/a"          # grid sizes does not try to solve these
        elif actual is None:
            size_correct = "unknown"      # no answers available to check against
        else:
            size_correct = "yes" if predicted == actual else "no"

        rows.append({
            "task_id": task_id,
            "n_train_pairs": len(train),
            "train_input_sizes": ";".join(fmt(size_of(p["input"])) for p in train),
            "train_output_sizes": ";".join(fmt(size_of(p["output"])) for p in train),
            "category": category,
            "detail": detail,
            "report": REPORT[category],
            "test_input_sizes": ";".join(fmt(size_of(t["input"])) for t in test),
            "predicted_output_sizes": ";".join(fmt(s) if s else "" for s in predicted),
            "actual_output_sizes": ";".join(fmt(s) for s in actual) if actual else "",
            "size_correct": size_correct,
        })

    # ---- write the per-task CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "grid_sizes_results.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summary table
    total = len(rows)
    counts = Counter(r["category"] for r in rows)
    details = Counter((r["category"], r["detail"]) for r in rows)

    print(f"\nGrid sizes: {total} tasks\n")
    print(f"{'Category':<10}{'Tasks':>7}{'Share':>8}{'Size right':>12}{'Size wrong':>12}{'Unknown':>9}")
    for cat in (CAT_SAME_AS_INPUT, CAT_SAME_AS_OUTPUT, CAT_UNSOLVED):
        in_cat = [r for r in rows if r["category"] == cat]
        yes = sum(r["size_correct"] == "yes" for r in in_cat)
        no = sum(r["size_correct"] == "no" for r in in_cat)
        unk = sum(r["size_correct"] == "unknown" for r in in_cat)
        right, wrong, unknown = (str(yes), str(no), str(unk)) if cat != CAT_UNSOLVED else ("-", "-", "-")
        share = f"{100 * counts[cat] / total:.1f}%" if total else "0%"
        print(f"{cat:<10}{counts[cat]:>7}{share:>8}{right:>12}{wrong:>12}{unknown:>9}")

    print("\nBreakdown:")
    for (cat, detail), n in sorted(details.items()):
        print(f"  Category {cat} | {detail}: {n}")

    solvable = [r for r in rows if r["category"] != CAT_UNSOLVED and r["size_correct"] in ("yes", "no")]
    if solvable:
        hits = sum(r["size_correct"] == "yes" for r in solvable)
        print(f"\nSize prediction on categories 1-2: {hits}/{len(solvable)} "
              f"({100 * hits / len(solvable):.1f}%)")
    print(f"\nPer-task results written to {out_path}")


if __name__ == "__main__":
    main()