"""
Step 4: Grid cells (investigation only, not used to solve yet)

Compares training inputs and outputs cell by cell (row, column).
Only possible when input and output have the SAME height x width.
A task is "applicable" when every training pair is same-size.
Other tasks are grouped as "not_same_size" (sizes step handles those).

Background = most frequent colour of the INPUT grid (ties: lowest colour number).

Rule A: HOW cells change   (per training pair, then per task)
      identical       : no cell changes
      added_only      : every changed cell goes background -> colour
      removed_only    : every changed cell goes colour -> background
      recoloured_only : every changed cell goes colour -> a different colour
      mixed           : more than one of the above
  Task group: every pair the same -> that group, anything else -> "other".

Rule B: WHY (can one colour map explain every cell?)
      map_consistent : across ALL training pairs each input colour always
                       becomes the same output colour (identity included)
      no_map         : at least one input colour becomes different colours
  Test check for map_consistent: apply the map to the test input
  (colours never seen in training map to themselves) and compare with the
  public test output. holds = exact match, breaks = no.

Test check for Rule A: holds = the test pair has the same kind of change as the
  training group, breaks = it does not (or the test pair is not same-size).

Outputs
  * tables in the terminal
  * results/grid_cells_results.csv  (one row per task)
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_tasks(folder):
    if not folder.is_dir():
        raise SystemExit(f"Data folder not found: {folder}")
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


def same_size(a, b):
    return len(a) == len(b) and len(a[0]) == len(b[0])


def background_colour(grid):
    counts = Counter(c for row in grid for c in row)
    return max(counts, key=lambda col: (counts[col], -col))


def cell_changes(grid_in, grid_out):
    """Returns (kind, changed_fraction) for a same-size pair."""
    bg = background_colour(grid_in)
    kinds = set()
    changed = 0
    total = 0
    for row_in, row_out in zip(grid_in, grid_out):
        for a, b in zip(row_in, row_out):
            total += 1
            if a == b:
                continue
            changed += 1
            if a == bg:
                kinds.add("added")
            elif b == bg:
                kinds.add("removed")
            else:
                kinds.add("recoloured")
    if not kinds:
        kind = "identical"
    elif len(kinds) == 1:
        kind = f"{next(iter(kinds))}_only"
    else:
        kind = "mixed"
    return kind, (changed / total if total else 0.0)


def build_colour_map(pairs):
    mapping = defaultdict(set)
    for grid_in, grid_out in pairs:
        for row_in, row_out in zip(grid_in, grid_out):
            for a, b in zip(row_in, row_out):
                mapping[a].add(b)
    consistent = all(len(v) == 1 for v in mapping.values())
    return ({a: next(iter(v)) for a, v in mapping.items()} if consistent else None)


def apply_map(grid, mapping):
    return [[mapping.get(c, c) for c in row] for row in grid]


def group_of(kinds):
    unique = set(kinds)
    if len(unique) == 1:
        return next(iter(unique)), "same_in_every_pair"
    return "other", "pairs_disagree"


def analyse_task(task):
    train = task.get("train", [])
    test = [t for t in task.get("test", []) if "output" in t]

    blank = {"applicable": "no", "cell_group": "not_same_size",
             "cell_reason": "", "cell_test": "n/a", "map_group": "n/a",
             "map_test": "n/a", "map_size": 0, "avg_changed_pct": "",
             "per_pair_changed": ""}

    if not train or not all(same_size(p["input"], p["output"]) for p in train):
        blank["cell_reason"] = "no_training_examples" if not train else "a_training_pair_changes_size"
        return blank

    results = [cell_changes(p["input"], p["output"]) for p in train]
    kinds = [k for k, _ in results]
    fractions = [f for _, f in results]
    group, reason = group_of(kinds)

    test_kinds = []
    for t in test:
        if same_size(t["input"], t["output"]):
            test_kinds.append(cell_changes(t["input"], t["output"])[0])
        else:
            test_kinds.append("size_differs")
    if group == "other" or not test_kinds:
        cell_test = "n/a"
    else:
        cell_test = "holds" if set(test_kinds) == {group} else "breaks"

    mapping = build_colour_map([(p["input"], p["output"]) for p in train])
    if mapping is None:
        map_group, map_test, map_size = "no_map", "n/a", 0
    else:
        map_group, map_size = "map_consistent", len(mapping)
        if not test:
            map_test = "n/a"
        else:
            ok = all(same_size(t["input"], t["output"])
                     and apply_map(t["input"], mapping) == t["output"] for t in test)
            map_test = "holds" if ok else "breaks"

    return {"applicable": "yes", "cell_group": group, "cell_reason": reason,
            "cell_test": cell_test, "map_group": map_group, "map_test": map_test,
            "map_size": map_size,
            "avg_changed_pct": f"{100 * sum(fractions) / len(fractions):.1f}",
            "per_pair_changed": " | ".join(f"{100 * f:.1f}%" for f in fractions)}


def print_group_table(title, results, group_key, test_key, order):
    print(f"\n{title}")
    header = f"{'Group':<16}{'Tasks':>7}{'Holds':>8}{'Breaks':>8}{'n/a':>6}{'Holds (when checked)':>24}"
    print(header)
    print("-" * len(header))
    by_group = defaultdict(list)
    for r in results:
        by_group[r[group_key]].append(r)
    for group in order:
        rows = by_group.get(group, [])
        if not rows:
            continue
        h = sum(1 for r in rows if r[test_key] == "holds")
        b = sum(1 for r in rows if r[test_key] == "breaks")
        n = len(rows) - h - b
        rate = f"{h / (h + b):.1%}" if (h + b) else "-"
        print(f"{group:<16}{len(rows):>7}{h:>8}{b:>8}{n:>6}{rate:>24}")


def print_dataset(name, results):
    applicable = [r for r in results if r["applicable"] == "yes"]
    print(f"\n=== {name}: cells ({len(results)} tasks, {len(applicable)} with same-size pairs) ===")
    print_group_table("Rule A: how cells change (does the training pattern hold on the test pair?)",
                      results, "cell_group", "cell_test",
                      ["identical", "added_only", "removed_only", "recoloured_only",
                       "mixed", "other", "not_same_size"])
    print_group_table("Rule B: one colour map explains every training cell (does it solve the test pair?)",
                      applicable, "map_group", "map_test", ["map_consistent", "no_map"])
    pcts = [float(r["avg_changed_pct"]) for r in applicable if r["avg_changed_pct"] != ""]
    if pcts:
        pcts.sort()
        print(f"\nCells changed per task (same-size tasks): median {pcts[len(pcts)//2]:.1f}%, "
              f"mean {sum(pcts)/len(pcts):.1f}%")


def write_csv(all_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "grid_cells_results.csv"
    columns = ["dataset", "task_id", "applicable", "cell_group", "cell_reason",
               "cell_test", "map_group", "map_test", "map_size",
               "avg_changed_pct", "per_pair_changed"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for dataset, results in all_results.items():
            for r in results:
                writer.writerow([dataset] + [r[c] for c in columns[1:]])
    print(f"\nPer-task results written to {out_path}")


def main():
    datasets = {"training": load_tasks(DATA_DIR / "training"),
                "evaluation": load_tasks(DATA_DIR / "evaluation")}
    all_results = {}
    for name, tasks in datasets.items():
        rows = []
        for task_id, task in tasks.items():
            row = analyse_task(task)
            row["task_id"] = task_id
            rows.append(row)
        all_results[name] = rows
        print_dataset(name, rows)
    write_csv(all_results)


if __name__ == "__main__":
    main()
