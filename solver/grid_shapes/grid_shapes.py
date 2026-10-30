"""
Step 3: Grid shapes (investigation only, not used to solve yet)

What counts as a shape
  A shape is a group of cells of the SAME colour joined side by side
  (up, down, left, right). Diagonal cells do NOT join.
  The background colour is not a shape.
  A cell with no same-coloured side neighbour is a shape on its own (size 1).
  Two shapes of the same colour that do not touch by a side are two shapes.

Form
  The form of a shape is its cell layout with the position removed.
  Colour is NOT part of the form (colour belongs to the colours step).

Rule A: shape COUNT   (per training pair, compare shapes in input vs output)
      added    : the output has more shapes than the input
      removed  : the output has fewer shapes
      same     : equal counts
  Task group: every pair the same -> that group, anything else -> "other".

Rule B: shape FORMS   (per training pair, compare the SET of forms)
      same_forms : input and output contain exactly the same forms
      kept_some  : not equal, but they share at least one form
      none_shared: no form in common
  Task group: every pair the same -> that group, anything else -> "other".
  Also recorded: forms that appear in EVERY training input and output.

Test check (this is the "where does it fail" part)
  Groups come from the training pairs only. Afterwards the public test
  answers are used to ask whether the same thing also happens in the test
  pair: holds = yes, breaks = no, n/a = group is "other" / "none".

Test input check
  The number of shapes in each test input is compared with the number of
  shapes in the training inputs: within the training range, more, or fewer.

Background
  Background = the most frequent colour in a grid. If two colours tie, the
  tied colour whose biggest connected region is largest is used, and the
  grid is counted in "bg_ties" so you can see how often this happens.

Usage
  python solver/grid_shapes.py

Outputs
  * tables in the terminal
  * results/grid_shapes_results.csv  (one row per task)
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_PATH = Path("results") / "grid_shapes_results.csv"

SIDE_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # no diagonals


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


# ----------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------
def find_regions(grid):
    """Every same-colour region joined by sides, background included."""
    height = len(grid)
    width = len(grid[0])
    seen = [[False] * width for _ in range(height)]
    regions = []

    for r in range(height):
        for c in range(width):
            if seen[r][c]:
                continue
            colour = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in SIDE_NEIGHBOURS:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr][nc] and grid[nr][nc] == colour):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            regions.append({"colour": colour, "cells": cells})
    return regions


def background_colour(grid, regions):
    """Most frequent colour. Returns (colour, was_tie)."""
    ranked = Counter(cell for row in grid for cell in row).most_common()
    top = ranked[0][1]
    tied = [colour for colour, n in ranked if n == top]
    if len(tied) == 1:
        return tied[0], False
    # Tie: pick the tied colour with the biggest single connected region.
    best = max(tied, key=lambda col: max(len(reg["cells"]) for reg in regions
                                         if reg["colour"] == col))
    return best, True


def form_of(cells):
    """Cell layout with the position removed (so it can be compared)."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def find_shapes(grid):
    """Returns (shapes, was_tie). Each shape: colour, size, form."""
    if not grid or not grid[0]:
        return [], False
    regions = find_regions(grid)
    bg, was_tie = background_colour(grid, regions)
    shapes = [{"colour": reg["colour"],
               "size": len(reg["cells"]),
               "form": form_of(reg["cells"])}
              for reg in regions if reg["colour"] != bg]
    return shapes, was_tie


# ----------------------------------------------------------------------
# Pair level
# ----------------------------------------------------------------------
def count_change(shapes_in, shapes_out):
    if len(shapes_out) > len(shapes_in):
        return "added"
    if len(shapes_out) < len(shapes_in):
        return "removed"
    return "same"


def form_change(shapes_in, shapes_out):
    forms_in = {s["form"] for s in shapes_in}
    forms_out = {s["form"] for s in shapes_out}
    if forms_in == forms_out:
        return "same_forms"
    if forms_in & forms_out:
        return "kept_some"
    return "none_shared"


# ----------------------------------------------------------------------
# Task level
# ----------------------------------------------------------------------
def group_of(kinds):
    unique = set(kinds)
    if len(unique) == 1:
        return next(iter(unique)), "same_in_every_pair"
    return "other", "pairs_disagree"


def test_check(group, test_kinds):
    if group in ("other", "none") or not test_kinds:
        return "n/a"
    return "holds" if set(test_kinds) == {group} else "breaks"


def shared_forms_check(shared, test_has_shared):
    if not shared or not test_has_shared:
        return "n/a"
    return "holds" if all(test_has_shared) else "breaks"


def test_input_vs_training(train_input_counts, test_input_counts):
    """Is the number of shapes in the test input within, above or below the training inputs?"""
    if not test_input_counts:
        return "n/a"
    low, high = min(train_input_counts), max(train_input_counts)
    labels = set()
    for n in test_input_counts:
        labels.add("more" if n > high else "fewer" if n < low else "within")
    return labels.pop() if len(labels) == 1 else "mixed"


def empty_result(reason):
    return {"count_group": "none", "count_reason": reason, "count_test": "n/a",
            "form_group": "none", "form_reason": reason, "form_test": "n/a",
            "shared_forms": 0, "shared_forms_in_test": "n/a",
            "per_pair_counts": "", "test_input_shapes": "",
            "test_vs_train_inputs": "n/a", "bg_ties": 0}


def analyse_task(task):
    train = task.get("train", [])
    if not train:
        return empty_result("no_training_examples")

    # Training pairs
    ties = 0
    pair_info = []
    for pair in train:
        s_in, tie_in = find_shapes(pair["input"])
        s_out, tie_out = find_shapes(pair["output"])
        ties += int(tie_in) + int(tie_out)
        pair_info.append((s_in, s_out))

    count_group, count_reason = group_of([count_change(a, b) for a, b in pair_info])
    form_group, form_reason = group_of([form_change(a, b) for a, b in pair_info])

    # Forms that appear in EVERY training input and EVERY training output.
    form_sets = []
    for s_in, s_out in pair_info:
        form_sets.append({s["form"] for s in s_in})
        form_sets.append({s["form"] for s in s_out})
    shared = set.intersection(*form_sets)

    # Test pairs (only the ones with a public answer)
    test_count_kinds, test_form_kinds, test_has_shared = [], [], []
    for t in task.get("test", []):
        if "output" not in t:
            continue
        s_in, _ = find_shapes(t["input"])
        s_out, _ = find_shapes(t["output"])
        test_count_kinds.append(count_change(s_in, s_out))
        test_form_kinds.append(form_change(s_in, s_out))
        test_has_shared.append(bool(shared & {s["form"] for s in s_in})
                               and bool(shared & {s["form"] for s in s_out}))

    # Test inputs (do not need an answer)
    test_input_counts = [len(find_shapes(t["input"])[0])
                         for t in task.get("test", []) if "input" in t]

    return {
        "count_group": count_group, "count_reason": count_reason,
        "count_test": test_check(count_group, test_count_kinds),
        "form_group": form_group, "form_reason": form_reason,
        "form_test": test_check(form_group, test_form_kinds),
        "shared_forms": len(shared),
        "shared_forms_in_test": shared_forms_check(shared, test_has_shared),
        "per_pair_counts": " | ".join(f"{len(a)}>{len(b)}" for a, b in pair_info),
        "test_input_shapes": " | ".join(str(n) for n in test_input_counts),
        "test_vs_train_inputs": test_input_vs_training(
            [len(a) for a, _ in pair_info], test_input_counts),
        "bg_ties": ties,
    }


def analyse_dataset(tasks):
    results = []
    for task_id, task in tasks.items():
        row = analyse_task(task)
        row["task_id"] = task_id
        results.append(row)
    return results


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def rate_text(holds, breaks):
    return f"{holds / (holds + breaks):.1%}" if (holds + breaks) else "-"


def print_group_table(title, results, group_key, test_key, order):
    print(f"\n{title}")
    header = f"{'Group':<14}{'Tasks':>7}{'Holds':>8}{'Breaks':>8}{'n/a':>6}{'Holds (when checked)':>24}"
    print(header)
    print("-" * len(header))
    by_group = defaultdict(list)
    for r in results:
        by_group[r[group_key]].append(r)
    for group in order:
        rows = by_group.get(group, [])
        if not rows:
            continue
        holds = sum(1 for r in rows if r[test_key] == "holds")
        breaks = sum(1 for r in rows if r[test_key] == "breaks")
        print(f"{group:<14}{len(rows):>7}{holds:>8}{breaks:>8}"
              f"{len(rows) - holds - breaks:>6}{rate_text(holds, breaks):>24}")


def print_dataset(name, results):
    print(f"\n=== {name}: shapes ({len(results)} tasks) ===")
    print_group_table("Rule A: shape count (does the training pattern hold on the test pair?)",
                      results, "count_group", "count_test",
                      ["added", "removed", "same", "other"])
    print_group_table("Rule B: shape forms (does the training pattern hold on the test pair?)",
                      results, "form_group", "form_test",
                      ["same_forms", "kept_some", "none_shared", "other"])

    with_shared = [r for r in results if r["shared_forms"] > 0]
    holds = sum(1 for r in with_shared if r["shared_forms_in_test"] == "holds")
    breaks = sum(1 for r in with_shared if r["shared_forms_in_test"] == "breaks")
    print(f"\nTasks with a form present in every training input and output: "
          f"{len(with_shared)}/{len(results)}")
    print(f"  Same kind of shape also in the test pair: holds {holds}, "
          f"breaks {breaks} ({rate_text(holds, breaks)})")

    where = Counter(r["test_vs_train_inputs"] for r in results)
    print("\nShapes in the test input vs the training inputs: "
          + ", ".join(f"{k} {where[k]}" for k in ("within", "more", "fewer", "mixed", "n/a")))
    print(f"Grids with a tied background (training pairs): "
          f"{sum(r['bg_ties'] for r in results)}")


def write_csv(all_results):
    columns = ["dataset", "task_id", "count_group", "count_reason", "count_test",
               "form_group", "form_reason", "form_test", "shared_forms",
               "shared_forms_in_test", "per_pair_counts", "test_input_shapes",
               "test_vs_train_inputs", "bg_ties"]
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for dataset, results in all_results.items():
            for r in results:
                row = dict(r, dataset=dataset)
                writer.writerow([row[c] for c in columns])
    print(f"\nPer-task results written to {RESULTS_PATH}")


# ----------------------------------------------------------------------
def main():
    all_results = {}
    for name in ("training", "evaluation"):
        all_results[name] = analyse_dataset(load_tasks(DATA_DIR / name))
        print_dataset(name, all_results[name])
    write_csv(all_results)


if __name__ == "__main__":
    main()