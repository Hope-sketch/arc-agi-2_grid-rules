"""
Step 3: Grid shapes (investigation only, not used to solve yet)

What counts as a shape
  A shape is a group of cells of the SAME colour that touch each other.
  The background colour is not a shape.
  A cell with no same-coloured neighbour is a shape on its own (size 1).
  Two shapes of the same colour that do not touch are two shapes.

Connectivity (one setting, run the script both ways and compare)
  4 : cells touch only side by side
  8 : cells also touch diagonally (corners)
  Usage:  python solver/grid_shapes.py 8      (default is 8)
          python solver/grid_shapes.py 4

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

Background
  Background = the most frequent colour in a grid. If two colours tie, the
  tied colour whose biggest connected region is largest is used, and the
  grid is counted in "bg_ties" so you can see how often this happens.

Outputs
  * tables in the terminal
  * results/grid_shapes_results_<4 or 8>.csv  (one row per task)
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
CONNECTIVITY = 8  # 4 or 8, can be overridden on the command line

NEIGHBOURS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBOURS_8 = NEIGHBOURS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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
def find_regions(grid, connectivity):
    """Every connected same-colour region, background included."""
    offsets = NEIGHBOURS_8 if connectivity == 8 else NEIGHBOURS_4
    height = len(grid)
    width = len(grid[0]) if grid else 0
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
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr][nc] and grid[nr][nc] == colour):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            regions.append({"colour": colour, "cells": cells})
    return regions


def background_colour(grid, regions):
    """Most frequent colour. Returns (colour, was_tie)."""
    counts = Counter(cell for row in grid for cell in row)
    if not counts:
        return None, False
    ranked = counts.most_common()
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


def find_shapes(grid, connectivity):
    """Returns (shapes, was_tie). Each shape: colour, size, form."""
    if not grid or not grid[0]:
        return [], False
    regions = find_regions(grid, connectivity)
    bg, was_tie = background_colour(grid, regions)
    shapes = []
    for reg in regions:
        if reg["colour"] == bg:
            continue
        shapes.append({"colour": reg["colour"],
                       "size": len(reg["cells"]),
                       "form": form_of(reg["cells"])})
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


def analyse_task(task, connectivity):
    train = task.get("train", [])
    test = [t for t in task.get("test", []) if "output" in t]

    if not train:
        return {"count_group": "none", "count_reason": "no_training_examples",
                "count_test": "n/a", "form_group": "none",
                "form_reason": "no_training_examples", "form_test": "n/a",
                "shared_forms": 0, "shared_forms_in_test": "n/a",
                "bg_ties": 0, "per_pair_counts": ""}

    ties = 0
    pair_info = []
    for p in train:
        s_in, t1 = find_shapes(p["input"], connectivity)
        s_out, t2 = find_shapes(p["output"], connectivity)
        ties += int(t1) + int(t2)
        pair_info.append((s_in, s_out))

    count_kinds = [count_change(a, b) for a, b in pair_info]
    form_kinds = [form_change(a, b) for a, b in pair_info]

    # Forms that appear in EVERY training input and EVERY training output.
    form_sets = []
    for s_in, s_out in pair_info:
        form_sets.append({s["form"] for s in s_in})
        form_sets.append({s["form"] for s in s_out})
    shared = set.intersection(*form_sets) if form_sets else set()

    test_count_kinds, test_form_kinds, test_has_shared = [], [], []
    for t in test:
        s_in, _ = find_shapes(t["input"], connectivity)
        s_out, _ = find_shapes(t["output"], connectivity)
        test_count_kinds.append(count_change(s_in, s_out))
        test_form_kinds.append(form_change(s_in, s_out))
        test_has_shared.append(bool(shared & {s["form"] for s in s_in})
                               and bool(shared & {s["form"] for s in s_out}))

    cg, c_reason = group_of(count_kinds)
    fg, f_reason = group_of(form_kinds)

    if not shared:
        shared_test = "n/a"
    elif not test_has_shared:
        shared_test = "n/a"
    else:
        shared_test = "holds" if all(test_has_shared) else "breaks"

    return {
        "count_group": cg, "count_reason": c_reason,
        "count_test": test_check(cg, test_count_kinds),
        "form_group": fg, "form_reason": f_reason,
        "form_test": test_check(fg, test_form_kinds),
        "shared_forms": len(shared), "shared_forms_in_test": shared_test,
        "bg_ties": ties,
        "per_pair_counts": " | ".join(f"{len(a)}>{len(b)}" for a, b in pair_info),
    }


def analyse_dataset(tasks, connectivity):
    results = []
    for task_id, task in tasks.items():
        row = analyse_task(task, connectivity)
        row["task_id"] = task_id
        results.append(row)
    return results


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
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
        h = sum(1 for r in rows if r[test_key] == "holds")
        b = sum(1 for r in rows if r[test_key] == "breaks")
        n = len(rows) - h - b
        rate = f"{h / (h + b):.1%}" if (h + b) else "-"
        print(f"{group:<14}{len(rows):>7}{h:>8}{b:>8}{n:>6}{rate:>24}")


def print_dataset(name, results, connectivity):
    print(f"\n=== {name}: shapes, {connectivity}-connectivity ({len(results)} tasks) ===")
    print_group_table("Rule A: shape count (does the training pattern hold on the test pair?)",
                      results, "count_group", "count_test",
                      ["added", "removed", "same", "other"])
    print_group_table("Rule B: shape forms (does the training pattern hold on the test pair?)",
                      results, "form_group", "form_test",
                      ["same_forms", "kept_some", "none_shared", "other"])

    with_shared = [r for r in results if r["shared_forms"] > 0]
    h = sum(1 for r in with_shared if r["shared_forms_in_test"] == "holds")
    b = sum(1 for r in with_shared if r["shared_forms_in_test"] == "breaks")
    rate = f"{h / (h + b):.1%}" if (h + b) else "-"
    print(f"\nTasks with a form present in every training input and output: "
          f"{len(with_shared)}/{len(results)}")
    print(f"  Same kind of shape also in the test pair: holds {h}, breaks {b} ({rate})")

    ties = sum(r["bg_ties"] for r in results)
    print(f"Grids with a tied background (training pairs): {ties}")


def write_csv(all_results, connectivity):
    out_path = Path("results") / f"grid_shapes_results_{connectivity}.csv"
    out_path.parent.mkdir(exist_ok=True)
    columns = ["dataset", "task_id", "count_group", "count_reason", "count_test",
               "form_group", "form_reason", "form_test", "shared_forms",
               "shared_forms_in_test", "bg_ties", "per_pair_counts"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for dataset, results in all_results.items():
            for r in results:
                writer.writerow([dataset] + [r[c] for c in columns[1:]])
    print(f"\nPer-task results written to {out_path}")


# ----------------------------------------------------------------------
def main():
    connectivity = CONNECTIVITY
    if len(sys.argv) > 1:
        if sys.argv[1] not in ("4", "8"):
            print("Use 4 or 8, for example: python solver/grid_shapes.py 8")
            return
        connectivity = int(sys.argv[1])

    datasets = {
        "training": load_tasks(DATA_DIR / "training"),
        "evaluation": load_tasks(DATA_DIR / "evaluation"),
    }
    all_results = {}
    for name, tasks in datasets.items():
        all_results[name] = analyse_dataset(tasks, connectivity)
        print_dataset(name, all_results[name], connectivity)

    write_csv(all_results, connectivity)


if __name__ == "__main__":
    main()
