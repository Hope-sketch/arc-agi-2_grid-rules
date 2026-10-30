"""
Step 6: Grid transformations, version 2 (investigation only, not used to solve yet)

The transformation is LEARNED from the training pairs, never hard-coded by colour
or shape. Which probe applies depends on what the earlier steps found:

  Probe 1  LOCAL RULE   (tasks where every training pair keeps the same size)
      Learn what each cell becomes from its neighbourhood. Neighbourhood kinds,
      simplest first:
        centre      : the cell's own colour only (a colour map)
        cross_rel   : own colour + which of the 4 side neighbours match it
        block3_rel  : own colour + which of the 8 neighbours match it
        cross       : own colour + the 4 side neighbours' colours
        block3      : own colour + the 8 neighbours' colours
      "consistent" = the same neighbourhood never gives two different outputs.
      Test check: how many test-input cells have a neighbourhood seen in training
      (coverage) and does the rule reproduce the public test output?
        exact          : every cell covered and the output matches
        exact_fallback : some cells not covered (kept as they are) and it matches
        wrong          : it does not match
  Probe 2  CONTAINED    (tasks where every training output fits inside its input)
      Is the output an exact piece of the input (or a flipped / rotated piece)?
      Also: is that piece the bounding box of everything that is not background?
      Test check: is the test output also a piece of the test input?
  Probe 3  WHOLE GRID   (every task, the old fixed library in whole_grid_library.py)
      Flips, rotations, mirrors, tiling, scaling, crop, halves, with one learned
      colour map. FOUND = reproduces every training output.

Groups (size word, palette word, shape word) come from the earlier steps, so the
tables show which kind of transformation works for which kind of task.

Usage
  python solver/grid_transformations/grid_transformations.py

Outputs
  * tables in the terminal
  * solver/grid_transformations/results/grid_transformations_results.csv
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from whole_grid_library import crop_non_background, find_whole_grid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (holds data/)
DATA_DIR = ROOT / "data"
GROUPS_CSV = ROOT / "solver" / "grid_shapes" / "results" / "grid_groups_shapes_results.csv"
CELLS_CSV = ROOT / "solver" / "grid_cells" / "results" / "grid_cells_results.csv"
OUT_CSV = HERE / "results" / "grid_transformations_results.csv"

NEIGHBOUR_KINDS = ["centre", "cross_rel", "block3_rel", "cross", "block3"]

DIHEDRAL = {
    "same":           lambda g: g,
    "flip_h":         lambda g: g[:, ::-1],
    "flip_v":         lambda g: g[::-1, :],
    "rotate_90":      lambda g: np.rot90(g, -1),
    "rotate_180":     lambda g: np.rot90(g, 2),
    "rotate_270":     lambda g: np.rot90(g, 1),
    "transpose":      lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, 2).T,
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


def read_lookup(path, label):
    if not path.exists():
        print(f"Note: {label} not found, its columns stay empty ({path})")
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return {(r["dataset"], r["task_id"]): r for r in csv.DictReader(file)}


def to_np(grid):
    return np.array(grid, dtype=int)


# ----------------------------------------------------------------------
# Probe 1: local rule (same-size tasks)
# ----------------------------------------------------------------------
def neighbourhood_keys(g, kind):
    """One key (a tuple) per cell, in row order."""
    p = np.pad(g, 1, constant_values=-1)      # -1 = outside the grid
    sides = [p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]
    corners = [p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:]]
    if kind == "centre":
        parts = [g]
    elif kind == "cross_rel":
        parts = [g] + [n == g for n in sides]
    elif kind == "block3_rel":
        parts = [g] + [n == g for n in sides + corners]
    elif kind == "cross":
        parts = [g] + sides
    else:
        parts = [g] + sides + corners
    return list(zip(*[a.ravel().tolist() for a in parts]))


def learn_local_rule(pairs, kind):
    """neighbourhood -> output colour, or None if it is not consistent."""
    rule = {}
    for grid_in, grid_out in pairs:
        for key, out in zip(neighbourhood_keys(grid_in, kind), grid_out.ravel().tolist()):
            if rule.setdefault(key, out) != out:
                return None
    return rule


def local_probe(train, tests):
    blank = {"local_kind": "n/a", "local_rule_size": "", "local_coverage": "",
             "local_test": "n/a"}
    if not all(i.shape == o.shape for i, o in train):
        return blank
    blank["local_kind"] = "none"
    for kind in NEIGHBOUR_KINDS:
        rule = learn_local_rule(train, kind)
        if rule is not None:
            break
    else:
        return blank

    results, coverages = [], []
    for grid_in, grid_out in tests:
        if grid_out is None:
            continue
        keys = neighbourhood_keys(grid_in, kind)
        covered = [k in rule for k in keys]
        coverages.append(sum(covered) / len(covered))
        predicted = np.array([rule.get(k, c) for k, c in
                              zip(keys, grid_in.ravel().tolist())]).reshape(grid_in.shape)
        matches = predicted.shape == grid_out.shape and np.array_equal(predicted, grid_out)
        results.append((matches, all(covered)))

    if not results:
        test = "n/a"
    elif not all(m for m, _ in results):
        test = "wrong"
    elif all(c for _, c in results):
        test = "exact"
    else:
        test = "exact_fallback"
    return {"local_kind": kind, "local_rule_size": len(rule),
            "local_coverage": f"{100 * min(coverages):.0f}" if coverages else "",
            "local_test": test}


# ----------------------------------------------------------------------
# Probe 2: output is a piece of the input
# ----------------------------------------------------------------------
def occurs(big, small):
    if small.shape[0] > big.shape[0] or small.shape[1] > big.shape[1]:
        return False
    windows = sliding_window_view(big, small.shape)
    return bool((windows == small).all(axis=(2, 3)).any())


def contained_probe(train, tests):
    blank = {"contained": "n/a", "contained_where": "", "contained_test": "n/a"}
    if all(i.shape == o.shape for i, o in train):
        return blank                                    # same-size task, probe 1's job
    if not all(o.shape[0] <= i.shape[0] and o.shape[1] <= i.shape[1] for i, o in train):
        return blank

    common = None
    for grid_in, grid_out in train:
        names = {n for n, t in DIHEDRAL.items() if occurs(grid_in, t(grid_out))}
        common = names if common is None else common & names
    if not common:
        blank["contained"] = "none"
        return blank

    name = next(n for n in DIHEDRAL if n in common)
    transform = DIHEDRAL[name]
    in_bbox = 0
    for grid_in, grid_out in train:
        crop = crop_non_background(grid_in)
        if crop is not None and crop.shape == grid_out.shape and np.array_equal(crop, transform(grid_out)):
            in_bbox += 1
    where = "bbox_crop" if in_bbox == len(train) else "elsewhere"

    checks = [occurs(grid_in, transform(grid_out)) for grid_in, grid_out in tests
              if grid_out is not None]
    test = "n/a" if not checks else ("holds" if all(checks) else "breaks")
    return {"contained": name, "contained_where": where, "contained_test": test}


# ----------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------
def analyse_task(task):
    train = [(to_np(p["input"]), to_np(p["output"])) for p in task["train"]]
    tests = [(to_np(p["input"]), to_np(p["output"]) if "output" in p else None)
             for p in task["test"]]

    row = {}
    row.update(local_probe(train, tests))
    row.update(contained_probe(train, tests))
    whole = find_whole_grid(train, tests)
    row["whole_grid"] = whole["category"]
    row["whole_grid_choice"] = whole["chosen"]
    row["whole_grid_test"] = whole["test_correct"]

    # Reproduces the public test output = a check on the public data, not a score.
    row["reproduces_test"] = (row["local_test"] in ("exact", "exact_fallback")
                              or whole["test_correct"] is True)
    return row


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def pct(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def probe_counts(rows):
    local = [r for r in rows if r["local_kind"] != "n/a"]
    contained = [r for r in rows if r["contained"] != "n/a"]
    return {
        "tasks": len(rows),
        "local_applicable": len(local),
        "local_consistent": sum(1 for r in local if r["local_kind"] != "none"),
        "local_exact": sum(1 for r in local if r["local_test"] == "exact"),
        "local_fallback": sum(1 for r in local if r["local_test"] == "exact_fallback"),
        "local_wrong": sum(1 for r in local if r["local_test"] == "wrong"),
        "contained_applicable": len(contained),
        "contained_all": sum(1 for r in contained if r["contained"] != "none"),
        "contained_holds": sum(1 for r in contained if r["contained_test"] == "holds"),
        "contained_bbox": sum(1 for r in contained if r["contained_where"] == "bbox_crop"),
        "whole_found": sum(1 for r in rows if r["whole_grid"] in ("found_exact", "found_with_colour_map")),
        "whole_right": sum(1 for r in rows if r["whole_grid_test"] is True),
        "reproduces": sum(1 for r in rows if r["reproduces_test"]),
    }


def print_overall(name, rows):
    c = probe_counts(rows)
    print(f"\n=== {name}: transformations ({c['tasks']} tasks) ===")
    print(f"Probe 1 local rule   (same-size tasks: {c['local_applicable']})")
    print(f"   consistent rule found : {c['local_consistent']:>4}  ({pct(c['local_consistent'], c['local_applicable'])})")
    print(f"   reproduces test       : exact {c['local_exact']}, exact_fallback {c['local_fallback']}, "
          f"wrong {c['local_wrong']}")
    print(f"Probe 2 contained    (output fits inside input: {c['contained_applicable']})")
    print(f"   output is a piece     : {c['contained_all']:>4}  ({pct(c['contained_all'], c['contained_applicable'])})"
          f"   of those the bounding box of non-background: {c['contained_bbox']}")
    print(f"   also a piece on test  : {c['contained_holds']:>4}")
    print(f"Probe 3 whole grid   (all tasks)")
    print(f"   fits every train pair : {c['whole_found']:>4}  ({pct(c['whole_found'], c['tasks'])}),"
          f" right on the test pair: {c['whole_right']}")
    print(f"Tasks where a probe reproduces the public test output: {c['reproduces']} "
          f"({pct(c['reproduces'], c['tasks'])})")


def print_by(title, rows, key, min_tasks=1):
    print(f"\n{title}")
    header = (f"{'':<14}{'Tasks':>6}{'Local ok':>10}{'Local test':>12}"
              f"{'Contained':>11}{'Cont.test':>11}{'WholeGrid':>11}{'Any':>6}")
    print(header)
    print("-" * len(header))
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    for value, group_rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(group_rows) < min_tasks:
            continue
        c = probe_counts(group_rows)
        local = f"{c['local_consistent']}/{c['local_applicable']}" if c["local_applicable"] else "-"
        local_test = (f"{c['local_exact'] + c['local_fallback']}" if c["local_applicable"] else "-")
        cont = f"{c['contained_all']}/{c['contained_applicable']}" if c["contained_applicable"] else "-"
        cont_test = f"{c['contained_holds']}" if c["contained_applicable"] else "-"
        print(f"{value or '(none)':<14}{c['tasks']:>6}{local:>10}{local_test:>12}"
              f"{cont:>11}{cont_test:>11}{c['whole_found']:>11}{c['reproduces']:>6}")


def write_csv(all_rows):
    columns = ["dataset", "task_id", "group", "shape_word", "cell_group",
               "local_kind", "local_rule_size", "local_coverage", "local_test",
               "contained", "contained_where", "contained_test",
               "whole_grid", "whole_grid_choice", "whole_grid_test", "reproduces_test"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in all_rows:
            writer.writerow([row[c] for c in columns])
    print(f"\nPer-task results written to {OUT_CSV}")


def main():
    groups = read_lookup(GROUPS_CSV, "grid_groups_shapes_results.csv")
    cells = read_lookup(CELLS_CSV, "grid_cells_results.csv")

    all_rows = []
    for dataset in ("training", "evaluation"):
        rows = []
        for task_id, task in load_tasks(DATA_DIR / dataset).items():
            row = analyse_task(task)
            row["dataset"] = dataset
            row["task_id"] = task_id
            g = groups.get((dataset, task_id), {})
            row["group"] = g.get("group", "")
            row["shape_word"] = g.get("shape_word", "")
            row["cell_group"] = cells.get((dataset, task_id), {}).get("cell_group", "")
            rows.append(row)
        print_overall(dataset, rows)
        print_by("By size-palette group (groups with 20+ tasks)", rows, "group", min_tasks=20)
        print_by("By shape word", rows, "shape_word")
        all_rows += rows

    write_csv(all_rows)


if __name__ == "__main__":
    main()