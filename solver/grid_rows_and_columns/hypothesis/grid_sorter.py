"""
Step 7 (part 1): the Grid Sorter (investigation only, not used to solve yet)

Joins the per-task results of every step into ONE profile per task and sorts the
task into a named group, one word per step, most general first:

  level 1  SIZE     grid sizes       STAY / FIXED / FREE / FUNNEL / SPREAD
  level 2  PALETTE  grid colours     KEEP / GAIN / DROP / SWAP / MIXED
  level 3  SHAPE    grid shapes      MORE / FEWER / EVEN / VARY
  level 4  DETAIL   the step that can compare this task:
                      same-size tasks -> grid cells         ADD / REMOVE / RECOLOUR / MIXED / OTHER
                      size-changing   -> grid rows&columns  SAME / GROW / SHRINK / BOTH / OTHER
                    (GROW = a line count is a whole multiple, SHRINK = a whole fraction)

A group name is the words joined, for example  STAY-GAIN-MORE-ADD  or  FREE-DROP-FEWER-SHRINK.

What it answers
  1. FUNNEL: how fast do groups get too small to learn from as the name gets longer?
     (groups with at least MIN_TASKS training tasks are "usable")
  2. BACKOFF: for each evaluation task, the longest group name that still has MIN_TASKS
     training tasks. The Grid Solver would trust that group's evidence.
  3. EVIDENCE: for the usable groups, which transformation probes work on their
     training tasks (from grid_transformations.py).

Reads (each is written by its own step; missing files are reported)
  solver/grid_shapes/results/grid_groups_shapes_results.csv
  solver/grid_cells/results/grid_cells_results.csv
  solver/grid_rows_and_columns/results/grid_rows_columns_results.csv
  solver/grid_transformations/results/grid_transformations_results.csv

Outputs
  * tables in the terminal
  * solver/grid_solver/results/grid_profiles.csv   one row per task

Usage
  python solver/grid_solver/grid_sorter.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parent
OUT_CSV = HERE / "results" / "grid_profiles.csv"

SOURCES = {
    "groups": SOLVER / "grid_shapes" / "results" / "grid_groups_shapes_results.csv",
    "cells": SOLVER / "grid_cells" / "results" / "grid_cells_results.csv",
    "lines": SOLVER / "grid_rows_and_columns" / "results" / "grid_rows_columns_results.csv",
    "transform": SOLVER / "grid_transformations" / "results" / "grid_transformations_results.csv",
}

MIN_TASKS = 10
LEVELS = ["SIZE", "PALETTE", "SHAPE", "DETAIL"]
CELL_WORD = {"added_only": "ADD", "removed_only": "REMOVE", "recoloured_only": "RECOLOUR",
             "mixed": "MIXED", "identical": "NONE", "other": "OTHER"}


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def read_lookup(name):
    path = SOURCES[name]
    if not path.exists():
        print(f"Note: {path.name} not found, its columns stay empty ({path})")
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return {(r["dataset"], r["task_id"]): r for r in csv.DictReader(file)}


def lines_word(rows_ratio, columns_ratio):
    ratios = [rows_ratio, columns_ratio]
    if all(r == "same" for r in ratios):
        return "SAME"
    grow = any(r.startswith("x") for r in ratios)
    shrink = any(r.startswith("/") for r in ratios)
    if grow and shrink:
        return "BOTH"
    return "GROW" if grow else "SHRINK" if shrink else "OTHER"


def detail_word(cells, lines):
    """Cells step for same-size tasks, rows & columns step for the others."""
    if cells.get("applicable") == "yes":
        return CELL_WORD.get(cells["kind"], "OTHER")
    if lines:
        return lines_word(lines["rows_ratio"], lines["columns_ratio"])
    return "UNKNOWN"


def build_profiles():
    lookups = {name: read_lookup(name) for name in SOURCES}
    profiles = []
    for (dataset, task_id), g in lookups["groups"].items():
        size_word, palette_word = g["group"].split("-", 1)
        cells = lookups["cells"].get((dataset, task_id), {})
        lines = lookups["lines"].get((dataset, task_id), {})
        trans = lookups["transform"].get((dataset, task_id), {})
        words = [size_word, palette_word, g["shape_word"], detail_word(cells, lines)]
        names = ["-".join(words[:n]) for n in range(1, len(words) + 1)]
        profiles.append({
            "dataset": dataset, "task_id": task_id,
            "size": words[0], "palette": words[1], "shape": words[2], "detail": words[3],
            "level1": names[0], "level2": names[1], "level3": names[2], "level4": names[3],
            "to": cells.get("to", ""), "where": cells.get("where_tightest", ""),
            "rows_story": lines.get("rows_story", ""), "columns_story": lines.get("columns_story", ""),
            "local_kind": trans.get("local_kind", ""), "local_test": trans.get("local_test", ""),
            "contained": trans.get("contained", ""), "whole_grid": trans.get("whole_grid", ""),
            "reproduces_test": trans.get("reproduces_test", ""),
        })
    return profiles


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def pct(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


def print_funnel(train, evaluation):
    print(f"\nFUNNEL: usable groups have at least {MIN_TASKS} training tasks")
    header = (f"{'Level':<24}{'Groups':>8}{'Usable':>8}{'Train tasks in usable':>23}"
              f"{'Eval tasks in usable':>22}")
    print(header)
    print("-" * len(header))
    for n, level in enumerate(LEVELS, start=1):
        key = f"level{n}"
        counts = Counter(p[key] for p in train)
        usable = {name for name, c in counts.items() if c >= MIN_TASKS}
        in_train = sum(c for name, c in counts.items() if name in usable)
        in_eval = sum(1 for p in evaluation if p[key] in usable)
        label = f"{n}: {'-'.join(LEVELS[:n]).lower()}"
        print(f"{label:<24}{len(counts):>8}{len(usable):>8}"
              f"{in_train:>9} ({pct(in_train, len(train)):>4}){in_eval:>11} ({pct(in_eval, len(evaluation)):>4})")


def backoff_level(task, usable_by_level):
    for n in range(len(LEVELS), 0, -1):
        if task[f"level{n}"] in usable_by_level[n]:
            return n
    return 0


def print_backoff(train, evaluation):
    usable = {}
    for n in range(1, len(LEVELS) + 1):
        counts = Counter(p[f"level{n}"] for p in train)
        usable[n] = {name for name, c in counts.items() if c >= MIN_TASKS}
    print(f"\nBACKOFF: the longest group name with {MIN_TASKS}+ training tasks")
    for dataset, tasks in (("training", train), ("evaluation", evaluation)):
        levels = Counter(backoff_level(t, usable) for t in tasks)
        parts = ", ".join(f"level {n}: {levels[n]}" for n in range(0, len(LEVELS) + 1) if levels[n])
        print(f"  {dataset:<11}{parts}")
    return usable


def print_evidence(train, usable):
    n = len(LEVELS)
    print(f"\nEVIDENCE: usable level-{n} groups, what works on their training tasks")
    header = (f"{'Group':<26}{'Tasks':>6}{'Local ok':>10}{'Piece':>7}{'WholeGrid':>11}{'Reproduced':>12}")
    print(header)
    print("-" * len(header))
    by = defaultdict(list)
    for p in train:
        if p[f"level{n}"] in usable[n]:
            by[p[f"level{n}"]].append(p)
    for name, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        local = sum(1 for r in rows if r["local_kind"] not in ("", "n/a", "none"))
        piece = sum(1 for r in rows if r["contained"] not in ("", "n/a", "none"))
        whole = sum(1 for r in rows if r["whole_grid"] in ("found_exact", "found_with_colour_map"))
        solved = sum(1 for r in rows if r["reproduces_test"] == "True")
        print(f"{name:<26}{len(rows):>6}{local:>10}{piece:>7}{whole:>11}{solved:>12}")


def write_csv(profiles, usable):
    columns = ["dataset", "task_id", "size", "palette", "shape", "detail",
               "level1", "level2", "level3", "level4", "backoff_level",
               "to", "where", "rows_story", "columns_story", "local_kind", "local_test",
               "contained", "whole_grid", "reproduces_test"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for p in profiles:
            row = dict(p, backoff_level=backoff_level(p, usable))
            writer.writerow([row[c] for c in columns])
    print(f"\nPer-task profiles written to {OUT_CSV}")


def main():
    profiles = build_profiles()
    if not profiles:
        raise SystemExit("No groups file found: run grid_groups.py first.")
    train = [p for p in profiles if p["dataset"] == "training"]
    evaluation = [p for p in profiles if p["dataset"] == "evaluation"]
    print(f"\nProfiles built: {len(train)} training, {len(evaluation)} evaluation tasks")

    print_funnel(train, evaluation)
    usable = print_backoff(train, evaluation)
    print_evidence(train, usable)
    write_csv(profiles, usable)


if __name__ == "__main__":
    main()