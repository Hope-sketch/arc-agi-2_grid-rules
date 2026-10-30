"""
Step 5: Grid Rows & Columns (investigation only)

For every training pair of every task:
  - split input and output into ROWS, and into COLUMNS
  - compare input row i with output row i (and input column j with output column j)
  - sort every difference into one of three groups:

      removed : a colour was in the input at that coordinate, and is gone in the output
                (the output has the background colour there, or the cell no longer exists)
      changed : a colour was in the input at that coordinate, and the output has a
                different (non-background) colour at the same coordinate
      added   : the input had nothing (background) at that coordinate, and the output
                now has a colour there

Background colour = most frequent colour in the INPUT grid (no colour is hard-coded).

Writes:
  results/grid_rows_columns_results.csv   one row per task (summary)
  results/grid_rows_columns_details.csv   one row per (pair, axis, line, group) with
                                          the colours and their coordinates

Usage:
  python grid_rows_columns.py <path>
  <path> = a folder of task .json files, or one challenges .json file
"""

import csv
import json
import os
import sys
from collections import Counter

DATA_PATH = "data/arc-agi_training_challenges.json"
RESULTS_DIR = "results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "grid_rows_columns_results.csv")
DETAILS_CSV = os.path.join(RESULTS_DIR, "grid_rows_columns_details.csv")

GROUPS = ["removed", "changed", "added"]


def load_tasks(path):
    """Returns {task_id: task_dict}. Accepts a folder of .json files or one .json file."""
    tasks = {}
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(path, name), "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "train" in data:
                tasks[os.path.splitext(name)[0]] = data
            elif isinstance(data, dict):
                for tid, t in data.items():
                    if isinstance(t, dict) and "train" in t:
                        tasks[tid] = t
    else:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "train" in data:
            tasks[os.path.splitext(os.path.basename(path))[0]] = data
        else:
            for tid, t in data.items():
                if isinstance(t, dict) and "train" in t:
                    tasks[tid] = t
    return tasks


def background_colour(grid):
    """Most frequent colour in the grid."""
    counts = Counter(c for row in grid for c in row)
    return counts.most_common(1)[0][0]


def transpose(grid):
    return [list(col) for col in zip(*grid)]


def compare_lines(in_lines, out_lines, bg, axis):
    """
    Compare input lines with output lines (lines = rows or columns).
    axis = "row"    -> line index is the row, position inside the line is the column
    axis = "column" -> line index is the column, position inside the line is the row
    Coordinates are always reported as (row, col).
    """
    results = []
    for i in range(max(len(in_lines), len(out_lines))):
        a_line = in_lines[i] if i < len(in_lines) else None
        b_line = out_lines[i] if i < len(out_lines) else None
        len_a = len(a_line) if a_line is not None else 0
        len_b = len(b_line) if b_line is not None else 0

        found = {g: [] for g in GROUPS}

        for j in range(max(len_a, len_b)):
            ca = a_line[j] if a_line is not None and j < len_a else None
            cb = b_line[j] if b_line is not None and j < len_b else None
            if ca == cb:
                continue

            coord = (i, j) if axis == "row" else (j, i)

            if cb is None:
                if ca != bg:
                    found["removed"].append((coord, ca, None))
            elif ca is None:
                if cb != bg:
                    found["added"].append((coord, None, cb))
            elif ca != bg and cb == bg:
                found["removed"].append((coord, ca, None))
            elif ca == bg and cb != bg:
                found["added"].append((coord, None, cb))
            else:
                found["changed"].append((coord, ca, cb))

        for g in GROUPS:
            if found[g]:
                results.append({"axis": axis, "line": i, "group": g, "cells": found[g]})
    return results


def cells_to_text(group, cells):
    parts = []
    for (r, c), ca, cb in cells:
        if group == "removed":
            parts.append(f"({r},{c}):{ca}")
        elif group == "added":
            parts.append(f"({r},{c}):{cb}")
        else:
            parts.append(f"({r},{c}):{ca}>{cb}")
    return ";".join(parts)


def colours_of(group, cells):
    if group == "removed":
        return sorted({ca for _, ca, _ in cells})
    if group == "added":
        return sorted({cb for _, _, cb in cells})
    return sorted({(ca, cb) for _, ca, cb in cells})


def pair_label(groups_present):
    ordered = [g for g in GROUPS if g in groups_present]
    return "+".join(ordered) if ordered else "none"


def task_label(pair_labels):
    return pair_labels[0] if len(set(pair_labels)) == 1 else "mixed"


def analyse_task(task_id, task):
    detail_rows = []
    row_pair_labels, col_pair_labels = [], []
    totals = {ax: {g: 0 for g in GROUPS} for ax in ("row", "column")}
    lines_touched = {"row": 0, "column": 0}
    same_size_all = True

    for p_idx, pair in enumerate(task["train"]):
        grid_in, grid_out = pair["input"], pair["output"]
        bg = background_colour(grid_in)
        if (len(grid_in), len(grid_in[0])) != (len(grid_out), len(grid_out[0])):
            same_size_all = False

        row_res = compare_lines(grid_in, grid_out, bg, "row")
        col_res = compare_lines(transpose(grid_in), transpose(grid_out), bg, "column")

        for axis, res, labels in (("row", row_res, row_pair_labels),
                                  ("column", col_res, col_pair_labels)):
            labels.append(pair_label({r["group"] for r in res}))
            lines_touched[axis] += len({r["line"] for r in res})
            for r in res:
                totals[axis][r["group"]] += len(r["cells"])
                detail_rows.append({
                    "task_id": task_id,
                    "pair": p_idx,
                    "background": bg,
                    "axis": axis,
                    "line_index": r["line"],
                    "group": r["group"],
                    "n_cells": len(r["cells"]),
                    "colours": str(colours_of(r["group"], r["cells"])),
                    "cells": cells_to_text(r["group"], r["cells"]),
                })

    summary = {
        "task_id": task_id,
        "n_train_pairs": len(task["train"]),
        "same_size_all_pairs": same_size_all,
        "rows_label": task_label(row_pair_labels),
        "columns_label": task_label(col_pair_labels),
        "rows_pair_labels": "|".join(row_pair_labels),
        "columns_pair_labels": "|".join(col_pair_labels),
        "rows_touched_total": lines_touched["row"],
        "columns_touched_total": lines_touched["column"],
    }
    for ax, name in (("row", "rows"), ("column", "columns")):
        for g in GROUPS:
            summary[f"{name}_{g}_cells"] = totals[ax][g]
    return summary, detail_rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    tasks = load_tasks(path)
    if not tasks:
        print(f"No tasks found at: {path}")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summaries, all_details = [], []
    for task_id, task in tasks.items():
        s, d = analyse_task(task_id, task)
        summaries.append(s)
        all_details.extend(d)

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    detail_fields = ["task_id", "pair", "background", "axis", "line_index",
                     "group", "n_cells", "colours", "cells"]
    with open(DETAILS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        w.writerows(all_details)

    print(f"Tasks analysed: {len(summaries)}\n")
    for key, title in (("rows_label", "ROWS"), ("columns_label", "COLUMNS")):
        counts = Counter(s[key] for s in summaries)
        print(f"{title}: label across all training pairs")
        for label, n in counts.most_common():
            print(f"  {label:<24} {n:>5}  ({100 * n / len(summaries):.1f}%)")
        print()
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {DETAILS_CSV}")


if __name__ == "__main__":
    main()
