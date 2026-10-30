"""
Step 5: Grid rows & columns, version 2 (investigation only, not used to solve yet)

Treats every ROW and every COLUMN as one unit and writes down how the output's
lines relate to the input's lines. Unlike the cells step it also works when the
size changes, as long as the lines have the same length (for rows: the same
width, for columns: the same height).

Background = most frequent colour of the INPUT grid (ties: lowest colour number).
No colour is hard-coded.

For each axis (rows, columns) and each training pair:

  COUNT   how many lines: same / more / fewer
  RATIO   same, x2, x3 ... (output is a whole multiple), /2, /3 ... (a whole fraction),
          or other_more / other_fewer
  STORY   how the output lines relate to the input lines
            identical        : the same lines in the same order
            reversed         : the same lines in the opposite order
            kept_in_order    : every output line is an input line, in order, some removed
            repeated         : the input lines in order, with extra copies
            reordered        : every output line is an input line, other order
            has_new_lines    : at least one output line is not an input line
            length_differs   : lines have different lengths, so they cannot be compared whole
  DETAIL  kept_in_order -> what was removed: blank (all background), duplicates
          (the line still exists in the output), or other
          same line count -> what changed: content_lines (the changed lines had
          something in them), empty_lines (they were all background), mixed_lines
  DRAWN   same line count -> uniform if every changed output line is a single colour
          (a line was drawn or filled), partial if not
  SPAN    same line count -> contiguous if the changed lines form one block of
          neighbouring lines, scattered if not

Task level: a description counts when it is the same in EVERY training pair, otherwise
"mixed". Test check: the public test pair is described the same way. holds = it matches
the training description, breaks = it does not.

Outputs
  * tables in the terminal
  * results/grid_rows_columns_results.csv   one row per task
  * results/grid_rows_columns_details.csv   one row per (pair, axis): the story and,
                                            for every output line, the input line it equals

Usage
  python solver/grid_rows_and_columns/grid_rows_and_columns.py
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (holds data/)
DATA_DIR = ROOT / "data"
GROUPS_CSV = ROOT / "solver" / "grid_shapes" / "results" / "grid_groups_shapes_results.csv"
RESULTS_PATH = HERE / "results" / "grid_rows_columns_results.csv"
DETAILS_PATH = HERE / "results" / "grid_rows_columns_details.csv"

AXES = ("rows", "columns")
FIELDS = ("count", "ratio", "story", "detail", "drawn", "span")


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load_tasks(folder):
    if not folder.is_dir():
        raise SystemExit(f"Data folder not found: {folder}")
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


def read_groups():
    if not GROUPS_CSV.exists():
        print(f"Note: {GROUPS_CSV.name} not found, the group column stays empty")
        return {}
    with GROUPS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        return {(r["dataset"], r["task_id"]): r["group"] for r in csv.DictReader(file)}


def background_colour(grid):
    values, counts = np.unique(np.array(grid), return_counts=True)
    return int(values[np.argmax(counts)])        # ties: lowest colour number


def split_lines(grid):
    rows = [tuple(r) for r in grid]
    columns = [tuple(c) for c in zip(*grid)]
    return {"rows": rows, "columns": columns}


# ----------------------------------------------------------------------
# Describing one axis of one pair
# ----------------------------------------------------------------------
def ratio_label(n_in, n_out):
    if n_in == n_out:
        return "same"
    if n_out % n_in == 0:
        return f"x{n_out // n_in}"
    if n_in % n_out == 0:
        return f"/{n_in // n_out}"
    return "other_more" if n_out > n_in else "other_fewer"


def greedy_subsequence(small, big):
    """Positions in big that match small in order, or None."""
    positions, start = [], 0
    for line in small:
        while start < len(big) and big[start] != line:
            start += 1
        if start == len(big):
            return None
        positions.append(start)
        start += 1
    return positions


def removed_kind(in_lines, out_lines, used, bg):
    removed = [in_lines[i] for i in range(len(in_lines)) if i not in set(used)]
    if all(set(line) == {bg} for line in removed):
        return "blank"
    if all(line in set(out_lines) for line in removed):
        return "duplicates"
    return "other"


def describe_axis(in_lines, out_lines, bg):
    n_in, n_out = len(in_lines), len(out_lines)
    info = {"count": "same" if n_in == n_out else "more" if n_out > n_in else "fewer",
            "ratio": ratio_label(n_in, n_out), "story": "length_differs",
            "detail": "", "drawn": "", "span": "", "mapping": "", "changed_pct": ""}
    if len(in_lines[0]) != len(out_lines[0]):
        return info

    first = {}
    for index, line in enumerate(in_lines):
        first.setdefault(line, index)
    mapping = [first.get(line) for line in out_lines]
    info["mapping"] = ",".join("new" if m is None else str(m) for m in mapping)

    if out_lines == in_lines:
        info["story"] = "identical"
    elif None in mapping:
        info["story"] = "has_new_lines"
    elif out_lines == in_lines[::-1]:
        info["story"] = "reversed"
    else:
        used = greedy_subsequence(out_lines, in_lines)
        if used is not None:
            info["story"] = "kept_in_order"
            info["detail"] = removed_kind(in_lines, out_lines, used, bg)
        elif greedy_subsequence(in_lines, out_lines) is not None:
            info["story"] = "repeated"
        else:
            info["story"] = "reordered"

    if n_in == n_out:
        changed = [i for i in range(n_in) if in_lines[i] != out_lines[i]]
        info["changed_pct"] = 100 * len(changed) / n_in
        if changed:
            info["drawn"] = ("uniform" if all(len(set(out_lines[i])) == 1 for i in changed)
                             else "partial")
            info["span"] = ("contiguous" if changed == list(range(changed[0], changed[-1] + 1))
                            else "scattered")
        if changed and not info["detail"]:
            has_content = [any(c != bg for c in in_lines[i]) for i in changed]
            info["detail"] = ("content_lines" if all(has_content) else
                              "empty_lines" if not any(has_content) else "mixed_lines")
    return info


def describe_pair(grid_in, grid_out):
    bg = background_colour(grid_in)
    lines_in, lines_out = split_lines(grid_in), split_lines(grid_out)
    return {axis: describe_axis(lines_in[axis], lines_out[axis], bg) for axis in AXES}


# ----------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------
def task_value(values):
    return values[0] if len(set(values)) == 1 else "mixed"


def check(flag):
    return "n/a" if flag is None else ("holds" if flag else "breaks")


def analyse_task(task):
    train = [describe_pair(p["input"], p["output"]) for p in task["train"]]
    tests = [describe_pair(p["input"], p["output"]) for p in task.get("test", [])
             if "output" in p]
    same_size = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0])
                    for p in task["train"])

    row = {"same_size_all": "yes" if same_size else "no", "details": []}
    for axis in AXES:
        task_labels = {f: task_value([pair[axis][f] for pair in train]) for f in FIELDS}
        for f in FIELDS:
            row[f"{axis}_{f}"] = task_labels[f]
        pcts = [pair[axis]["changed_pct"] for pair in train if pair[axis]["changed_pct"] != ""]
        row[f"{axis}_changed_pct"] = f"{sum(pcts) / len(pcts):.1f}" if pcts else ""

        if task_labels["story"] == "mixed" or not tests:
            row[f"{axis}_test"] = "n/a"
        else:
            row[f"{axis}_test"] = check(all(
                all(task_labels[f] in ("mixed", t[axis][f]) for f in FIELDS) for t in tests))

        for index, pair in enumerate(train, start=1):
            info = pair[axis]
            row["details"].append({"pair": index, "axis": axis, "lines_in": None,
                                   "count": info["count"], "story": info["story"],
                                   "detail": info["detail"], "mapping": info["mapping"]})
    return row


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def rate(holds, breaks):
    return f"{holds / (holds + breaks):.0%}" if (holds + breaks) else "-"


def print_table(title, rows, key, test_key):
    print(f"\n{title}")
    header = f"{'Group':<18}{'Tasks':>7}{'Holds':>8}{'Breaks':>8}{'n/a':>6}{'Holds (checked)':>17}"
    print(header)
    print("-" * len(header))
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    for name, group_rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        h = sum(1 for r in group_rows if r[test_key] == "holds")
        b = sum(1 for r in group_rows if r[test_key] == "breaks")
        print(f"{name:<18}{len(group_rows):>7}{h:>8}{b:>8}{len(group_rows) - h - b:>6}{rate(h, b):>17}")


def print_dataset(name, rows):
    print(f"\n=== {name}: rows & columns ({len(rows)} tasks) ===")
    for axis in AXES:
        print_table(f"{axis.upper()}: story (does the training pattern hold on the test pair?)",
                    rows, f"{axis}_story", f"{axis}_test")
        print_table(f"{axis.upper()}: how the number of lines changes", rows,
                    f"{axis}_count", f"{axis}_test")
        for key, text in (("drawn", "changed lines drawn as one colour?"),
                          ("span", "changed lines in one block?")):
            filled = [r for r in rows if r[f"{axis}_{key}"] not in ("",)]
            print_table(f"{axis.upper()}: {text} (same line count only)", filled,
                        f"{axis}_{key}", f"{axis}_test")

    print("\nROWS story x COLUMNS story (top 12 combinations)")
    combos = Counter((r["rows_story"], r["columns_story"]) for r in rows)
    for (row_story, col_story), n in combos.most_common(12):
        print(f"  rows {row_story:<16} columns {col_story:<16} {n:>5}")

    kept = Counter(r["rows_detail"] for r in rows if r["rows_story"] == "kept_in_order")
    kept_c = Counter(r["columns_detail"] for r in rows if r["columns_story"] == "kept_in_order")
    if kept or kept_c:
        print(f"\nLines removed (kept_in_order)  rows: {dict(kept)}   columns: {dict(kept_c)}")


def write_files(all_rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "task_id", "group", "same_size_all"]
    for axis in AXES:
        columns += [f"{axis}_{f}" for f in FIELDS] + [f"{axis}_changed_pct", f"{axis}_test"]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for r in all_rows:
            writer.writerow([r[c] for c in columns])

    detail_columns = ["pair", "axis", "count", "story", "detail", "mapping"]
    with DETAILS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "task_id"] + detail_columns)
        for r in all_rows:
            for d in r["details"]:
                writer.writerow([r["dataset"], r["task_id"]] + [d[c] for c in detail_columns])
    print(f"\nPer-task results written to {RESULTS_PATH}")
    print(f"Per-pair details written to {DETAILS_PATH}")


def main():
    groups = read_groups()
    all_rows = []
    for dataset in ("training", "evaluation"):
        rows = []
        for task_id, task in load_tasks(DATA_DIR / dataset).items():
            row = analyse_task(task)
            row["dataset"], row["task_id"] = dataset, task_id
            row["group"] = groups.get((dataset, task_id), "")
            rows.append(row)
        print_dataset(dataset, rows)
        all_rows += rows
    write_files(all_rows)


if __name__ == "__main__":
    main()