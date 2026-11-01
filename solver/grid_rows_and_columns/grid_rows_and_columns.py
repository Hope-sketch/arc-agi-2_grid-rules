"""
Step 5: Grid rows and columns (investigation only)

The grid is read as a Cartesian plane:
  X axis = columns, positive X is RIGHT.
  Y axis = rows,    positive Y is DOWN (row numbers grow downward).
A move of (dy, dx) = (+2, -1) means two rows down and one column left.

Compares every training input with its output (and the training outputs with each
other) line by line. Blank lines (only background) are ignored. A line pair is
"similar" when 60% or more of its non-background cells match.

  Rows are comparable when input and output have the same width,
  columns are comparable when they have the same height.

Events (true or false per training pair)
  Colour moves (each colour is slid over the grid to find where it went)
    COLOUR_MOVED_X / COLOUR_MOVED_Y                 a colour moved along that axis
    COLOUR_MOVED_POSITIVE_X / _NEGATIVE_X           right / left
    COLOUR_MOVED_POSITIVE_Y / _NEGATIVE_Y           down / up
    COLOURS_SHIFT_TOGETHER                          every moved colour moved by the same (dy, dx)
  Rows and columns
    ROW_SIMILAR / ROW_NOT_SIMILAR                   some row is / is not similar
    COLUMN_SIMILAR / COLUMN_NOT_SIMILAR             some column is / is not similar
    ALL_ROWS_SIMILAR / ALL_COLUMNS_SIMILAR          every non-blank line is similar, none added or removed
    ROW_ADDED / ROW_REMOVED                         a blank row gets content / a row becomes blank
    COLUMN_ADDED / COLUMN_REMOVED                   same for columns
    ROW_MOVED_POSITIVE_Y / _NEGATIVE_Y              an identical row reappears further down / up
    COLUMN_MOVED_POSITIVE_X / _NEGATIVE_X           an identical column reappears further right / left
    ROWS_REORDERED / COLUMNS_REORDERED              same rows (columns), different order
Output vs output (training outputs of one size, compared with each other)
    OUTPUT_ROW_SIMILAR / OUTPUT_ROW_NOT_SIMILAR / OUTPUT_COLUMN_SIMILAR / OUTPUT_COLUMN_NOT_SIMILAR

Receipt: "flags" = events true in EVERY training pair (or every output pair),
"sometimes" = true in some. "consistent_shift" = [dy, dx] when every training pair
moves its colours by the same shift, else null. The receipt never contains the test
output; the public test output is only used for the "holds on test" checks.

Moves are only measured on same-size grids. A colour with a single cell is ambiguous
(moved, or removed and added elsewhere) and is reported as moved.

Background comes from find_background() in solver/grid_colours/grid_colours.py.

Usage
  python solver/grid_rows_and_columns/grid_rows_and_columns.py

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_rows_and_columns/results/grid_rows_and_columns_results.csv
  * solver/grid_rows_and_columns/results/grid_rows_and_columns_receipts.json

From another step
  import sys; sys.path.insert(0, str(ROOT / "solver" / "grid_rows_and_columns"))
  from grid_rows_and_columns import load_rows_columns_receipts, check_rows_columns_receipts
"""

import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_rows_and_columns_results.csv"
OUT_JSON = RESULTS_DIR / "grid_rows_and_columns_receipts.json"
DATASETS = ("training", "evaluation")

sys.path.insert(0, str(ROOT / "solver" / "grid_colours"))
try:
    from grid_colours import find_background
except ImportError:
    raise SystemExit("grid_rows_and_columns.py needs find_background() from "
                     "solver/grid_colours/grid_colours.py. Put the new grid_colours.py there first.")

LINE_SIMILAR_MIN = 60.0       # percent of non-background cells that match
MOVE_MIN_SHARE = 0.8          # a colour counts as moved if this share of it lands at the new spot

EVENTS = [
    "COLOUR_MOVED_X", "COLOUR_MOVED_Y",
    "COLOUR_MOVED_POSITIVE_X", "COLOUR_MOVED_NEGATIVE_X",
    "COLOUR_MOVED_POSITIVE_Y", "COLOUR_MOVED_NEGATIVE_Y", "COLOURS_SHIFT_TOGETHER",
    "ROW_SIMILAR", "ROW_NOT_SIMILAR", "COLUMN_SIMILAR", "COLUMN_NOT_SIMILAR",
    "ALL_ROWS_SIMILAR", "ALL_COLUMNS_SIMILAR",
    "ROW_ADDED", "ROW_REMOVED", "COLUMN_ADDED", "COLUMN_REMOVED",
    "ROW_MOVED_POSITIVE_Y", "ROW_MOVED_NEGATIVE_Y",
    "COLUMN_MOVED_POSITIVE_X", "COLUMN_MOVED_NEGATIVE_X",
    "ROWS_REORDERED", "COLUMNS_REORDERED",
]
OUTPUT_EVENTS = ["OUTPUT_ROW_SIMILAR", "OUTPUT_ROW_NOT_SIMILAR",
                 "OUTPUT_COLUMN_SIMILAR", "OUTPUT_COLUMN_NOT_SIMILAR"]


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            tasks[file_path.stem] = json.load(file)
    return tasks


def to_np(grid):
    return np.array(grid, dtype=int)


# ----------------------------------------------------------------------
# Lines (rows = axis 0, columns = axis 1)
# ----------------------------------------------------------------------
def is_blank(line, background):
    return background is not None and bool((line == background).all())


def line_share(line_a, line_b, bg_a, bg_b):
    """Percent of non-background cells (in either line) that match."""
    mask_a = np.ones(line_a.shape, bool) if bg_a is None else line_a != bg_a
    mask_b = np.ones(line_b.shape, bool) if bg_b is None else line_b != bg_b
    union = mask_a | mask_b
    total = int(union.sum())
    return 100.0 if total == 0 else 100 * int(((line_a == line_b) & union).sum()) / total


def compare_lines(a, b, bg_a, bg_b, axis):
    """Line-by-line comparison, or None if the lines have different lengths."""
    if axis == 1:
        a, b = a.T, b.T
    if a.shape[1] != b.shape[1]:
        return None
    n_a, n_b = len(a), len(b)
    shared = min(n_a, n_b)

    similar, not_similar, added, removed, changed = [], [], [], [], []
    for i in range(shared):
        if not np.array_equal(a[i], b[i]):
            changed.append(i)
        blank_a, blank_b = is_blank(a[i], bg_a), is_blank(b[i], bg_b)
        if blank_a and blank_b:
            continue
        if blank_a:
            added.append(i)
        elif blank_b:
            removed.append(i)
        elif line_share(a[i], b[i], bg_a, bg_b) >= LINE_SIMILAR_MIN:
            similar.append(i)
        else:
            not_similar.append(i)
    added += [j for j in range(shared, n_b) if not is_blank(b[j], bg_b)]
    removed += [i for i in range(shared, n_a) if not is_blank(a[i], bg_a)]

    # identical lines that reappear at another index
    moves = []
    free_out = {}
    for j in range(n_b):
        if is_blank(b[j], bg_b) or (j < n_a and np.array_equal(a[j], b[j])):
            continue
        free_out.setdefault(tuple(b[j].tolist()), []).append(j)
    for i in range(n_a):
        if is_blank(a[i], bg_a) or (i < n_b and np.array_equal(a[i], b[i])):
            continue
        options = free_out.get(tuple(a[i].tolist()))
        if options:
            j = min(options, key=lambda j: abs(j - i))
            options.remove(j)
            moves.append([i, j])

    reordered = (n_a == n_b and not np.array_equal(a, b)
                 and sorted(map(tuple, a.tolist())) == sorted(map(tuple, b.tolist())))
    return {"similar": similar, "not_similar": not_similar, "added": added, "removed": removed,
            "changed": changed, "moves": moves, "reordered": reordered}


# ----------------------------------------------------------------------
# Colour moves
# ----------------------------------------------------------------------
def best_shift(mask_a, mask_b):
    """(dy, dx, overlap, overlap_without_moving) that lines mask_a up best with mask_b."""
    h, w = mask_a.shape
    size = (2 * h, 2 * w)
    corr = np.rint(np.fft.irfft2(np.conj(np.fft.rfft2(mask_a.astype(float), s=size))
                                 * np.fft.rfft2(mask_b.astype(float), s=size), s=size)).astype(int)
    best = corr.max()
    candidates = np.argwhere(corr == best)

    def signed(idx, n):
        return int(idx) if idx < n else int(idx) - 2 * n

    shifts = [(signed(y, h), signed(x, w)) for y, x in candidates]
    dy, dx = min(shifts, key=lambda s: (abs(s[0]) + abs(s[1]), s))
    return dy, dx, int(best), int(corr[0, 0])


def colour_moves(a, b, bg_a, bg_b):
    """[{colour, dy, dx, overlap}] for colours that moved. Same-size grids only."""
    if a.shape != b.shape:
        return []
    shared = (set(np.unique(a).tolist()) & set(np.unique(b).tolist())) - {bg_a, bg_b}
    moves = []
    for colour in sorted(shared):
        mask_a, mask_b = a == colour, b == colour
        count = min(int(mask_a.sum()), int(mask_b.sum()))
        dy, dx, overlap, overlap_still = best_shift(mask_a, mask_b)
        if (dy, dx) != (0, 0) and overlap > overlap_still and overlap >= MOVE_MIN_SHARE * count:
            moves.append({"colour": int(colour), "dy": dy, "dx": dx,
                          "overlap": round(100 * overlap / count, 1)})
    return moves


# ----------------------------------------------------------------------
# One pair
# ----------------------------------------------------------------------
def line_summary(result):
    if result is None:
        return {"comparable": False}
    return {"comparable": True, "similar": len(result["similar"]),
            "not_similar": len(result["not_similar"]), "added": len(result["added"]),
            "removed": len(result["removed"]), "changed_indexes": result["changed"],
            "moves": result["moves"], "reordered": result["reordered"]}


def compare_pair(grid_in, grid_out):
    a, b = to_np(grid_in), to_np(grid_out)
    bg_a, bg_b = find_background(grid_in), find_background(grid_out)
    rows = compare_lines(a, b, bg_a, bg_b, 0)
    columns = compare_lines(a, b, bg_a, bg_b, 1)
    moves = colour_moves(a, b, bg_a, bg_b)

    ev = set()
    for m in moves:
        if m["dx"]:
            ev.update(("COLOUR_MOVED_X", "COLOUR_MOVED_POSITIVE_X" if m["dx"] > 0 else "COLOUR_MOVED_NEGATIVE_X"))
        if m["dy"]:
            ev.update(("COLOUR_MOVED_Y", "COLOUR_MOVED_POSITIVE_Y" if m["dy"] > 0 else "COLOUR_MOVED_NEGATIVE_Y"))
    shifts = {(m["dy"], m["dx"]) for m in moves}
    if len(shifts) == 1 and len(moves) >= 2:
        ev.add("COLOURS_SHIFT_TOGETHER")
    common_shift = list(next(iter(shifts))) if len(shifts) == 1 else None

    for name, result, up, down in (("ROW", rows, "ROW_MOVED_NEGATIVE_Y", "ROW_MOVED_POSITIVE_Y"),
                                   ("COLUMN", columns, "COLUMN_MOVED_NEGATIVE_X", "COLUMN_MOVED_POSITIVE_X")):
        if result is None:
            continue
        if result["similar"]: ev.add(f"{name}_SIMILAR")
        if result["not_similar"]: ev.add(f"{name}_NOT_SIMILAR")
        if result["similar"] and not (result["not_similar"] or result["added"] or result["removed"]):
            ev.add(f"ALL_{name}S_SIMILAR")
        if result["added"]: ev.add(f"{name}_ADDED")
        if result["removed"]: ev.add(f"{name}_REMOVED")
        if result["reordered"]: ev.add(f"{name}S_REORDERED")
        for i, j in result["moves"]:
            ev.add(down if j > i else up)

    return {"rows": line_summary(rows), "columns": line_summary(columns),
            "colour_moves": moves, "common_shift": common_shift,
            "events": [e for e in EVENTS if e in ev]}


def output_pair_events(grid_a, grid_b):
    a, b = to_np(grid_a), to_np(grid_b)
    bg_a, bg_b = find_background(grid_a), find_background(grid_b)
    ev = set()
    for name, axis in (("ROW", 0), ("COLUMN", 1)):
        result = compare_lines(a, b, bg_a, bg_b, axis)
        if result is not None:
            if result["similar"]: ev.add(f"OUTPUT_{name}_SIMILAR")
            if result["not_similar"]: ev.add(f"OUTPUT_{name}_NOT_SIMILAR")
    return ev


# ----------------------------------------------------------------------
# Receipt (training pairs only)
# ----------------------------------------------------------------------
def get_rows_columns_receipt(task):
    pairs = [compare_pair(p["input"], p["output"]) for p in task["train"]]
    flags = [e for e in EVENTS if all(e in p["events"] for p in pairs)]
    sometimes = [e for e in EVENTS if any(e in p["events"] for p in pairs) and e not in flags]

    shifts = [tuple(p["common_shift"]) if p["common_shift"] else None for p in pairs]
    consistent_shift = list(shifts[0]) if shifts[0] is not None and len(set(shifts)) == 1 else None

    outputs = [to_np(p["output"]) for p in task["train"]]
    output_flags, comparable = [], False
    if len(outputs) >= 2 and len({o.shape for o in outputs}) == 1:
        comparable = True
        per_pair = [output_pair_events(x, y) for x, y in
                    combinations([p["output"] for p in task["train"]], 2)]
        output_flags = [e for e in OUTPUT_EVENTS if all(e in ev for ev in per_pair)]
        flags += output_flags
        sometimes += [e for e in OUTPUT_EVENTS if any(e in ev for ev in per_pair) and e not in output_flags]

    return {"flags": flags, "sometimes": sometimes, "consistent_shift": consistent_shift,
            "outputs_comparable": comparable, "pairs": pairs,
            "test_input_sizes": [[len(t["input"]), len(t["input"][0])] for t in task["test"]]}


def check_on_public_test(receipt, task):
    """Which flags (and the shift) also hold on the public test pair? Verification only."""
    if not all("output" in t for t in task["test"]):
        return None
    test_pairs = [compare_pair(t["input"], t["output"]) for t in task["test"]]
    events = [set(p["events"]) for p in test_pairs]

    train_outputs = [p["output"] for p in task["train"]]
    # output flags: the test output is compared with every training output
    out_checks = []
    for t in task["test"]:
        per_output = [output_pair_events(t["output"], o) for o in train_outputs
                      if (len(o), len(o[0])) == (len(t["output"]), len(t["output"][0]))]
        out_checks.append(per_output)

    holding, failing = [], []
    for f in receipt["flags"]:
        if f in EVENTS:
            ok = all(f in e for e in events)
        else:
            ok = bool(out_checks) and all(per and all(f in e for e in per) for per in out_checks)
        (holding if ok else failing).append(f)

    shift = "n/a"
    if receipt["consistent_shift"] is not None:
        shift = "yes" if all(p["common_shift"] == receipt["consistent_shift"] for p in test_pairs) else "no"
    return {"holding": holding, "failing": failing, "shift": shift}


# ----------------------------------------------------------------------
# For the other steps
# ----------------------------------------------------------------------
def load_rows_columns_receipts(path=OUT_JSON):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid rows and columns receipts not found: {path}\n"
                         f"Run solver/grid_rows_and_columns/grid_rows_and_columns.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_rows_columns_receipts(receipts, dataset, task_ids):
    """Stop if any task has no grid rows and columns receipt, and say which ones."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid rows and columns receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_rows_and_columns/grid_rows_and_columns.py.")
    print(f"Grid rows and columns receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def csv_row(dataset, task_id, task, receipt, check):
    pairs = receipt["pairs"]

    def lines(axis, key):
        return ";".join(str(p[axis][key]) if p[axis]["comparable"] else "n/a" for p in pairs)

    return {
        "dataset": dataset, "task_id": task_id, "n_train_pairs": len(task["train"]),
        "flags": "|".join(receipt["flags"]), "sometimes": "|".join(receipt["sometimes"]),
        "consistent_shift": "" if receipt["consistent_shift"] is None else
        f"dy={receipt['consistent_shift'][0]},dx={receipt['consistent_shift'][1]}",
        "colour_moves": ";".join("|".join(f"{m['colour']}:dy={m['dy']},dx={m['dx']}" for m in p["colour_moves"])
                                 for p in pairs),
        "rows_similar": lines("rows", "similar"), "rows_not_similar": lines("rows", "not_similar"),
        "rows_added": lines("rows", "added"), "rows_removed": lines("rows", "removed"),
        "columns_similar": lines("columns", "similar"), "columns_not_similar": lines("columns", "not_similar"),
        "columns_added": lines("columns", "added"), "columns_removed": lines("columns", "removed"),
        "outputs_comparable": "yes" if receipt["outputs_comparable"] else "no",
        "test_input_sizes": ";".join(f"{h}x{w}" for h, w in receipt["test_input_sizes"]),
        "holds_shift": "unknown" if check is None else check["shift"],
        "flags_holding_on_test": "unknown" if check is None else "|".join(check["holding"]),
        "flags_failing_on_test": "unknown" if check is None else "|".join(check["failing"]),
    }


def print_summary(dataset, rows):
    total = len(rows)
    print(f"\n=== {dataset}: grid rows and columns ({total} tasks) ===")
    header = f"{'Event':<28}{'In every pair':>14}{'Holds on test':>16}{'Sometimes':>11}"
    print(header)
    print("-" * len(header))
    for event in EVENTS + OUTPUT_EVENTS:
        flagged = [r for r in rows if event in r["flags"].split("|")]
        sometimes = sum(event in r["sometimes"].split("|") for r in rows)
        checked = [r for r in flagged if r["flags_holding_on_test"] != "unknown"]
        holds = sum(event in r["flags_holding_on_test"].split("|") for r in checked)
        print(f"{event:<28}{len(flagged):>14}{(f'{holds}/{len(checked)}' if checked else '-'):>16}{sometimes:>11}")

    shifted = [r for r in rows if r["consistent_shift"]]
    checked = [r for r in shifted if r["holds_shift"] in ("yes", "no")]
    holds = sum(r["holds_shift"] == "yes" for r in checked)
    print(f"\nSame colour shift in every training pair: {len(shifted)} tasks"
          f" (holds on test: {holds}/{len(checked)})" if checked else
          f"\nSame colour shift in every training pair: {len(shifted)} tasks")
    common = Counter(r["consistent_shift"] for r in shifted).most_common(6)
    if common:
        print("  most common: " + ", ".join(f"{s} x{n}" for s, n in common))
    print(f"Outputs comparable with each other: {sum(r['outputs_comparable'] == 'yes' for r in rows)}/{total}")


def main():
    all_rows, receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        rows, receipts[dataset] = [], {}
        for task_id, task in load_tasks(folder).items():
            receipt = get_rows_columns_receipt(task)
            receipts[dataset][task_id] = receipt
            rows.append(csv_row(dataset, task_id, task, receipt, check_on_public_test(receipt, task)))
        print_summary(dataset, rows)
        all_rows += rows

    if not all_rows:
        raise SystemExit(f"No tasks found under {DATA_DIR}. Expected data/training and data/evaluation.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with OUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(receipts, file)
    print(f"\nPer-task results written to {OUT_CSV}")
    print(f"Receipts for later steps written to {OUT_JSON}")


if __name__ == "__main__":
    main()