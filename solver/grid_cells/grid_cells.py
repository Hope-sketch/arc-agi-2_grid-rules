"""
Step 4: Grid cells, version 2 (investigation only, not used to solve yet)

Compares every training input with its output cell by cell and writes down
WHAT IS DIFFERENT, so the transformations step knows what it has to explain.
Only possible when input and output have the same height x width; tasks where a
training pair changes size are "not_same_size" (the sizes step handles those).

Background = most frequent colour of the INPUT grid (ties: lowest colour number).
No colour is hard-coded.

For every pair the changed cells are described by four things:

  KIND     how cells change
             identical / added_only (background -> colour) / removed_only (colour ->
             background) / recoloured_only (colour -> other colour) / mixed
  TO       where the new colour of the changed cells comes from
             to_background : every changed cell becomes background
             fixed:<c>     : every changed cell becomes the same colour c
             neighbour     : the new colour is the colour of a side neighbour in the input
             line          : the new colour is a colour found in the same row or column
             other
  WHERE    what the changed cells have in common (true for 95%+ of them)
             enclosed        : colour on the left, right, above and below (in the input)
             touches_content : a side neighbour that is not background
             shares_line     : another non-background cell in the same row or column
             inside_content  : inside the bounding box of the input's non-background cells
             on_border       : on the outer edge of the grid
  BLOBS    how many separate groups the changed cells form (side by side)

Task level: a description counts when it is the same in EVERY training pair.
Test check: the public test pair is described the same way. holds = it matches the
training description, breaks = it does not.

Outputs
  * tables in the terminal
  * results/grid_cells_results.csv   one row per task
  * results/grid_cells_details.csv   one row per (pair, kind of change): the colours,
                                     the number of cells, their bounding box and coordinates

Usage
  python solver/grid_cells/grid_cells.py
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
RESULTS_PATH = HERE / "results" / "grid_cells_results.csv"
DETAILS_PATH = HERE / "results" / "grid_cells_details.csv"

WHERE_ORDER = ["enclosed", "touches_content", "on_border", "shares_line", "inside_content"]
SHARE_NEEDED = 0.95          # share of the changed cells a "where" test must cover
MAX_COORDS = 40              # coordinates written per row in the details file


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


def to_np(grid):
    return np.array(grid, dtype=int)


def background_colour(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])        # ties: lowest colour number


# ----------------------------------------------------------------------
# Describing one same-size pair
# ----------------------------------------------------------------------
def kind_of(grid_in, grid_out, bg, changed):
    if not changed.any():
        return "identical"
    was, now = grid_in[changed], grid_out[changed]
    kinds = set()
    kinds.update(["added"] if (was == bg).any() else [])
    kinds.update(["removed"] if (now == bg).any() else [])
    kinds.update(["recoloured"] if ((was != bg) & (now != bg)).any() else [])
    return f"{next(iter(kinds))}_only" if len(kinds) == 1 else "mixed"


def to_label(grid_in, grid_out, bg, changed):
    """Where the new colour of the changed cells comes from."""
    rows, cols = np.where(changed)
    new = grid_out[changed]
    if (new == bg).all():
        return "to_background"
    if len(set(new.tolist())) == 1:
        return f"fixed:{int(new[0])}"

    padded = np.pad(grid_in, 1, constant_values=-1)
    sides = [padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]]
    from_side = np.zeros(grid_in.shape, dtype=bool)
    for s in sides:
        from_side |= (s == grid_out) & (s != bg)
    if from_side[changed].mean() >= SHARE_NEEDED:
        return "neighbour"

    on_line = 0
    for r, c, colour in zip(rows, cols, new.tolist()):
        line = set(grid_in[r, :].tolist()) | set(grid_in[:, c].tolist())
        on_line += colour in line and colour != bg
    return "line" if on_line / len(rows) >= SHARE_NEEDED else "other"


def where_tests(grid_in, bg):
    """For every cell: does it pass each 'where' test? (dict of boolean grids)"""
    content = grid_in != bg
    padded = np.pad(content, 1)
    touches = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]

    def before(a, axis):         # any content strictly before each cell along the axis
        acc = np.logical_or.accumulate(a, axis=axis)
        out = np.zeros_like(a)
        if axis == 1:
            out[:, 1:] = acc[:, :-1]
        else:
            out[1:, :] = acc[:-1, :]
        return out

    left, up = before(content, 1), before(content, 0)
    right = before(content[:, ::-1], 1)[:, ::-1]
    down = before(content[::-1, :], 0)[::-1, :]
    enclosed = left & right & up & down

    row_count = content.sum(axis=1)[:, None] - content
    col_count = content.sum(axis=0)[None, :] - content
    shares_line = (row_count > 0) | (col_count > 0)

    inside = np.zeros_like(content)
    if content.any():
        ys, xs = np.where(content)
        inside[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True

    border = np.zeros_like(content)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    return {"enclosed": enclosed, "touches_content": touches, "on_border": border,
            "shares_line": shares_line, "inside_content": inside}


def where_label(grid_in, bg, changed):
    """The set of 'where' tests that cover 95%+ of the changed cells."""
    if not changed.any():
        return set()
    tests = where_tests(grid_in, bg)
    return {name for name, mask in tests.items()
            if mask[changed].mean() >= SHARE_NEEDED}


def blob_count(changed):
    """Separate groups of changed cells (side by side)."""
    seen = np.zeros_like(changed)
    groups = 0
    height, width = changed.shape
    for r, c in zip(*np.where(changed)):
        if seen[r, c]:
            continue
        groups += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and changed[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return groups


def describe_pair(grid_in, grid_out):
    bg = background_colour(grid_in)
    changed = grid_in != grid_out
    return {"kind": kind_of(grid_in, grid_out, bg, changed),
            "to": to_label(grid_in, grid_out, bg, changed) if changed.any() else "none",
            "where": where_label(grid_in, bg, changed),
            "blobs": blob_count(changed),
            "changed_pct": 100 * changed.mean(),
            "bg": bg, "changed": changed}


# ----------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------
def to_group(labels):
    """Task-level 'to' description from the per-pair labels."""
    unique = set(labels)
    if len(unique) == 1:
        label = next(iter(unique))
        return "fixed" if label.startswith("fixed:") else label
    if all(label.startswith("fixed:") for label in labels):
        return "fixed_per_pair"
    return "mixed"


def to_matches(group, task_labels, test_label):
    if group == "fixed":
        return test_label == task_labels[0]
    if group == "fixed_per_pair":
        return test_label.startswith("fixed:")
    return test_label == group


def check(flag):
    return "n/a" if flag is None else ("holds" if flag else "breaks")


def analyse_task(task):
    train = [(to_np(p["input"]), to_np(p["output"])) for p in task.get("train", [])]
    tests = [(to_np(p["input"]), to_np(p["output"])) for p in task.get("test", [])
             if "output" in p]
    blank = {"applicable": "no", "kind": "not_same_size", "kind_test": "n/a",
             "to": "n/a", "to_test": "n/a", "where_holds": "", "where_tightest": "n/a",
             "where_test": "n/a", "blobs": "n/a", "blobs_test": "n/a",
             "avg_changed_pct": "", "details": []}
    if not train or not all(i.shape == o.shape for i, o in train):
        return blank

    pairs = [describe_pair(i, o) for i, o in train]
    test_pairs = [describe_pair(i, o) if i.shape == o.shape else None for i, o in tests]
    row = dict(blank, applicable="yes")

    # KIND
    kinds = [p["kind"] for p in pairs]
    row["kind"] = kinds[0] if len(set(kinds)) == 1 else "other"
    row["kind_test"] = ("n/a" if row["kind"] == "other" or not test_pairs else
                        check(all(t is not None and t["kind"] == row["kind"] for t in test_pairs)))

    # Nothing changed in any pair: nothing more to describe
    changed_pairs = [p for p in pairs if p["kind"] != "identical"]
    if changed_pairs:
        labels = [p["to"] for p in changed_pairs]
        row["to"] = to_group(labels)
        row["to_test"] = ("n/a" if row["to"] == "mixed" or not test_pairs else check(
            all(t is not None and t["to"] != "none" and to_matches(row["to"], labels, t["to"])
                for t in test_pairs)))

        holds = set.intersection(*[p["where"] for p in changed_pairs])
        row["where_holds"] = "|".join(w for w in WHERE_ORDER if w in holds)
        row["where_tightest"] = next((w for w in WHERE_ORDER if w in holds), "none")
        row["where_test"] = ("n/a" if not holds or not test_pairs else check(
            all(t is not None and holds <= t["where"] for t in test_pairs)))

        counts = {p["blobs"] for p in changed_pairs}
        row["blobs"] = f"constant:{next(iter(counts))}" if len(counts) == 1 else "varies"
        row["blobs_test"] = ("n/a" if len(counts) != 1 or not test_pairs else check(
            all(t is not None and t["blobs"] == next(iter(counts)) for t in test_pairs)))

    row["avg_changed_pct"] = f"{sum(p['changed_pct'] for p in pairs) / len(pairs):.1f}"
    row["details"] = pair_details(train, pairs)
    return row


def pair_details(train, pairs):
    """One entry per (pair, kind of change, from colour, to colour)."""
    details = []
    for index, ((grid_in, grid_out), p) in enumerate(zip(train, pairs), start=1):
        bg, changed = p["bg"], p["changed"]
        groups = defaultdict(list)
        for r, c in zip(*np.where(changed)):
            was, now = int(grid_in[r, c]), int(grid_out[r, c])
            kind = "added" if was == bg else "removed" if now == bg else "changed"
            groups[(kind, was, now)].append((int(r), int(c)))
        for (kind, was, now), cells in sorted(groups.items()):
            rs, cs = [r for r, _ in cells], [c for _, c in cells]
            coords = ";".join(f"{r},{c}" for r, c in cells[:MAX_COORDS])
            details.append({"pair": index, "group": kind, "from_colour": was,
                            "to_colour": now, "cells": len(cells),
                            "bbox": f"rows {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)}",
                            "coordinates": coords + (";..." if len(cells) > MAX_COORDS else "")})
    return details


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def rate(holds, breaks):
    return f"{holds / (holds + breaks):.0%}" if (holds + breaks) else "-"


def print_table(title, rows, key, test_key, order=None):
    print(f"\n{title}")
    header = f"{'Group':<18}{'Tasks':>7}{'Holds':>8}{'Breaks':>8}{'n/a':>6}{'Holds (checked)':>17}"
    print(header)
    print("-" * len(header))
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    names = order or [k for k, _ in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    for name in names:
        group_rows = by.get(name, [])
        if not group_rows:
            continue
        h = sum(1 for r in group_rows if r[test_key] == "holds")
        b = sum(1 for r in group_rows if r[test_key] == "breaks")
        print(f"{name:<18}{len(group_rows):>7}{h:>8}{b:>8}{len(group_rows) - h - b:>6}{rate(h, b):>17}")


def print_dataset(name, rows):
    applicable = [r for r in rows if r["applicable"] == "yes"]
    changed = [r for r in applicable if r["to"] != "n/a"]
    print(f"\n=== {name}: cells ({len(rows)} tasks, {len(applicable)} with same-size pairs) ===")
    print_table("KIND: how cells change", rows, "kind", "kind_test",
                ["identical", "added_only", "removed_only", "recoloured_only", "mixed",
                 "other", "not_same_size"])
    print_table("TO: where the new colour comes from", changed, "to", "to_test")
    print_table("WHERE: the tightest thing the changed cells have in common",
                changed, "where_tightest", "where_test")
    print_table("BLOBS: separate groups of changed cells", changed,
                "blobs", "blobs_test")
    pcts = sorted(float(r["avg_changed_pct"]) for r in applicable)
    if pcts:
        print(f"\nCells changed per task (same-size tasks): median {pcts[len(pcts) // 2]:.1f}%,"
              f" mean {sum(pcts) / len(pcts):.1f}%")


def write_files(all_rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "task_id", "group", "applicable", "kind", "kind_test",
               "to", "to_test", "where_holds", "where_tightest", "where_test",
               "blobs", "blobs_test", "avg_changed_pct"]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for r in all_rows:
            writer.writerow([r[c] for c in columns])

    detail_columns = ["pair", "group", "from_colour", "to_colour", "cells", "bbox", "coordinates"]
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