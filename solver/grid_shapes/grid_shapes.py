"""
Step 3: Grid shapes (investigation only)

A shape is a group of cells of ONE colour joined through their sides (up, down,
left, right = 4-connectivity). A single cell is a shape. The background is never a
shape (it comes from find_background() in solver/grid_colours/grid_colours.py, so
there is one place to improve it; with no background every cell is in a shape).

For every training pair the shapes of the input are matched to the shapes of the
output, strongest evidence first:
  exact     same cells, same place
  form      same colour, same form, moved
  recolour  same form (colour changed), moved or in place
  rotated   same colour, same form after a rotation / flip
  overlap   same-size grids only: the output shape sits on top of the input shape
            (it grew, shrank, or changed form)
Input shapes left unmatched are REMOVED, output shapes left unmatched are ADDED.

Events (true or false per training pair)
   1 SHAPES_ADDED               an output shape has no input shape
   2 SHAPES_REMOVED             an input shape has no output shape
   3 SHAPES_MAINTAINED          at least one shape exists in input and output
   4 SHAPE_SIZE_ADDED           a matched shape has more cells in the output
   5 SHAPE_SIZE_REDUCED         a matched shape has fewer cells
   6 SHAPE_SIZE_MAINTAINED      a matched shape has the same number of cells
   7 SHAPE_STRUCTURE_MAINTAINED a matched shape keeps exactly the same form
   8 SHAPE_STRUCTURE_REMOVED    a matched shape stays but its form changes
   9 SHAPE_HAS_0_HOLES          input: no shape has a hole
  10 SHAPE_HAS_1_HOLE           input: the most holes in one shape is 1
  11 SHAPE_HAS_2_PLUS_HOLES     input: some shape has 2 or more holes
  12 SHAPE_HOLES_MAINTAINED     a matched shape keeps its holes
  13 SHAPE_HOLES_REMOVED        a matched shape loses holes (opened up), not filled
  14 SHAPE_HOLES_FILLED_IN      empty hole cells of the input become non-background
  15 SHAPE_MOVED_POSITIVE_X     a shape moved right  (columns increase)
  16 SHAPE_MOVED_NEGATIVE_X     a shape moved left
  17 SHAPE_MOVED_POSITIVE_Y     a shape moved down   (rows increase)
  18 SHAPE_MOVED_NEGATIVE_Y     a shape moved up
  Extras
     SHAPE_COUNT_MORE / FEWER / SAME   more, fewer or equally many shapes in the output
     SHAPE_RECOLOURED                  a matched shape has the same form, another colour
     SHAPE_ROTATED_OR_FLIPPED          a shape comes back rotated or flipped
     NO_SHAPES                         neither grid has a shape

A hole is a region of cells enclosed by one shape (the outside is reached through
corners too, so a diagonal gap is not a hole). Movement is only measured on
same-size grids, and when several identical shapes exist the nearest ones are
paired, so movement is a best guess.

Receipt: "flags" = events true in EVERY training pair, "sometimes" = events true in
some pairs but not all. The receipt never contains the test output; the public test
output is only used for the "holds on test" check in the CSV and tables.

Usage
  python solver/grid_shapes/grid_shapes.py

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_shapes/results/grid_shapes_results.csv    (one row per task)
  * solver/grid_shapes/results/grid_shapes_receipts.json  (what later steps load)

From another step
  import sys; sys.path.insert(0, str(ROOT / "solver" / "grid_shapes"))
  from grid_shapes import load_shapes_receipts, check_shapes_receipts
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_shapes_results.csv"
OUT_JSON = RESULTS_DIR / "grid_shapes_receipts.json"
DATASETS = ("training", "evaluation")

sys.path.insert(0, str(ROOT / "solver" / "grid_colours"))
try:
    from grid_colours import find_background
except ImportError:
    raise SystemExit("grid_shapes.py needs find_background() from "
                     "solver/grid_colours/grid_colours.py. Put the new grid_colours.py there first.")

EVENTS = [
    "SHAPES_ADDED", "SHAPES_REMOVED", "SHAPES_MAINTAINED",
    "SHAPE_SIZE_ADDED", "SHAPE_SIZE_REDUCED", "SHAPE_SIZE_MAINTAINED",
    "SHAPE_STRUCTURE_MAINTAINED", "SHAPE_STRUCTURE_REMOVED",
    "SHAPE_HAS_0_HOLES", "SHAPE_HAS_1_HOLE", "SHAPE_HAS_2_PLUS_HOLES",
    "SHAPE_HOLES_MAINTAINED", "SHAPE_HOLES_REMOVED", "SHAPE_HOLES_FILLED_IN",
    "SHAPE_MOVED_POSITIVE_X", "SHAPE_MOVED_NEGATIVE_X",
    "SHAPE_MOVED_POSITIVE_Y", "SHAPE_MOVED_NEGATIVE_Y",
    "SHAPE_COUNT_MORE", "SHAPE_COUNT_FEWER", "SHAPE_COUNT_SAME",
    "SHAPE_RECOLOURED", "SHAPE_ROTATED_OR_FLIPPED", "NO_SHAPES",
]

TRANSFORMS = [lambda y, x: (y, x), lambda y, x: (y, -x), lambda y, x: (-y, x),
              lambda y, x: (-y, -x), lambda y, x: (x, y), lambda y, x: (x, -y),
              lambda y, x: (-x, y), lambda y, x: (-x, -y)]
NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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
# Shapes in one grid
# ----------------------------------------------------------------------
class Shape:
    __slots__ = ("colour", "cells", "cellset", "size", "top", "left", "bottom", "right",
                 "form", "holes", "hole_cells")


@lru_cache(maxsize=None)
def canonical(form):
    """The form with rotations and flips ignored."""
    best = None
    for transform in TRANSFORMS:
        points = [transform(y, x) for y, x in form]
        min_y = min(p[0] for p in points)
        min_x = min(p[1] for p in points)
        normal = tuple(sorted((y - min_y, x - min_x) for y, x in points))
        if best is None or normal < best:
            best = normal
    return best


def find_holes(cellset, top, left, bottom, right):
    """(number of holes, their cells). Outside is reached through corners too."""
    height, width = bottom - top + 3, right - left + 3
    blocked = [[False] * width for _ in range(height)]
    for y, x in cellset:
        blocked[y - top + 1][x - left + 1] = True

    def flood(start, allowed):
        stack, seen = [start], {start}
        while stack:
            y, x = stack.pop()
            for dy, dx in NEIGHBOURS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and allowed(ny, nx):
                    seen.add((ny, nx))
                    stack.append((ny, nx))
        return seen

    outside = flood((0, 0), lambda y, x: not blocked[y][x])
    enclosed = {(y, x) for y in range(height) for x in range(width)
                if not blocked[y][x] and (y, x) not in outside}
    count, left_over = 0, set(enclosed)
    while left_over:
        region = flood(next(iter(left_over)), lambda y, x: (y, x) in enclosed)
        left_over -= region
        count += 1
    return count, sorted((y + top - 1, x + left - 1) for y, x in enclosed)


def make_shape(colour, cells):
    shape = Shape()
    shape.colour = colour
    shape.cells = cells
    shape.cellset = frozenset(cells)
    shape.size = len(cells)
    shape.top = min(y for y, _ in cells)
    shape.left = min(x for _, x in cells)
    shape.bottom = max(y for y, _ in cells)
    shape.right = max(x for _, x in cells)
    shape.form = tuple(sorted((y - shape.top, x - shape.left) for y, x in cells))
    if shape.size >= 8:                      # the smallest possible ring has 8 cells
        shape.holes, shape.hole_cells = find_holes(shape.cellset, shape.top, shape.left,
                                                   shape.bottom, shape.right)
    else:
        shape.holes, shape.hole_cells = 0, []
    return shape


def extract_shapes(grid, background):
    height, width = len(grid), len(grid[0])
    seen = [[False] * width for _ in range(height)]
    shapes = []
    for r in range(height):
        for c in range(width):
            colour = grid[r][c]
            if seen[r][c] or colour == background:
                continue
            cells, stack = [], [(r, c)]
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny][nx] and grid[ny][nx] == colour:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            shapes.append(make_shape(colour, cells))
    return shapes


def hole_class(shapes):
    if not shapes:
        return "none"
    most = max(s.holes for s in shapes)
    return "0" if most == 0 else "1" if most == 1 else "2+"


# ----------------------------------------------------------------------
# Matching input shapes to output shapes
# ----------------------------------------------------------------------
def match_shapes(sin, sout, same_size):
    """List of (input index, output index, how). The rest are removed / added."""
    matches = []
    free_in, free_out = set(range(len(sin))), set(range(len(sout)))

    def take(i, j, how):
        matches.append((i, j, how))
        free_in.discard(i)
        free_out.discard(j)

    # exact: same cells in the same place
    by_cells = {sout[j].cellset: j for j in free_out}
    for i in sorted(free_in):
        j = by_cells.get(sin[i].cellset)
        if j is not None:
            take(i, j, "exact")

    def by_corner(a, b):
        return abs(a.top - b.top) + abs(a.left - b.left)

    def by_centre(a, b):
        return abs(a.top + a.bottom - b.top - b.bottom) + abs(a.left + a.right - b.left - b.right)

    def group_match(key, how, distance):
        groups_in, groups_out = defaultdict(list), defaultdict(list)
        for i in free_in:
            groups_in[key(sin[i])].append(i)
        for j in free_out:
            groups_out[key(sout[j])].append(j)
        for k, ins in groups_in.items():
            outs = groups_out.get(k)
            if not outs:
                continue
            used_in, used_out = set(), set()
            for _, i, j in sorted((distance(sin[i], sout[j]), i, j) for i in ins for j in outs):
                if i not in used_in and j not in used_out:
                    used_in.add(i)
                    used_out.add(j)
                    take(i, j, how)

    group_match(lambda s: (s.colour, s.form), "form", by_corner)
    group_match(lambda s: s.form, "recolour", by_corner)
    group_match(lambda s: (s.colour, canonical(s.form)), "rotated", by_centre)

    if same_size and free_in and free_out:
        out_at = {cell: j for j in free_out for cell in sout[j].cells}
        candidates = []
        for i in free_in:
            for j, n in Counter(out_at[c] for c in sin[i].cells if c in out_at).items():
                candidates.append((-n, sin[i].colour != sout[j].colour, i, j))
        for _, _, i, j in sorted(candidates):
            if i in free_in and j in free_out:
                take(i, j, "overlap")
    return matches, free_in, free_out


# ----------------------------------------------------------------------
# One pair
# ----------------------------------------------------------------------
def shape_record(s):
    return [s.colour, s.size, s.holes, s.top, s.left, s.bottom - s.top + 1, s.right - s.left + 1]


def describe_pair(grid_in, grid_out):
    bg_in, bg_out = find_background(grid_in), find_background(grid_out)
    sin, sout = extract_shapes(grid_in, bg_in), extract_shapes(grid_out, bg_out)
    same_size = (len(grid_in), len(grid_in[0])) == (len(grid_out), len(grid_out[0]))
    matches, removed, added = match_shapes(sin, sout, same_size)

    ev = set()
    if added: ev.add("SHAPES_ADDED")
    if removed: ev.add("SHAPES_REMOVED")
    if matches: ev.add("SHAPES_MAINTAINED")

    klass = hole_class(sin)
    if klass == "0": ev.add("SHAPE_HAS_0_HOLES")
    elif klass == "1": ev.add("SHAPE_HAS_1_HOLE")
    elif klass == "2+": ev.add("SHAPE_HAS_2_PLUS_HOLES")

    match_list = []
    for i, j, how in matches:
        a, b = sin[i], sout[j]
        dy, dx = b.top - a.top, b.left - a.left
        match_list.append([i, j, how, dy, dx])

        if b.size > a.size: ev.add("SHAPE_SIZE_ADDED")
        elif b.size < a.size: ev.add("SHAPE_SIZE_REDUCED")
        else: ev.add("SHAPE_SIZE_MAINTAINED")

        if how in ("exact", "form", "recolour"):
            ev.add("SHAPE_STRUCTURE_MAINTAINED")
        elif how == "overlap" and a.form != b.form:
            ev.add("SHAPE_STRUCTURE_REMOVED")
        if how == "recolour" or (how == "exact" and a.colour != b.colour):
            ev.add("SHAPE_RECOLOURED")
        if how == "rotated":
            ev.add("SHAPE_ROTATED_OR_FLIPPED")

        if same_size and how in ("form", "recolour"):
            if dx > 0: ev.add("SHAPE_MOVED_POSITIVE_X")
            if dx < 0: ev.add("SHAPE_MOVED_NEGATIVE_X")
            if dy > 0: ev.add("SHAPE_MOVED_POSITIVE_Y")
            if dy < 0: ev.add("SHAPE_MOVED_NEGATIVE_Y")

        if a.holes > 0 and how != "rotated":
            shift_y, shift_x = (dy, dx) if how in ("form", "recolour") else (0, 0)
            filled = False
            if bg_in is not None and bg_out is not None:
                for y, x in a.hole_cells:
                    ny, nx = y + shift_y, x + shift_x
                    if (grid_in[y][x] == bg_in and 0 <= ny < len(grid_out)
                            and 0 <= nx < len(grid_out[0]) and grid_out[ny][nx] != bg_out):
                        filled = True
                        break
            if filled: ev.add("SHAPE_HOLES_FILLED_IN")
            elif b.holes == a.holes: ev.add("SHAPE_HOLES_MAINTAINED")
            elif b.holes < a.holes: ev.add("SHAPE_HOLES_REMOVED")

    if len(sout) > len(sin): ev.add("SHAPE_COUNT_MORE")
    elif len(sout) < len(sin): ev.add("SHAPE_COUNT_FEWER")
    else: ev.add("SHAPE_COUNT_SAME")
    if not sin and not sout: ev.add("NO_SHAPES")

    return {"events": [e for e in EVENTS if e in ev],
            "shapes_in": len(sin), "shapes_out": len(sout),
            "matched": len(matches), "added": len(added), "removed": len(removed),
            "input_hole_class": klass, "output_hole_class": hole_class(sout),
            "input_background": bg_in, "output_background": bg_out,
            "input_shapes": [shape_record(s) for s in sin],
            "output_shapes": [shape_record(s) for s in sout],
            "matches": match_list}


# ----------------------------------------------------------------------
# Receipt (training pairs only)
# ----------------------------------------------------------------------
def get_shapes_receipt(task):
    pairs = [describe_pair(p["input"], p["output"]) for p in task["train"]]
    flags = [e for e in EVENTS if all(e in p["events"] for p in pairs)]
    sometimes = [e for e in EVENTS if any(e in p["events"] for p in pairs) and e not in flags]

    tests = []
    for t in task["test"]:
        bg = find_background(t["input"])
        shapes = extract_shapes(t["input"], bg)
        tests.append({"background": bg, "shape_count": len(shapes),
                      "hole_class": hole_class(shapes), "shapes": [shape_record(s) for s in shapes]})
    return {"flags": flags, "sometimes": sometimes, "pairs": pairs, "test_inputs": tests}


def check_on_public_test(receipt, task):
    """Which flags also hold on the public test pair(s)? Verification only."""
    if not all("output" in t for t in task["test"]):
        return None
    test_events = [set(describe_pair(t["input"], t["output"])["events"]) for t in task["test"]]
    holding = [f for f in receipt["flags"] if all(f in e for e in test_events)]
    failing = [f for f in receipt["flags"] if f not in holding]
    return {"holding": holding, "failing": failing}


# ----------------------------------------------------------------------
# For the other steps
# ----------------------------------------------------------------------
def load_shapes_receipts(path=OUT_JSON):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid shapes receipts not found: {path}\n"
                         f"Run solver/grid_shapes/grid_shapes.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_shapes_receipts(receipts, dataset, task_ids):
    """Stop if any task has no grid shapes receipt, and say which ones."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid shapes receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_shapes/grid_shapes.py.")
    print(f"Grid shapes receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def csv_row(dataset, task_id, task, receipt, check):
    pairs = receipt["pairs"]
    return {
        "dataset": dataset, "task_id": task_id, "n_train_pairs": len(task["train"]),
        "flags": "|".join(receipt["flags"]),
        "sometimes": "|".join(receipt["sometimes"]),
        "shapes_in": ";".join(str(p["shapes_in"]) for p in pairs),
        "shapes_out": ";".join(str(p["shapes_out"]) for p in pairs),
        "matched": ";".join(str(p["matched"]) for p in pairs),
        "added": ";".join(str(p["added"]) for p in pairs),
        "removed": ";".join(str(p["removed"]) for p in pairs),
        "input_hole_class": ";".join(p["input_hole_class"] for p in pairs),
        "output_hole_class": ";".join(p["output_hole_class"] for p in pairs),
        "test_shape_count": ";".join(str(t["shape_count"]) for t in receipt["test_inputs"]),
        "flags_holding_on_test": "unknown" if check is None else "|".join(check["holding"]),
        "flags_failing_on_test": "unknown" if check is None else "|".join(check["failing"]),
    }


def print_summary(dataset, rows):
    total = len(rows)
    print(f"\n=== {dataset}: grid shapes ({total} tasks) ===")
    header = f"{'Event':<30}{'In every pair':>14}{'Holds on test':>16}{'Sometimes':>11}"
    print(header)
    print("-" * len(header))
    for event in EVENTS:
        flagged = [r for r in rows if event in r["flags"].split("|")]
        sometimes = sum(event in r["sometimes"].split("|") for r in rows)
        checked = [r for r in flagged if r["flags_holding_on_test"] != "unknown"]
        holds = sum(event in r["flags_holding_on_test"].split("|") for r in checked)
        holds_text = f"{holds}/{len(checked)}" if checked else "-"
        print(f"{event:<30}{len(flagged):>14}{holds_text:>16}{sometimes:>11}")
    no_flags = sum(1 for r in rows if not r["flags"])
    print(f"\nTasks with no flag true in every pair: {no_flags}/{total}")


def main():
    all_rows, receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        rows, receipts[dataset] = [], {}
        for task_id, task in load_tasks(folder).items():
            receipt = get_shapes_receipt(task)
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