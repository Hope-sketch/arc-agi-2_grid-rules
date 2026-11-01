"""
Step 6: Grid transformations (investigation only, feeds the Grid Solver)

Does two things for every task.

PART 1  COLLECT. Gathers the receipts of steps 1-5 (sizes, colours, shapes, cells,
rows and columns) into one transformation receipt:
  * "steps"      the flags of every step
  * "flags"      all flags that are true in every training pair, "sometimes" = some pairs
  * "combined"   flags that only appear when steps are read together
                 (COMBINED_NEAR_COPY, COMBINED_RECOLOUR_ONLY, COMBINED_MOVE_ONLY,
                  COMBINED_ADD_COLOURED_SHAPES, COMBINED_REMOVE_SHAPES_AND_COLOURS,
                  COMBINED_CROP, COMBINED_FILL_HOLES, COMBINED_RESIZE_SHAPES,
                  COMBINED_ROTATE_OR_FLIP, COMBINED_REARRANGE_LINES)
  * "group", "story"  a short label for the kind of task (size / palette / shape count)

PART 2  LEARN MORE. Looks for transformations nobody described yet, by trying a
library on the training pairs. A candidate "fits" if it reproduces EVERY training
output (after one learned colour map):
  whole_grid   flips, rotations, transpose, mirrors, tiling, scaling up and down,
               halves, crop to the non-background, crop to the box of a colour
               (outer or inner), shift by up to 3 cells, plain recolouring
  local_rule   same-size tasks: what a cell becomes, learned from its neighbourhood
Candidates are listed simplest first ("best" = the first). If nothing fits, the three
closest are kept as "partial" (how many training pairs they reproduce exactly).
Also stored: size_ratio when output/input size is a constant ratio in every pair.

The Grid Solver predicts with these two functions:
  specs = find_transformations(train_pairs)       # train_pairs: [(input, output), ...]
  prediction, coverage = apply_transformation(specs[0], test_input)
The receipt JSON holds the compact description of each candidate; the learned local
rule itself lives in memory (it is rebuilt by find_transformations).

The receipt never contains the test output. The public test output is only used for
the "reproduces test" checks in the CSV and tables (a check, not a score).

Usage
  python solver/grid_transformations/grid_transformations.py
  (run steps 1-5 first, they write the receipts this step reads)

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_transformations/results/grid_transformations_results.csv
  * solver/grid_transformations/results/grid_transformations_receipts.json

This file does not need whole_grid_library.py.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_transformations_results.csv"
OUT_JSON = RESULTS_DIR / "grid_transformations_receipts.json"
DATASETS = ("training", "evaluation")

for step in ("grid_sizes", "grid_colours", "grid_shapes", "grid_cells", "grid_rows_and_columns"):
    sys.path.insert(0, str(ROOT / "solver" / step))
try:
    from grid_colours import find_background, load_colours_receipts, check_colours_receipts
    from grid_sizes import load_receipts as load_sizes_receipts, check_receipts as check_sizes_receipts
    from grid_shapes import load_shapes_receipts, check_shapes_receipts
    from grid_cells import load_cells_receipts, check_cells_receipts
    from grid_rows_and_columns import load_rows_columns_receipts, check_rows_columns_receipts
except ImportError as error:
    raise SystemExit(f"Could not import a step script ({error}).\n"
                     "Steps 1-5 (grid_sizes, grid_colours, grid_shapes, grid_cells, "
                     "grid_rows_and_columns) must be in their solver/ folders.")

NEIGHBOUR_KINDS = ["cross_rel", "block3_rel", "cross", "block3"]
SHIFT_RANGE = 3


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
# Whole-grid library
# ----------------------------------------------------------------------
def background_of(g):
    return find_background(g.tolist())


def crop_non_background(g):
    bg = background_of(g)
    if bg is None:
        return None
    ys, xs = np.where(g != bg)
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else None


def crop_colour_box(g, colour, inner):
    ys, xs = np.where(g == colour)
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if inner:
        y0, x0, y1, x1 = y0 + 1, x0 + 1, y1 - 1, x1 - 1
    return g[y0:y1 + 1, x0:x1 + 1] if y1 >= y0 and x1 >= x0 else None


def shift(g, dy, dx):
    bg = background_of(g)
    out = np.full(g.shape, 0 if bg is None else bg, dtype=int)
    h, w = g.shape
    src_y, src_x = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
    dst_y, dst_x = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
    out[dst_y, dst_x] = g[src_y, src_x]
    return out


def downscale(g, k):
    h, w = g.shape
    if h % k or w % k:
        return None
    small = g[::k, ::k]
    return small if np.array_equal(np.kron(small, np.ones((k, k), int)), g) else None


def build_registry():
    r = {"identity": lambda g: g,
         "flip_h": lambda g: g[:, ::-1], "flip_v": lambda g: g[::-1, :],
         "rotate_90": lambda g: np.rot90(g, -1), "rotate_180": lambda g: np.rot90(g, 2),
         "rotate_270": lambda g: np.rot90(g, 1), "transpose": lambda g: g.T,
         "anti_transpose": lambda g: np.rot90(g, 2).T,
         "mirror_h": lambda g: np.hstack([g, g[:, ::-1]]),
         "mirror_h_reversed": lambda g: np.hstack([g[:, ::-1], g]),
         "mirror_v": lambda g: np.vstack([g, g[::-1, :]]),
         "mirror_v_reversed": lambda g: np.vstack([g[::-1, :], g]),
         "mirror_both": lambda g: np.vstack([np.hstack([g, g[:, ::-1]]),
                                             np.hstack([g[::-1, :], np.rot90(g, 2)])]),
         "top_half": lambda g: g[:g.shape[0] // 2] if g.shape[0] > 1 else None,
         "bottom_half": lambda g: g[(g.shape[0] + 1) // 2:] if g.shape[0] > 1 else None,
         "left_half": lambda g: g[:, :g.shape[1] // 2] if g.shape[1] > 1 else None,
         "right_half": lambda g: g[:, (g.shape[1] + 1) // 2:] if g.shape[1] > 1 else None,
         "crop_non_background": crop_non_background}
    for ry in range(1, 6):
        for rx in range(1, 6):
            if (ry, rx) != (1, 1):
                r[f"tile_{ry}x{rx}"] = lambda g, ry=ry, rx=rx: np.tile(g, (ry, rx))
    for k in range(2, 7):
        r[f"upscale_{k}"] = lambda g, k=k: np.kron(g, np.ones((k, k), int))
        r[f"downscale_{k}"] = lambda g, k=k: downscale(g, k)
    for colour in range(10):
        r[f"crop_box_colour_{colour}"] = lambda g, c=colour: crop_colour_box(g, c, False)
        r[f"crop_inner_box_colour_{colour}"] = lambda g, c=colour: crop_colour_box(g, c, True)
    for dy in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
        for dx in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
            if (dy, dx) != (0, 0):
                r[f"shift_dy{dy}_dx{dx}"] = lambda g, dy=dy, dx=dx: shift(g, dy, dx)
    return r


REGISTRY = build_registry()


def learn_colour_map(predictions, outputs):
    """One colour map that turns every prediction into its output, or None."""
    mapping = {}
    for p, o in zip(predictions, outputs):
        for src, dst in set(zip(p.ravel().tolist(), o.ravel().tolist())):
            if mapping.setdefault(src, dst) != dst:
                return None
    return mapping


# ----------------------------------------------------------------------
# Local rule (same-size tasks)
# ----------------------------------------------------------------------
def neighbourhood_keys(g, kind):
    p = np.pad(g, 1, constant_values=-1)                  # -1 = outside the grid
    sides = [p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]
    corners = [p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:]]
    if kind == "cross_rel":
        parts = [g] + [n == g for n in sides]
    elif kind == "block3_rel":
        parts = [g] + [n == g for n in sides + corners]
    elif kind == "cross":
        parts = [g] + sides
    else:
        parts = [g] + sides + corners
    return list(zip(*[a.ravel().tolist() for a in parts]))


def learn_local_rule(train, kind):
    rule = {}
    for grid_in, grid_out in train:
        for key, out in zip(neighbourhood_keys(grid_in, kind), grid_out.ravel().tolist()):
            if rule.setdefault(key, out) != out:
                return None
    return rule


# ----------------------------------------------------------------------
# Learn and apply (the Grid Solver uses these two)
# ----------------------------------------------------------------------
def find_transformations(train):
    """Candidate specs that reproduce every training pair, simplest first.
    train = [(input_array, output_array), ...]. Returns (specs, partial)."""
    exact, with_map, partial = [], [], []
    for name, function in REGISTRY.items():
        predictions = []
        for grid_in, grid_out in train:
            try:
                p = function(grid_in)
            except Exception:
                p = None
            if p is None or p.shape != grid_out.shape:
                predictions = None
                break
            predictions.append(p)
        if predictions is None:
            continue
        mapping = learn_colour_map(predictions, [o for _, o in train])
        if mapping is not None:
            changes = {str(k): v for k, v in mapping.items() if k != v}
            spec = {"kind": "whole_grid", "name": name, "colour_map": changes, "exact": not changes}
            (exact if not changes else with_map).append(spec)
        else:
            fit = sum(np.array_equal(p, o) for p, (_, o) in zip(predictions, train))
            if fit:
                partial.append([name, fit])

    specs = exact + with_map
    if all(i.shape == o.shape for i, o in train):
        for kind in NEIGHBOUR_KINDS:
            rule = learn_local_rule(train, kind)
            if rule is not None:
                specs.append({"kind": "local_rule", "name": f"local_{kind}", "neighbourhood": kind,
                              "rule_size": len(rule), "rule": rule, "exact": False})
                break
    partial.sort(key=lambda item: -item[1])
    return specs, partial[:3]


def apply_transformation(spec, grid):
    """(prediction, coverage). Coverage is the share of cells a local rule knew (else None)."""
    g = to_np(grid)
    if spec["kind"] == "whole_grid":
        p = REGISTRY[spec["name"]](g)
        if p is None:
            return None, None
        mapping = {int(k): v for k, v in spec["colour_map"].items()}
        return np.vectorize(lambda c: mapping.get(c, c))(p) if mapping else p, None
    rule = spec["rule"]
    keys = neighbourhood_keys(g, spec["neighbourhood"])
    covered = [k in rule for k in keys]
    p = np.array([rule.get(k, c) for k, c in zip(keys, g.ravel().tolist())]).reshape(g.shape)
    return p, sum(covered) / len(covered)


def public_spec(spec):
    return {k: v for k, v in spec.items() if k != "rule"}


# ----------------------------------------------------------------------
# Part 1: collect the receipts of steps 1-5
# ----------------------------------------------------------------------
SIZE_FLAG = {"1": "SIZE_SAME_AS_INPUT", "2": "SIZE_SAME_AS_TRAINING_OUTPUTS", "3": "SIZE_UNSOLVED"}
SIZE_TEXT = {"1": "same size", "2": "fixed output size", "3": "size unsolved"}
SIMILARITY_TEXT = {"very_similar": "near copy", "somewhat_similar": "partly alike",
                   "not_similar": "different", "mixed": "mixed similarity"}
PALETTE_TEXT = {"added": "colours added", "removed": "colours removed",
                "added_and_removed": "colours added and removed", "same": "colours kept",
                "mixed": "colours mixed"}


def combine(steps):
    """Flags that only appear when several steps are read together."""
    f = set(steps["flags"])
    moved = bool(f & {"COLOUR_MOVED_X", "COLOUR_MOVED_Y", "SHAPE_MOVED_POSITIVE_X", "SHAPE_MOVED_NEGATIVE_X",
                      "SHAPE_MOVED_POSITIVE_Y", "SHAPE_MOVED_NEGATIVE_Y"})
    same_size = "SAME_SIZE" in f
    combined = []
    if same_size and "CELLS_VERY_SIMILAR" in f:
        combined.append("COMBINED_NEAR_COPY")
    if same_size and "SHAPE_COUNT_SAME" in f and "SHAPE_RECOLOURED" in f:
        combined.append("COMBINED_RECOLOUR_ONLY")
    if same_size and moved and "SHAPE_COUNT_SAME" in f and "COLOURS_SAME" in f:
        combined.append("COMBINED_MOVE_ONLY")
    if {"COLOURS_ADDED", "SHAPES_ADDED"} <= f:
        combined.append("COMBINED_ADD_COLOURED_SHAPES")
    if {"COLOURS_REMOVED", "SHAPES_REMOVED"} <= f:
        combined.append("COMBINED_REMOVE_SHAPES_AND_COLOURS")
    if "OUTPUT_SMALLER" in f and "OUTPUT_IS_PIECE_OF_INPUT" in f:
        combined.append("COMBINED_CROP")
    if "SHAPE_HOLES_FILLED_IN" in f:
        combined.append("COMBINED_FILL_HOLES")
    if same_size and "SHAPE_COUNT_SAME" in f and f & {"SHAPE_SIZE_ADDED", "SHAPE_SIZE_REDUCED"}:
        combined.append("COMBINED_RESIZE_SHAPES")
    if f & {"SIMILAR_AFTER_FLIP_OR_ROTATION", "SHAPE_ROTATED_OR_FLIPPED"}:
        combined.append("COMBINED_ROTATE_OR_FLIP")
    if f & {"ROWS_REORDERED", "COLUMNS_REORDERED"}:
        combined.append("COMBINED_REARRANGE_LINES")
    return combined


def collect(sizes, colours, shapes, cells, rows_columns):
    flags = [SIZE_FLAG[sizes["category"]]] + colours["flags"] + shapes["flags"] \
        + cells["flags"] + rows_columns["flags"]
    flags = list(dict.fromkeys(flags))
    sometimes = [f for f in dict.fromkeys(shapes["sometimes"] + cells["sometimes"] + rows_columns["sometimes"])
                 if f not in flags]
    steps = {"sizes": {"category": sizes["category"], "report": sizes["report"],
                       "predicted_output_sizes": sizes["predicted_output_sizes"]},
             "colours": {"flags": colours["flags"], "background_colour": colours["background_colour"],
                         "colours_always_added": colours["colours_always_added"],
                         "colours_always_removed": colours["colours_always_removed"]},
             "shapes": {"flags": shapes["flags"], "sometimes": shapes["sometimes"]},
             "cells": {"flags": cells["flags"], "sometimes": cells["sometimes"],
                       "mean_share": cells["mean_share"]},
             "rows_columns": {"flags": rows_columns["flags"], "sometimes": rows_columns["sometimes"],
                              "consistent_shift": rows_columns["consistent_shift"]},
             "flags": flags}
    count_word = ("more" if "SHAPE_COUNT_MORE" in flags else "fewer" if "SHAPE_COUNT_FEWER" in flags
                  else "same" if "SHAPE_COUNT_SAME" in flags else "varies")
    story = ", ".join([SIZE_TEXT[sizes["category"]], SIMILARITY_TEXT[cells["similarity_word"]],
                       f"{count_word} shapes" if count_word != "varies" else "shape count varies",
                       PALETTE_TEXT[colours["palette_word"]]])
    return {"steps": {k: v for k, v in steps.items() if k != "flags"}, "flags": flags,
            "sometimes": sometimes, "combined": combine(steps),
            "group": f"{sizes['category']}|{colours['palette_word']}|{count_word}", "story": story}


# ----------------------------------------------------------------------
# Receipt
# ----------------------------------------------------------------------
def size_ratio(train):
    ratios = {(round(o.shape[0] / i.shape[0], 3), round(o.shape[1] / i.shape[1], 3)) for i, o in train}
    return list(next(iter(ratios))) if len(ratios) == 1 else None


def get_transformations_receipt(task, step_receipts):
    receipt = collect(**step_receipts)
    train = [(to_np(p["input"]), to_np(p["output"])) for p in task["train"]]
    specs, partial = find_transformations(train)
    receipt["size_ratio"] = size_ratio(train)
    receipt["candidates"] = [public_spec(s) for s in specs]
    receipt["best"] = specs[0]["name"] if specs else None
    receipt["partial"] = [] if specs else partial
    return receipt, specs


def check_on_public_test(specs, task):
    """How does each candidate do on the public test pair(s)? Verification only."""
    if not specs or not all("output" in t for t in task["test"]):
        return None
    results = []
    for spec in specs:
        status = "exact"
        for t in task["test"]:
            try:
                p, coverage = apply_transformation(spec, t["input"])
            except Exception:
                p, coverage = None, None
            if p is None or p.shape != to_np(t["output"]).shape or not np.array_equal(p, to_np(t["output"])):
                status = "wrong"
                break
            if coverage is not None and coverage < 1:
                status = "exact_fallback"
        results.append(status)
    return results


# ----------------------------------------------------------------------
# For the other steps
# ----------------------------------------------------------------------
def load_transformations_receipts(path=OUT_JSON):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid transformations receipts not found: {path}\n"
                         f"Run solver/grid_transformations/grid_transformations.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_transformations_receipts(receipts, dataset, task_ids):
    """Stop if any task has no grid transformations receipt, and say which ones."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid transformations receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_transformations/grid_transformations.py.")
    print(f"Grid transformations receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def csv_row(dataset, task_id, receipt, results):
    candidates = receipt["candidates"]
    best_test = "unknown" if results is None else results[0]
    any_test = ("unknown" if results is None else
                "exact" if "exact" in results else "exact_fallback" if "exact_fallback" in results else "wrong")
    return {
        "dataset": dataset, "task_id": task_id, "group": receipt["group"], "story": receipt["story"],
        "combined": "|".join(receipt["combined"]), "flags": "|".join(receipt["flags"]),
        "sometimes": "|".join(receipt["sometimes"]),
        "size_ratio": "" if receipt["size_ratio"] is None else f"{receipt['size_ratio'][0]}x{receipt['size_ratio'][1]}",
        "n_candidates": len(candidates), "best": receipt["best"] or "",
        "best_kind": candidates[0]["kind"] if candidates else "",
        "best_has_colour_map": ("" if not candidates else
                                "yes" if candidates[0].get("colour_map") else "no"),
        "candidates": "|".join(c["name"] for c in candidates[:10]),
        "partial": "|".join(f"{n}:{k}" for n, k in receipt["partial"]),
        "best_on_test": best_test, "any_candidate_on_test": any_test,
    }


def print_summary(dataset, rows):
    total = len(rows)
    with_candidate = [r for r in rows if r["n_candidates"]]
    checked = [r for r in with_candidate if r["best_on_test"] != "unknown"]
    hit = lambda r: r["best_on_test"] in ("exact", "exact_fallback")
    print(f"\n=== {dataset}: grid transformations ({total} tasks) ===")
    print(f"Tasks with a candidate that fits every training pair: {len(with_candidate)} "
          f"({100 * len(with_candidate) / total:.1f}%)")
    print(f"  best candidate reproduces the public test output : {sum(map(hit, checked))}/{len(checked)}")
    print(f"  any candidate reproduces it                      : "
          f"{sum(r['any_candidate_on_test'] in ('exact', 'exact_fallback') for r in checked)}/{len(checked)}")

    print("\nBy kind of best candidate")
    header = f"{'Kind':<24}{'Tasks':>7}{'Reproduce test':>16}"
    print(header)
    print("-" * len(header))
    by_kind = defaultdict(list)
    for r in checked:
        by_kind["local_rule" if r["best_kind"] == "local_rule"
                else "whole_grid + colour map" if r["best_has_colour_map"] == "yes"
                else "whole_grid exact"].append(r)
    for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        print(f"{kind:<24}{len(group):>7}{sum(map(hit, group)):>16}")

    print("\nMost common best candidates")
    for name, n in Counter(r["best"] for r in with_candidate).most_common(8):
        group = [r for r in checked if r["best"] == name]
        print(f"  {name:<28}{n:>5}   reproduce test: {sum(map(hit, group))}/{len(group)}")

    print("\nCombined flags (steps read together)")
    header = f"{'Flag':<38}{'Tasks':>7}{'With candidate':>16}{'Reproduce test':>16}"
    print(header)
    print("-" * len(header))
    names = sorted({c for r in rows for c in r["combined"].split("|") if c})
    for name in names:
        group = [r for r in rows if name in r["combined"].split("|")]
        has = [r for r in group if r["n_candidates"]]
        ok = [r for r in has if r["best_on_test"] != "unknown"]
        print(f"{name:<38}{len(group):>7}{len(has):>16}{sum(map(hit, ok)):>16}")
    print(f"Tasks with no combined flag: {sum(1 for r in rows if not r['combined'])}/{total}")


def main():
    receipts_in = {
        "sizes": load_sizes_receipts(), "colours": load_colours_receipts(),
        "shapes": load_shapes_receipts(), "cells": load_cells_receipts(),
        "rows_columns": load_rows_columns_receipts(),
    }
    checkers = {"sizes": check_sizes_receipts, "colours": check_colours_receipts,
                "shapes": check_shapes_receipts, "cells": check_cells_receipts,
                "rows_columns": check_rows_columns_receipts}

    all_rows, receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        tasks = load_tasks(folder)
        for step, check in checkers.items():
            check(receipts_in[step], dataset, list(tasks))

        rows, receipts[dataset] = [], {}
        for task_id, task in tasks.items():
            step_receipts = {name: receipts_in[name][dataset][task_id] for name in checkers}
            receipt, specs = get_transformations_receipt(task, step_receipts)
            receipts[dataset][task_id] = receipt
            rows.append(csv_row(dataset, task_id, receipt, check_on_public_test(specs, task)))
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