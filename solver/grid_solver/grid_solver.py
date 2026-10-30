"""
Step 7 (part 2): the Grid Solver

Predicts the test output of every task using ONLY the training pairs and the test
input. The test answers are never read here (grid_verifier.py checks them later).

How it connects the steps
  1. SORT      the task's group name (size, palette, shape, detail) comes from
               grid_sorter.py's profiles when the task is known there.
  2. LEARN     three methods each try to learn a transformation from the training
               pairs. A method only counts when it reproduces EVERY training output.
                 whole_grid    : a flip / rotation / mirror / tile / scale / crop (and one
                                 colour map), from the transformations step
                 select_piece  : the output is a piece of the input; learns WHICH piece
                                 (largest, smallest, topmost, unique form, unique colour ...)
                 local_rule    : what each cell becomes from its neighbourhood (same-size
                                 tasks), from the transformations step
  3. RANK      every learned method gives a candidate prediction with a confidence:
                 whole_grid 3.0 (2.8 with a colour map), select_piece 2.5 and below,
                 local_rule 1.0 + the share of test cells whose neighbourhood was seen
               plus a small bonus (up to 0.5) from the group: the share of TRAINING tasks
               in the same group where that kind of method fitted. The task itself is left
               out of its own bonus.
  4. ATTEMPTS  attempt_1 and attempt_2 are the two best DIFFERENT predictions. If fewer than
               two methods fit, simple fallbacks fill in (the input itself, or its crop).

Usage
  python solver/grid_solver/grid_solver.py evaluation      (or training, or a folder path)

Outputs
  * results/grid_solver_predictions_<name>.json   submission format:
        {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...one per test input]}
  * results/grid_solver_results_<name>.csv        one row per task: group and methods used
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOLVER = next((p for p in Path(__file__).resolve().parents if p.name == "solver"), HERE.parent)
ROOT = SOLVER.parent
DATA_DIR = ROOT / "data"
OUT_DIR = SOLVER / "grid_solver" / "results"
PROFILES_CSV = OUT_DIR / "grid_profiles.csv"

sys.path.insert(0, str(SOLVER / "grid_transformations"))
try:
    from whole_grid_library import (apply_map, background, crop_non_background,   # noqa: E402
                                    fixed_transforms, learn_colour_map,
                                    parameterised_transforms, safe_apply)
    from grid_transformations import NEIGHBOUR_KINDS, learn_local_rule, neighbourhood_keys   # noqa: E402
except (ImportError, ModuleNotFoundError) as error:
    raise SystemExit(
        f"Cannot load the transformations step ({error}).\n"
        f"solver/grid_transformations/ must contain the NEW grid_transformations.py "
        f"and whole_grid_library.py.")

SIDES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_SELECT_CANDIDATES = 4


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
def to_np(grid):
    return np.array(grid, dtype=int)


def load_tasks(folder):
    """Training pairs and TEST INPUTS only. Test outputs are dropped on purpose."""
    if not folder.is_dir():
        raise SystemExit(f"Data folder not found: {folder}")
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        tasks[file_path.stem] = {
            "train": [(to_np(p["input"]), to_np(p["output"])) for p in raw["train"]],
            "test_inputs": [to_np(p["input"]) for p in raw["test"]],
        }
    return tasks


def load_profiles():
    if not PROFILES_CSV.exists():
        print(f"Note: {PROFILES_CSV.name} not found (run grid_sorter.py): no group bonus, groups left empty")
        return {}, {}
    with PROFILES_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    profiles = {(r["dataset"], r["task_id"]): r for r in rows}
    by_group = defaultdict(list)
    for r in rows:
        if r["dataset"] == "training" and int(r["backoff_level"]) > 0:
            by_group[r[f"level{r['backoff_level']}"]].append(r)
    return profiles, by_group


def group_of(profile):
    level = int(profile["backoff_level"])
    return profile[f"level{level}"] if level > 0 else ""


FITTED = {
    "whole_grid": lambda r: r["whole_grid"] in ("found_exact", "found_with_colour_map"),
    "select_piece": lambda r: r["contained"] not in ("", "n/a", "none"),
    "local_rule": lambda r: r["local_kind"] not in ("", "n/a", "none"),
}


def group_bonus(family, group, by_group, dataset, task_id):
    rows = [r for r in by_group.get(group, [])
            if not (dataset == "training" and r["task_id"] == task_id)]
    if not rows:
        return 0.0
    return 0.5 * sum(1 for r in rows if FITTED[family](r)) / len(rows)


# ----------------------------------------------------------------------
# Method 1: whole-grid transformations
# ----------------------------------------------------------------------
def whole_grid_candidates(train, test_inputs):
    library = fixed_transforms()
    library.update(parameterised_transforms(*train[0]))
    found = []
    for name, fn in library.items():
        preds = [safe_apply(fn, i) for i, _ in train]
        exact = all(p is not None and p.shape == o.shape and np.array_equal(p, o)
                    for p, (_, o) in zip(preds, train))
        mapping = None
        if not exact:
            mapping = learn_colour_map(preds, [o for _, o in train])
            if mapping is None:
                continue
        outputs = []
        for grid in test_inputs:
            p = safe_apply(fn, grid)
            if p is not None and mapping is not None:
                p = apply_map(p, mapping)
            if p is None:
                break
            outputs.append(p)
        else:
            label = f"whole_grid:{name}" + ("+colour_map" if mapping else "")
            found.append({"family": "whole_grid", "method": label,
                          "confidence": 2.8 if mapping else 3.0, "predictions": outputs})
    return found


# ----------------------------------------------------------------------
# Method 2: select a piece of the input
# ----------------------------------------------------------------------
FAMILIES = ["object8", "object4", "shape4", "bbox_all"]
VARIANTS = ["raw", "masked"]

PROPERTIES = {
    "cells": lambda c, g: len(c["cells"]),
    "box_area": lambda c, g: (c["box"][2] - c["box"][0] + 1) * (c["box"][3] - c["box"][1] + 1),
    "colours": lambda c, g: len({int(g[y, x]) for y, x in c["cells"]}),
    "top": lambda c, g: c["box"][0],
    "left": lambda c, g: c["box"][1],
    "bottom": lambda c, g: c["box"][2],
    "right": lambda c, g: c["box"][3],
}
FLAGS = ["unique_form", "unique_colour", "touches_border", "symmetric"]
SELECTORS = ([("max", p) for p in PROPERTIES] + [("min", p) for p in PROPERTIES]
             + [("flag_true", f) for f in FLAGS] + [("flag_false", f) for f in FLAGS])


def components(grid, bg, family):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    steps = SIDES + DIAGONALS if family == "object8" else SIDES
    comps = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or grid[r, c] == bg:
                continue
            colour = grid[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and not seen[ny, nx]
                            and grid[ny, nx] != bg
                            and (family != "shape4" or grid[ny, nx] == colour)):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def make_candidates(grid, family):
    bg = int(background(grid))
    if family == "bbox_all":
        cells = [(int(y), int(x)) for y, x in zip(*np.where(grid != bg))]
        comps = [cells] if cells else []
    else:
        comps = components(grid, bg, family)
    candidates = []
    for cells in comps:
        ys, xs = [y for y, _ in cells], [x for _, x in cells]
        candidates.append({"cells": cells, "box": (min(ys), min(xs), max(ys), max(xs))})
    return candidates, bg


def crop(grid, cand, variant, bg):
    r0, c0, r1, c1 = cand["box"]
    if variant == "raw":
        return grid[r0:r1 + 1, c0:c1 + 1]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for y, x in cand["cells"]:
        out[y - r0, x - c0] = grid[y, x]
    return out


def flag_values(grid, candidates, bg):
    height, width = grid.shape
    forms = [frozenset((y - c["box"][0], x - c["box"][1]) for y, x in c["cells"]) for c in candidates]
    dominant = [Counter(int(grid[y, x]) for y, x in c["cells"]).most_common(1)[0][0]
                for c in candidates]
    form_count, colour_count = Counter(forms), Counter(dominant)
    symmetric = []
    for c in candidates:
        piece = crop(grid, c, "raw", bg)
        symmetric.append(bool(np.array_equal(piece, piece[:, ::-1]) or np.array_equal(piece, piece[::-1])))
    return {
        "unique_form": [form_count[f] == 1 for f in forms],
        "unique_colour": [colour_count[d] == 1 for d in dominant],
        "touches_border": [c["box"][0] == 0 or c["box"][1] == 0 or c["box"][2] == height - 1
                           or c["box"][3] == width - 1 for c in candidates],
        "symmetric": symmetric,
    }


def choose(grid, candidates, flags, selector):
    kind, name = selector
    if kind in ("max", "min"):
        values = [PROPERTIES[name](c, grid) for c in candidates]
        target = max(values) if kind == "max" else min(values)
        hits = [i for i, v in enumerate(values) if v == target]
    else:
        wanted = kind == "flag_true"
        hits = [i for i, f in enumerate(flags[name]) if f == wanted]
    return hits[0] if len(hits) == 1 else None


def prepare(grid, family):
    candidates, bg = make_candidates(grid, family)
    flags = flag_values(grid, candidates, bg) if candidates else {}
    return candidates, bg, flags


def select_piece_candidates(train, test_inputs):
    found = []
    for family in FAMILIES:
        prepared = [prepare(i, family) for i, _ in train]
        if not all(p[0] for p in prepared):
            continue
        for variant in VARIANTS:
            for selector in SELECTORS:
                ok = True
                for (grid_in, grid_out), (cands, bg, flags) in zip(train, prepared):
                    index = choose(grid_in, cands, flags, selector)
                    if index is None:
                        ok = False
                        break
                    piece = crop(grid_in, cands[index], variant, bg)
                    if piece.shape != grid_out.shape or not np.array_equal(piece, grid_out):
                        ok = False
                        break
                if not ok:
                    continue
                outputs = []
                for grid in test_inputs:
                    cands, bg, flags = prepare(grid, family)
                    index = choose(grid, cands, flags, selector) if cands else None
                    if index is None:
                        break
                    outputs.append(crop(grid, cands[index], variant, bg))
                else:
                    label = f"select_piece:{family}/{variant}/{selector[0]} {selector[1]}"
                    found.append({"family": "select_piece", "method": label,
                                  "confidence": 2.5 - 0.05 * len(found), "predictions": outputs})
                    if len(found) >= MAX_SELECT_CANDIDATES:
                        return found
    return found


# ----------------------------------------------------------------------
# Method 3: local rule (same-size tasks)
# ----------------------------------------------------------------------
def local_rule_candidates(train, test_inputs):
    if not all(i.shape == o.shape for i, o in train):
        return []
    found = []
    for kind in NEIGHBOUR_KINDS:
        rule = learn_local_rule(train, kind)
        if rule is None:
            continue
        outputs, coverage = [], []
        for grid in test_inputs:
            keys = neighbourhood_keys(grid, kind)
            coverage.append(sum(k in rule for k in keys) / len(keys))
            cells = [rule.get(k, c) for k, c in zip(keys, grid.ravel().tolist())]
            outputs.append(np.array(cells).reshape(grid.shape))
        found.append({"family": "local_rule", "method": f"local_rule:{kind}",
                      "confidence": 1.0 + min(coverage), "predictions": outputs})
    return found


# ----------------------------------------------------------------------
# Fallbacks and ranking
# ----------------------------------------------------------------------
def fallback_candidates(train, test_inputs):
    same_size = all(i.shape == o.shape for i, o in train)
    options = [("fallback:copy_input", lambda g: g), ("fallback:crop", crop_non_background)]
    if not same_size:
        options.reverse()
    found = []
    for confidence, (name, fn) in zip((0.1, 0.05), options):
        outputs = [safe_apply(fn, g) for g in test_inputs]
        if all(o is not None for o in outputs):
            found.append({"family": "fallback", "method": name,
                          "confidence": confidence, "predictions": outputs})
    return found


def signature(candidate):
    return tuple((p.shape, p.tobytes()) for p in candidate["predictions"])


def solve_task(task, group, by_group, dataset, task_id):
    train, test_inputs = task["train"], task["test_inputs"]
    learned = (whole_grid_candidates(train, test_inputs)
               + select_piece_candidates(train, test_inputs)
               + local_rule_candidates(train, test_inputs))
    for c in learned:
        c["score"] = c["confidence"] + (group_bonus(c["family"], group, by_group, dataset, task_id)
                                        if group else 0.0)
    learned.sort(key=lambda c: -c["score"])

    chosen, seen = [], set()
    for c in learned + fallback_candidates(train, test_inputs):
        key = signature(c)
        if key not in seen:
            seen.add(key)
            c.setdefault("score", c["confidence"])
            chosen.append(c)
        if len(chosen) == 2:
            break
    while len(chosen) < 2:                       # last resort: repeat what we have
        chosen.append(chosen[-1])
    return chosen, len(learned)


# ----------------------------------------------------------------------
def main():
    name_or_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    folder = Path(name_or_path) if Path(name_or_path).is_dir() else DATA_DIR / name_or_path
    dataset = folder.name
    tasks = load_tasks(folder)
    profiles, by_group = load_profiles()

    predictions, rows = {}, []
    for number, (task_id, task) in enumerate(tasks.items(), start=1):
        profile = profiles.get((dataset, task_id))
        group = group_of(profile) if profile else ""
        chosen, n_learned = solve_task(task, group, by_group, dataset, task_id)
        first, second = chosen
        predictions[task_id] = [
            {"attempt_1": first["predictions"][k].tolist(), "attempt_2": second["predictions"][k].tolist()}
            for k in range(len(task["test_inputs"]))]
        rows.append({"dataset": dataset, "task_id": task_id, "group": group,
                     "learned_methods": n_learned,
                     "attempt_1_method": first["method"], "attempt_1_score": f"{first['score']:.2f}",
                     "attempt_2_method": second["method"], "attempt_2_score": f"{second['score']:.2f}"})
        if number % 200 == 0:
            print(f"  {number}/{len(tasks)} tasks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"grid_solver_predictions_{dataset}.json"
    csv_path = OUT_DIR / f"grid_solver_results_{dataset}.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    learned = sum(1 for r in rows if r["learned_methods"] > 0)
    families = Counter(r["attempt_1_method"].split(":")[0] for r in rows)
    print(f"\n{dataset}: {len(rows)} tasks, a learned method fitted {learned} "
          f"({100 * learned / len(rows):.0f}%)")
    print("First attempt comes from: " + ", ".join(f"{k} {v}" for k, v in families.most_common()))
    print(f"Predictions written to {json_path}")
    print(f"Per-task results written to {csv_path}")
    print("Next: python solver/grid_verifier/grid_verifier.py " + dataset)


if __name__ == "__main__":
    main()