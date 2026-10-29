"""
Step 6: Grid Transformations (investigation only)

Tests a library of whole-grid transformations against every training pair of every task:

  geometric   : identity, flip, rotate, transpose
  scale       : upscale / downscale by the size ratio seen in training
  tile        : repeat the grid n x m times (plain or mirrored)
  mirror join : grid + its mirror (side by side / stacked / 2x2)
  crop        : bounding box of everything that is not background; halves
  colour map  : any of the above followed by ONE consistent colour mapping
                learned from the training pairs (no colour is hard-coded)

A transformation counts as FOUND when it reproduces the output of EVERY training pair.
As a sanity check against fooling ourselves, a found transformation is then applied to the
test input and compared with the real test output (these are public, so we can check).

Writes:
  results/grid_transformations_results.csv   one row per task

Usage:
  python grid_transformations.py <folder of task .json files>
"""

import csv
import json
import os
import sys
from collections import Counter

import numpy as np

DATA_PATH = "data/training"
RESULTS_DIR = "results"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "grid_transformations_results.csv")


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------
def load_tasks(path):
    tasks = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".json"):
            with open(os.path.join(path, name)) as f:
                data = json.load(f)
            if isinstance(data, dict) and "train" in data:
                tasks[os.path.splitext(name)[0]] = data
    return tasks


def to_np(g):
    return np.array(g, dtype=int)


def background(g):
    vals, counts = np.unique(g, return_counts=True)
    return vals[np.argmax(counts)]


# ----------------------------------------------------------------------------
# TRANSFORMATION LIBRARY
# ----------------------------------------------------------------------------
def crop_non_background(g):
    bg = background(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return None
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def fixed_transforms():
    """Parameter-free transformations: name -> function."""
    t = {
        "identity":        lambda g: g,
        "flip_horizontal": lambda g: g[:, ::-1],
        "flip_vertical":   lambda g: g[::-1, :],
        "rotate_90":       lambda g: np.rot90(g, -1),
        "rotate_180":      lambda g: np.rot90(g, 2),
        "rotate_270":      lambda g: np.rot90(g, 1),
        "transpose":       lambda g: g.T,
        "anti_transpose":  lambda g: np.rot90(g, 2).T,
        "mirror_right":    lambda g: np.hstack([g, g[:, ::-1]]),
        "mirror_left":     lambda g: np.hstack([g[:, ::-1], g]),
        "mirror_down":     lambda g: np.vstack([g, g[::-1, :]]),
        "mirror_up":       lambda g: np.vstack([g[::-1, :], g]),
        "mirror_2x2":      lambda g: np.vstack([np.hstack([g, g[:, ::-1]]),
                                                np.hstack([g[::-1, :], g[::-1, ::-1]])]),
        "rotate_2x2":      lambda g: np.vstack([np.hstack([g, np.rot90(g, -1)]),
                                                np.hstack([np.rot90(g, 1), np.rot90(g, 2)])])
                           if g.shape[0] == g.shape[1] else None,
        "crop_non_background": crop_non_background,
        "half_left":       lambda g: g[:, :g.shape[1] // 2],
        "half_right":      lambda g: g[:, (g.shape[1] + 1) // 2:],
        "half_top":        lambda g: g[:g.shape[0] // 2, :],
        "half_bottom":     lambda g: g[(g.shape[0] + 1) // 2:, :],
    }
    return t


def parameterised_transforms(first_in, first_out):
    """Scale / tile transformations, parameters taken from the size ratio of training pair 0."""
    t = {}
    hi, wi = first_in.shape
    ho, wo = first_out.shape
    # upscale
    if ho % hi == 0 and wo % wi == 0 and (ho // hi, wo // wi) != (1, 1):
        ky, kx = ho // hi, wo // wi
        t[f"upscale_{ky}x{kx}"] = lambda g, ky=ky, kx=kx: np.kron(g, np.ones((ky, kx), dtype=int))
        t[f"tile_{ky}x{kx}"] = lambda g, ky=ky, kx=kx: np.tile(g, (ky, kx))
        t[f"tile_mirrored_{ky}x{kx}"] = lambda g, ky=ky, kx=kx: mirrored_tile(g, ky, kx)
    # downscale
    if hi % ho == 0 and wi % wo == 0 and (hi // ho, wi // wo) != (1, 1):
        ky, kx = hi // ho, wi // wo
        t[f"downscale_{ky}x{kx}"] = lambda g, ky=ky, kx=kx: g[::ky, ::kx] if g.shape[0] % ky == 0 and g.shape[1] % kx == 0 else None
    return t


def mirrored_tile(g, ky, kx):
    rows = []
    for i in range(ky):
        row = []
        for j in range(kx):
            tile = g
            if i % 2 == 1:
                tile = tile[::-1, :]
            if j % 2 == 1:
                tile = tile[:, ::-1]
            row.append(tile)
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ----------------------------------------------------------------------------
# TESTING A TRANSFORMATION
# ----------------------------------------------------------------------------
def safe_apply(fn, g):
    try:
        out = fn(g)
    except Exception:
        return None
    if out is None or out.size == 0:
        return None
    return np.array(out)


def learn_colour_map(predictions, targets):
    """One colour map (a -> b) consistent over every training pair, or None."""
    mapping = {}
    for p, t in zip(predictions, targets):
        if p is None or p.shape != t.shape:
            return None
        for a, b in zip(p.ravel(), t.ravel()):
            if mapping.setdefault(int(a), int(b)) != int(b):
                return None
    return mapping


def apply_map(g, mapping):
    out = g.copy()
    for a in np.unique(g):
        if int(a) not in mapping:
            return None          # colour never seen in training: cannot map it
        out[g == a] = mapping[int(a)]
    return out


def cell_accuracy(p, t):
    if p is None or p.shape != t.shape:
        return 0.0
    return float((p == t).mean())


def analyse_task(task_id, task):
    train = [(to_np(p["input"]), to_np(p["output"])) for p in task["train"]]
    tests = [(to_np(p["input"]), to_np(p["output"]) if "output" in p else None)
             for p in task["test"]]

    library = fixed_transforms()
    library.update(parameterised_transforms(*train[0]))

    exact, colour_map_found, scores = [], [], []
    maps = {}

    for name, fn in library.items():
        preds = [safe_apply(fn, i) for i, _ in train]
        per_pair_exact = [p is not None and p.shape == o.shape and np.array_equal(p, o)
                          for p, (_, o) in zip(preds, train)]
        mean_acc = float(np.mean([cell_accuracy(p, o) for p, (_, o) in zip(preds, train)]))
        scores.append((sum(per_pair_exact), mean_acc, name))

        if all(per_pair_exact):
            exact.append(name)
        else:
            m = learn_colour_map(preds, [o for _, o in train])
            if m is not None:
                colour_map_found.append(name)
                maps[name] = m

    # ---- choose the answer (simplest first = library order) ----
    chosen, kind = None, "none"
    if exact:
        chosen, kind = exact[0], "exact"
    elif colour_map_found:
        chosen, kind = colour_map_found[0], "colour_map"

    # ---- sanity check on the public test pairs ----
    test_correct = ""
    if chosen is not None:
        results = []
        for ti, to in tests:
            if to is None:
                continue
            p = safe_apply(library[chosen], ti)
            if p is not None and kind == "colour_map":
                p = apply_map(p, maps[chosen])
            results.append(p is not None and p.shape == to.shape and np.array_equal(p, to))
        if results:
            test_correct = all(results)

    best = max(scores, key=lambda s: (s[0], s[1]))
    if kind == "exact":
        category = "found_exact"
    elif kind == "colour_map":
        category = "found_with_colour_map"
    elif best[0] > 0:
        category = "partial"          # works on some training pairs but not all
    else:
        category = "none"

    return {
        "task_id": task_id,
        "n_train_pairs": len(train),
        "category": category,
        "chosen_transformation": chosen or "",
        "all_exact_transformations": "|".join(exact),
        "colour_map_transformations": "|".join(colour_map_found),
        "best_partial": best[2] if category == "partial" else "",
        "best_partial_pairs_matched": best[0] if category == "partial" else "",
        "best_mean_cell_accuracy": round(best[1], 3),
        "test_correct": test_correct,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    tasks = load_tasks(path)
    if not tasks:
        print(f"No tasks found at: {path}")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = [analyse_task(tid, t) for tid, t in tasks.items()]

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"Tasks analysed: {n}\n")
    print("CATEGORY")
    for cat, c in Counter(r["category"] for r in rows).most_common():
        print(f"  {cat:<24} {c:>5}  ({100 * c / n:.1f}%)")

    found = [r for r in rows if r["category"] in ("found_exact", "found_with_colour_map")]
    right = sum(1 for r in found if r["test_correct"] is True)
    wrong = sum(1 for r in found if r["test_correct"] is False)
    print(f"\nOf the {len(found)} tasks where a transformation fit ALL training pairs:")
    print(f"  also correct on the test pair : {right}")
    print(f"  wrong on the test pair        : {wrong}")
    print("\nCHOSEN TRANSFORMATION")
    for name, c in Counter(r["chosen_transformation"] for r in found).most_common():
        print(f"  {name:<24} {c:>5}")
    print(f"\nSaved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
