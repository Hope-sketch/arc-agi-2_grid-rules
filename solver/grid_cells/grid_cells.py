"""
Step 4: Grid cells (investigation only)

How alike are the grids? Compares every training input with its output (and the
training outputs with each other) cell by cell. Height x width is the first clue:

  * same size    : cells are compared position by position
  * different    : the smaller grid is slid over the bigger one and the best fit
                   is used (the output can be a piece of the input, or the reverse)

Similarity word per training pair (share of cells that are the same):
  very_similar      60% or more   (grids extremely similar, minor changes)
  somewhat_similar  50% to 59.9%  (not that similar)
  not_similar       under 50%     (not similar)
The share uses the bigger grid's area, so a small piece of a big grid is not
"similar"; the piece itself is reported separately (OUTPUT_IS_PIECE_OF_INPUT).

Receipt flags (true in EVERY training pair; "sometimes" = true in some pairs)
  Similarity   CELLS_VERY_SIMILAR / CELLS_SOMEWHAT_SIMILAR / CELLS_NOT_SIMILAR / CELLS_MIXED
  Size         SAME_SIZE / OUTPUT_BIGGER / OUTPUT_SMALLER / MIXED_SIZE / SIZE_VARIES
  Events       GRID_IDENTICAL                  output is a copy of the input
               CELLS_ADDED                     background cells become coloured
               CELLS_ERASED                    coloured cells become background
               CELLS_RECOLOURED                coloured cells change colour
               OUTPUT_IS_PIECE_OF_INPUT        output is an exact piece of the input
               INPUT_IS_PIECE_OF_OUTPUT        input is an exact piece of the output
               SIMILAR_AFTER_FLIP_OR_ROTATION  a flip / rotation of the input is 90%+ the output
               SIMILARITY_FROM_BACKGROUND      60%+ alike, but under 50% of the
                                               non-background cells match (the
                                               background is doing the work)
The receipt also holds output-vs-output similarity (when all training outputs share
a size) and never contains the test output. The public test output is only used for
the "holds on test" checks in the CSV and tables.

Background comes from find_background() in solver/grid_colours/grid_colours.py.

Usage
  python solver/grid_cells/grid_cells.py

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_cells/results/grid_cells_results.csv    (one row per task)
  * solver/grid_cells/results/grid_cells_receipts.json  (what later steps load)

From another step
  import sys; sys.path.insert(0, str(ROOT / "solver" / "grid_cells"))
  from grid_cells import load_cells_receipts, check_cells_receipts
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_cells_results.csv"
OUT_JSON = RESULTS_DIR / "grid_cells_receipts.json"
DATASETS = ("training", "evaluation")

sys.path.insert(0, str(ROOT / "solver" / "grid_colours"))
try:
    from grid_colours import find_background
except ImportError:
    raise SystemExit("grid_cells.py needs find_background() from "
                     "solver/grid_colours/grid_colours.py. Put the new grid_colours.py there first.")

VERY_SIMILAR_MIN = 60.0          # percent
SOMEWHAT_SIMILAR_MIN = 50.0      # percent
SYMMETRY_MIN = 90.0              # percent, for SIMILAR_AFTER_FLIP_OR_ROTATION

SIMILARITY_FLAG = {"very_similar": "CELLS_VERY_SIMILAR", "somewhat_similar": "CELLS_SOMEWHAT_SIMILAR",
                   "not_similar": "CELLS_NOT_SIMILAR", "mixed": "CELLS_MIXED"}
SIZE_FLAG = {"same": "SAME_SIZE", "output_bigger": "OUTPUT_BIGGER", "output_smaller": "OUTPUT_SMALLER",
             "mixed_size": "MIXED_SIZE", "varies": "SIZE_VARIES"}
EVENTS = ["GRID_IDENTICAL", "CELLS_ADDED", "CELLS_ERASED", "CELLS_RECOLOURED",
          "OUTPUT_IS_PIECE_OF_INPUT", "INPUT_IS_PIECE_OF_OUTPUT",
          "SIMILAR_AFTER_FLIP_OR_ROTATION", "SIMILARITY_FROM_BACKGROUND"]

SYMMETRIES = {"flip_h": lambda g: g[:, ::-1], "flip_v": lambda g: g[::-1, :],
              "rotate_90": lambda g: np.rot90(g, -1), "rotate_180": lambda g: np.rot90(g, 2),
              "rotate_270": lambda g: np.rot90(g, 1), "transpose": lambda g: g.T,
              "anti_transpose": lambda g: np.rot90(g, 2).T}


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
# Comparing two grids
# ----------------------------------------------------------------------
def similarity_label(percent):
    if percent >= VERY_SIMILAR_MIN:
        return "very_similar"
    return "somewhat_similar" if percent >= SOMEWHAT_SIMILAR_MIN else "not_similar"


def size_relation(a, b):
    (ha, wa), (hb, wb) = a.shape, b.shape
    if (ha, wa) == (hb, wb):
        return "same"
    if ha <= hb and wa <= wb:
        return "output_bigger"
    if ha >= hb and wa >= wb:
        return "output_smaller"
    return "mixed_size"


def non_background(window, background):
    return np.ones(window.shape, dtype=bool) if background is None else window != background


def align(a, b, bg_a, bg_b):
    """Slide the smaller grid over the bigger one (per side). Best fit wins."""
    (ha, wa), (hb, wb) = a.shape, b.shape
    ch, cw = min(ha, hb), min(wa, wb)
    rows = ([(i, 0) for i in range(ha - hb + 1)] if ha > hb else
            [(0, i) for i in range(hb - ha + 1)] if hb > ha else [(0, 0)])
    cols = ([(i, 0) for i in range(wa - wb + 1)] if wa > wb else
            [(0, i) for i in range(wb - wa + 1)] if wb > wa else [(0, 0)])
    best, exact = None, False
    for ra, rb in rows:
        for ca, cb in cols:
            win_a, win_b = a[ra:ra + ch, ca:ca + cw], b[rb:rb + ch, cb:cb + cw]
            equal = win_a == win_b
            union = non_background(win_a, bg_a) | non_background(win_b, bg_b)
            raw, nb_match, nb_total = int(equal.sum()), int((equal & union).sum()), int(union.sum())
            exact = exact or raw == ch * cw
            if best is None or (nb_match, raw) > (best[0], best[1]):
                best = (nb_match, raw, nb_total, ra, ca, rb, cb)
    nb_match, raw, nb_total, ra, ca, rb, cb = best
    return {"raw": raw, "window": ch * cw, "nb_match": nb_match, "nb_total": nb_total,
            "exact_piece": exact, "offset_in": [ra, ca], "offset_out": [rb, cb]}


def compare_pair(grid_in, grid_out):
    a, b = to_np(grid_in), to_np(grid_out)
    bg_a, bg_b = find_background(grid_in), find_background(grid_out)
    relation = size_relation(a, b)
    fit = align(a, b, bg_a, bg_b)
    bigger = max(a.size, b.size)

    overall = 100 * fit["raw"] / bigger
    aligned = 100 * fit["raw"] / fit["window"]
    non_bg = 100 * fit["nb_match"] / fit["nb_total"] if fit["nb_total"] else 100.0
    label = similarity_label(overall)

    ev = set()
    changed = added = erased = recoloured = 0
    if relation == "same":
        changed = int((a != b).sum())
        if changed == 0:
            ev.add("GRID_IDENTICAL")
        if bg_a is not None and bg_b is not None:
            added = int(((a == bg_a) & (b != bg_b)).sum())
            erased = int(((a != bg_a) & (b == bg_b)).sum())
            recoloured = int(((a != bg_a) & (b != bg_b) & (a != b)).sum())
            if added: ev.add("CELLS_ADDED")
            if erased: ev.add("CELLS_ERASED")
            if recoloured: ev.add("CELLS_RECOLOURED")
    elif fit["exact_piece"]:
        ev.add("OUTPUT_IS_PIECE_OF_INPUT" if relation == "output_smaller" else "INPUT_IS_PIECE_OF_OUTPUT")

    best_name, best_share = "", 0.0
    for name, transform in SYMMETRIES.items():
        t = transform(a)
        if t.shape == b.shape:
            share = 100 * float((t == b).mean())
            if share > best_share:
                best_name, best_share = name, share
    if best_share >= SYMMETRY_MIN and best_share > overall + 10:
        ev.add("SIMILAR_AFTER_FLIP_OR_ROTATION")
    if overall >= VERY_SIMILAR_MIN and non_bg < SOMEWHAT_SIMILAR_MIN:
        ev.add("SIMILARITY_FROM_BACKGROUND")

    return {"size_relation": relation,
            "input_size": list(a.shape), "output_size": list(b.shape),
            "area_ratio": round(b.size / a.size, 3),
            "overall_share": round(overall, 1), "aligned_share": round(aligned, 1),
            "non_background_share": round(non_bg, 1), "similarity_label": label,
            "changed_cells": changed, "cells_added": added, "cells_erased": erased,
            "cells_recoloured": recoloured,
            "best_symmetry": best_name, "best_symmetry_share": round(best_share, 1),
            "events": [e for e in EVENTS if e in ev]}


# ----------------------------------------------------------------------
# Receipt (training pairs only)
# ----------------------------------------------------------------------
def task_word(labels, allowed, fallback):
    first = labels[0]
    return first if all(l == first for l in labels) and first in allowed else fallback


def output_vs_output(outputs):
    """Mean cell match between training outputs, when they all share one size."""
    arrays = [to_np(o) for o in outputs]
    if len(arrays) < 2:
        return {"word": "n/a", "share": None}
    if len({a.shape for a in arrays}) != 1:
        return {"word": "n/a", "share": None}
    shares = [100 * float((x == y).mean()) for x, y in combinations(arrays, 2)]
    mean = sum(shares) / len(shares)
    return {"word": similarity_label(mean), "share": round(mean, 1)}


def get_cells_receipt(task):
    pairs = [compare_pair(p["input"], p["output"]) for p in task["train"]]
    shares = [p["overall_share"] for p in pairs]

    similarity_word = task_word([p["similarity_label"] for p in pairs],
                                {"very_similar", "somewhat_similar", "not_similar"}, "mixed")
    size_word = task_word([p["size_relation"] for p in pairs],
                          {"same", "output_bigger", "output_smaller", "mixed_size"}, "varies")
    flags = [SIMILARITY_FLAG[similarity_word], SIZE_FLAG[size_word]]
    flags += [e for e in EVENTS if all(e in p["events"] for p in pairs)]
    sometimes = [e for e in EVENTS if any(e in p["events"] for p in pairs) and e not in flags]

    return {"flags": flags, "sometimes": sometimes,
            "similarity_word": similarity_word, "size_word": size_word,
            "mean_share": round(sum(shares) / len(shares), 1),
            "min_share": min(shares), "max_share": max(shares),
            "output_vs_output": output_vs_output([p["output"] for p in task["train"]]),
            "pairs": pairs,
            "test_input_sizes": [[len(t["input"]), len(t["input"][0])] for t in task["test"]]}


def check_on_public_test(receipt, task):
    """Do the training words also describe the public test pair? Verification only."""
    if not all("output" in t for t in task["test"]):
        return None
    test_pairs = [compare_pair(t["input"], t["output"]) for t in task["test"]]
    word = receipt["similarity_word"]
    similarity = ("n/a" if word == "mixed"
                  else "yes" if all(p["similarity_label"] == word for p in test_pairs) else "no")
    size_word = receipt["size_word"]
    size = ("n/a" if size_word == "varies"
            else "yes" if all(p["size_relation"] == size_word for p in test_pairs) else "no")
    events = [set(p["events"]) for p in test_pairs]
    event_flags = [f for f in receipt["flags"] if f in EVENTS]
    holding = [f for f in event_flags if all(f in e for e in events)]
    return {"similarity": similarity, "size": size, "holding": holding,
            "failing": [f for f in event_flags if f not in holding]}


# ----------------------------------------------------------------------
# For the other steps
# ----------------------------------------------------------------------
def load_cells_receipts(path=OUT_JSON):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid cells receipts not found: {path}\n"
                         f"Run solver/grid_cells/grid_cells.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_cells_receipts(receipts, dataset, task_ids):
    """Stop if any task has no grid cells receipt, and say which ones."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid cells receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_cells/grid_cells.py.")
    print(f"Grid cells receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def csv_row(dataset, task_id, task, receipt, check):
    pairs = receipt["pairs"]
    each = lambda key: ";".join(str(p[key]) for p in pairs)
    oo = receipt["output_vs_output"]
    return {
        "dataset": dataset, "task_id": task_id, "n_train_pairs": len(task["train"]),
        # kept so grid_transformations.py (which reads a "cell_group" column) still finds one
        "cell_group": f"{receipt['size_word']}/{receipt['similarity_word']}",
        "similarity_word": receipt["similarity_word"], "size_word": receipt["size_word"],
        "flags": "|".join(receipt["flags"]), "sometimes": "|".join(receipt["sometimes"]),
        "mean_share": receipt["mean_share"], "min_share": receipt["min_share"],
        "max_share": receipt["max_share"],
        "size_relation": each("size_relation"), "area_ratio": each("area_ratio"),
        "overall_share": each("overall_share"), "aligned_share": each("aligned_share"),
        "non_background_share": each("non_background_share"),
        "changed_cells": each("changed_cells"), "cells_added": each("cells_added"),
        "cells_erased": each("cells_erased"), "cells_recoloured": each("cells_recoloured"),
        "best_symmetry": each("best_symmetry"),
        "output_vs_output_word": oo["word"], "output_vs_output_share": "" if oo["share"] is None else oo["share"],
        "test_input_sizes": ";".join(f"{h}x{w}" for h, w in receipt["test_input_sizes"]),
        "holds_similarity": "unknown" if check is None else check["similarity"],
        "holds_size": "unknown" if check is None else check["size"],
        "events_holding_on_test": "unknown" if check is None else "|".join(check["holding"]),
        "events_failing_on_test": "unknown" if check is None else "|".join(check["failing"]),
    }


def print_summary(dataset, rows):
    total = len(rows)
    print(f"\n=== {dataset}: grid cells ({total} tasks) ===")

    print("\nSimilarity of input and output")
    header = f"{'Word':<20}{'Tasks':>7}{'Share':>8}{'Holds on test':>16}"
    print(header)
    print("-" * len(header))
    for word in ("very_similar", "somewhat_similar", "not_similar", "mixed"):
        group = [r for r in rows if r["similarity_word"] == word]
        checked = [r for r in group if r["holds_similarity"] in ("yes", "no")]
        holds = f"{sum(r['holds_similarity'] == 'yes' for r in checked)}/{len(checked)}" if checked else "-"
        print(f"{word:<20}{len(group):>7}{100 * len(group) / total:>7.1f}%{holds:>16}")

    print("\nSize relation x similarity (tasks)")
    words = ("very_similar", "somewhat_similar", "not_similar", "mixed")
    header = f"{'':<16}" + "".join(f"{w:>18}" for w in words) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    for size in ("same", "output_bigger", "output_smaller", "mixed_size", "varies"):
        group = [r for r in rows if r["size_word"] == size]
        if not group:
            continue
        print(f"{size:<16}" + "".join(f"{sum(r['similarity_word'] == w for r in group):>18}" for w in words)
              + f"{len(group):>8}")

    print("\nOutput vs output (outputs all the same size)")
    oo = Counter(r["output_vs_output_word"] for r in rows)
    print("  " + ", ".join(f"{w}: {oo[w]}" for w in ("very_similar", "somewhat_similar", "not_similar", "n/a")))

    print("\nEvents")
    header = f"{'Event':<34}{'In every pair':>14}{'Holds on test':>16}{'Sometimes':>11}"
    print(header)
    print("-" * len(header))
    for event in EVENTS:
        flagged = [r for r in rows if event in r["flags"].split("|")]
        sometimes = sum(event in r["sometimes"].split("|") for r in rows)
        checked = [r for r in flagged if r["events_holding_on_test"] != "unknown"]
        holds = sum(event in r["events_holding_on_test"].split("|") for r in checked)
        print(f"{event:<34}{len(flagged):>14}{(f'{holds}/{len(checked)}' if checked else '-'):>16}{sometimes:>11}")


def main():
    all_rows, receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        rows, receipts[dataset] = [], {}
        for task_id, task in load_tasks(folder).items():
            receipt = get_cells_receipt(task)
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