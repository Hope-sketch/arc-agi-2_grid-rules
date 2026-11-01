"""
Step 2: Grid colours (investigation only)

For every task this finds, from the TRAINING pairs only:
  * which colours are used and how many (input and output, per pair)
  * which colours are added / removed from input to output
  * what the background colour is, and whether it stays or changes
  * what colour the shapes are, and whether that stays or changes

and turns it into a RECEIPT with one flag per question:

  Colours     COLOURS_ADDED              new colours appear in every pair, none removed
              COLOURS_REMOVED            colours disappear in every pair, none added
              COLOURS_ADDED_AND_REMOVED  every pair both adds and removes colours
              COLOURS_SAME               the palette does not change
              COLOURS_MIXED              the pairs disagree
  Background  BACKGROUND_SAME / BACKGROUND_CHANGES / NO_BACKGROUND / BACKGROUND_MIXED
  Shapes      SHAPE_COLOUR_STAYS / SHAPE_COLOUR_CHANGES / NO_SHAPE / SHAPE_COLOUR_MIXED

Definitions (change the constants below to tune them)
  background : the single most frequent colour (no tie), if it touches the border
               and covers at least BACKGROUND_MIN_SHARE of the grid; otherwise the
               grid has no background.
  shape      : cells of one colour joined through their sides (4-connectivity), the
               background is never a shape. With no background every cell is in a shape.
  shape colour changes : the set of shape colours in the input differs from the output.

Shapes: the shape finder is the single function find_shapes(). It uses the settled
shape definition itself for now; when grid shapes writes its own receipts JSON, this
is the one place to swap over.

The receipt never contains the test output. The public test output is only used for
the "holds on test" columns in the CSV (a check, not a score).

Usage
  python solver/grid_colours/grid_colours.py

Outputs
  * tables in the terminal (training, then evaluation)
  * solver/grid_colours/results/grid_colours_results.csv    (one row per task)
  * solver/grid_colours/results/grid_colours_receipts.json (what later steps load)

From another step
  import sys; sys.path.insert(0, str(ROOT / "solver" / "grid_colours"))
  from grid_colours import load_colours_receipts, check_colours_receipts
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repo root (holds data/)
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
OUT_CSV = RESULTS_DIR / "grid_colours_results.csv"
OUT_JSON = RESULTS_DIR / "grid_colours_receipts.json"
DATASETS = ("training", "evaluation")

BACKGROUND_MIN_SHARE = 0.4    # the background must cover at least this share of the grid

PALETTE_FLAG = {"added": "COLOURS_ADDED", "removed": "COLOURS_REMOVED",
                "added_and_removed": "COLOURS_ADDED_AND_REMOVED",
                "same": "COLOURS_SAME", "mixed": "COLOURS_MIXED"}
BACKGROUND_FLAG = {"same": "BACKGROUND_SAME", "changes": "BACKGROUND_CHANGES",
                   "none": "NO_BACKGROUND", "mixed": "BACKGROUND_MIXED"}
SHAPE_FLAG = {"stays": "SHAPE_COLOUR_STAYS", "changes": "SHAPE_COLOUR_CHANGES",
              "no_shape": "NO_SHAPE", "mixed": "SHAPE_COLOUR_MIXED"}


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
# One grid
# ----------------------------------------------------------------------
def find_background(grid):
    """The background colour, or None if the grid has no clear background."""
    counts = Counter(c for row in grid for c in row)
    ranked = counts.most_common()
    colour, count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == count:
        return None                                        # tie
    if count / sum(counts.values()) < BACKGROUND_MIN_SHARE:
        return None                                        # does not dominate
    border = set(grid[0]) | set(grid[-1]) | {r[0] for r in grid} | {r[-1] for r in grid}
    return colour if colour in border else None


def find_shapes(grid, background):
    """{colour: number of shapes} (4-connectivity, same colour, background excluded)."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    shapes = Counter()
    for r in range(h):
        for c in range(w):
            colour = grid[r][c]
            if seen[r][c] or colour == background:
                continue
            shapes[colour] += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == colour:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
    return dict(shapes)


def describe_grid(grid):
    background = find_background(grid)
    return {"colours": sorted({c for row in grid for c in row}),
            "background": background,
            "shapes": find_shapes(grid, background)}


# ----------------------------------------------------------------------
# One pair
# ----------------------------------------------------------------------
def palette_label(added, removed):
    if added and removed:
        return "added_and_removed"
    return "added" if added else "removed" if removed else "same"


def background_label(bg_in, bg_out):
    if bg_in is None and bg_out is None:
        return "none"
    if bg_in is None or bg_out is None:
        return "one_side_none"
    return "same" if bg_in == bg_out else "changes"


def shape_label(shapes_in, shapes_out):
    if not shapes_in and not shapes_out:
        return "no_shape"
    return "stays" if set(shapes_in) == set(shapes_out) else "changes"


def describe_pair(grid_in, grid_out):
    a, b = describe_grid(grid_in), describe_grid(grid_out)
    added = sorted(set(b["colours"]) - set(a["colours"]))
    removed = sorted(set(a["colours"]) - set(b["colours"]))
    return {"input_colours": a["colours"], "output_colours": b["colours"],
            "added": added, "removed": removed,
            "palette_label": palette_label(added, removed),
            "input_background": a["background"], "output_background": b["background"],
            "background_label": background_label(a["background"], b["background"]),
            "input_shapes": {str(k): v for k, v in a["shapes"].items()},
            "output_shapes": {str(k): v for k, v in b["shapes"].items()},
            "shape_colour_label": shape_label(a["shapes"], b["shapes"])}


def task_word(labels, allowed):
    """One word for the whole task: the shared label, or 'mixed' if the pairs disagree."""
    first = labels[0]
    return first if all(l == first for l in labels) and first in allowed else "mixed"


# ----------------------------------------------------------------------
# Receipt (training pairs only)
# ----------------------------------------------------------------------
def get_colours_receipt(task):
    pairs = [describe_pair(p["input"], p["output"]) for p in task["train"]]

    palette_word = task_word([p["palette_label"] for p in pairs],
                             {"added", "removed", "added_and_removed", "same"})
    background_word = task_word([p["background_label"] for p in pairs],
                                {"same", "changes", "none"})
    shape_word = task_word([p["shape_colour_label"] for p in pairs],
                           {"stays", "changes", "no_shape"})

    backgrounds = {p["input_background"] for p in pairs} | {p["output_background"] for p in pairs}
    background_colour = backgrounds.pop() if len(backgrounds) == 1 else None

    always_added = set.intersection(*[set(p["added"]) for p in pairs])
    always_removed = set.intersection(*[set(p["removed"]) for p in pairs])

    seen_in_training = {c for p in pairs for c in p["input_colours"]}
    tests = []
    for t in task["test"]:
        d = describe_grid(t["input"])
        tests.append({"colours": d["colours"], "background": d["background"],
                      "shapes": {str(k): v for k, v in d["shapes"].items()}})

    return {
        "flags": [PALETTE_FLAG[palette_word], BACKGROUND_FLAG[background_word], SHAPE_FLAG[shape_word]],
        "palette_word": palette_word, "background_word": background_word,
        "shape_colour_word": shape_word,
        "background_colour": background_colour,
        "colours_always_added": sorted(always_added),
        "colours_always_removed": sorted(always_removed),
        "pairs": pairs,
        "test_inputs": tests,
        "test_colours_seen_in_training": all(set(t["colours"]) <= seen_in_training for t in tests),
    }


def check_on_public_test(receipt, task):
    """Does each training word also describe the public test pair? Verification only."""
    if not all("output" in t for t in task["test"]):
        return {"palette": "unknown", "background": "unknown", "shape_colour": "unknown"}
    test_pairs = [describe_pair(t["input"], t["output"]) for t in task["test"]]
    out = {}
    for name, word, key in (("palette", receipt["palette_word"], "palette_label"),
                            ("background", receipt["background_word"], "background_label"),
                            ("shape_colour", receipt["shape_colour_word"], "shape_colour_label")):
        if word == "mixed":
            out[name] = "n/a"
        else:
            out[name] = "yes" if all(p[key] == word for p in test_pairs) else "no"
    return out


# ----------------------------------------------------------------------
# For the other steps
# ----------------------------------------------------------------------
def load_colours_receipts(path=OUT_JSON):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid colours receipts not found: {path}\n"
                         f"Run solver/grid_colours/grid_colours.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_colours_receipts(receipts, dataset, task_ids):
    """Stop if any task has no grid colours receipt, and say which ones."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid colours receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_colours/grid_colours.py.")
    print(f"Grid colours receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def join(values):
    return ",".join(str(v) for v in values)


def per_pair(receipt, key):
    return ";".join(join(p[key]) if isinstance(p[key], list) else str(p[key]) for p in receipt["pairs"])


def csv_row(dataset, task_id, task, receipt, holds):
    return {
        "dataset": dataset, "task_id": task_id, "n_train_pairs": len(task["train"]),
        "flags": "|".join(receipt["flags"]),
        "palette_word": receipt["palette_word"],
        "background_word": receipt["background_word"],
        "shape_colour_word": receipt["shape_colour_word"],
        "n_colours_in": ";".join(str(len(p["input_colours"])) for p in receipt["pairs"]),
        "n_colours_out": ";".join(str(len(p["output_colours"])) for p in receipt["pairs"]),
        "colours_in": per_pair(receipt, "input_colours"),
        "colours_out": per_pair(receipt, "output_colours"),
        "colours_added": per_pair(receipt, "added"),
        "colours_removed": per_pair(receipt, "removed"),
        "background_in": per_pair(receipt, "input_background"),
        "background_out": per_pair(receipt, "output_background"),
        "background_colour": "" if receipt["background_colour"] is None else receipt["background_colour"],
        "shape_colours_in": ";".join(join(p["input_shapes"]) for p in receipt["pairs"]),
        "shape_colours_out": ";".join(join(p["output_shapes"]) for p in receipt["pairs"]),
        "test_colours": ";".join(join(t["colours"]) for t in receipt["test_inputs"]),
        "test_background": ";".join(str(t["background"]) for t in receipt["test_inputs"]),
        "test_colours_seen_in_training": "yes" if receipt["test_colours_seen_in_training"] else "no",
        "holds_palette": holds["palette"], "holds_background": holds["background"],
        "holds_shape_colour": holds["shape_colour"],
    }


def print_summary(dataset, rows):
    total = len(rows)
    print(f"\n=== {dataset}: grid colours ({total} tasks) ===")
    for title, word_key, holds_key in (("Colours", "palette_word", "holds_palette"),
                                       ("Background", "background_word", "holds_background"),
                                       ("Shape colour", "shape_colour_word", "holds_shape_colour")):
        print(f"\n{title}")
        header = f"{'Word':<20}{'Tasks':>7}{'Share':>8}{'Holds on test':>16}"
        print(header)
        print("-" * len(header))
        by = defaultdict(list)
        for r in rows:
            by[r[word_key]].append(r)
        for word, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
            checked = [r for r in group if r[holds_key] in ("yes", "no")]
            holds = (f"{sum(r[holds_key] == 'yes' for r in checked)}/{len(checked)}"
                     if checked else "-")
            print(f"{word:<20}{len(group):>7}{100 * len(group) / total:>7.1f}%{holds:>16}")
    seen = sum(r["test_colours_seen_in_training"] == "yes" for r in rows)
    print(f"\nTest input uses only colours seen in training inputs: {seen}/{total}")


def main():
    all_rows, receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        rows, receipts[dataset] = [], {}
        for task_id, task in load_tasks(folder).items():
            receipt = get_colours_receipt(task)
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