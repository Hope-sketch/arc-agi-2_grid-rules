"""
Parked for later: size HYPOTHESES by stages (not part of the sorter).

This is the previous grid_sizes.py, kept because grid_sizes.py no longer predicts
anything. It is the engine to reuse when the other steps start to help with the
output size of rules 4 and 5.

Goal: predict the test output size with evidence from the TRAINING pairs only.

Rules (TRAINING grids only, the test input is not used)
  1 : every pair has input size == output size
  2 : every pair differs; inputs all one size; outputs all one size
  3 : every pair differs; inputs vary;         outputs all one size
  4 : every pair differs; inputs all one size; outputs vary
  5 : everything else (both vary, or a mix of same/different pairs)

Direction labels (per pair, height and width compared separately)
  absolutely_smaller  both smaller         absolutely_bigger  both bigger
  mixed_1  h bigger,  w smaller            mixed_2  h smaller, w bigger
  mixed_3  h same,    w changed            mixed_4  w same,    h changed
  total_drop  output is 1x1 (takes priority over every other label)
  same        nothing changed
  If every training pair has the same direction, a predicted test size that
  points the other way is FLAGGED in the CSV (direction_flag), not rejected:
  on the public data rejecting it removed correct predictions and no wrong ones.

How it works
  Stage 1  sizes only            features: input height / width / area
  Stage 2  + colours             features: colour counts, background count, ...
  Stage 3  + shapes              features: shape counts, bounding boxes, kinds, ...
  Stage 4  + rows & columns      features: non-empty / distinct rows and columns

  Stages are cumulative: stage 3 may mix sizes, colour and shape features.
  The runner stops at the FIRST stage where at least one hypothesis holds on
  EVERY training pair (height and width are fitted separately).
  A hypothesis that needs features from a later stage is never tried earlier,
  so nothing is hard-coded to a task: the combination is what survives.

Rules 1-3 are decided first:
  1   -> test input size
  2/3 -> the one output size seen in training
Everything else (rules 4 and 5) goes through the stages.

Outputs
  - table in the terminal (per rule, per stage, per dataset)
  - results/grid_size_hypotheses_results.csv  (one row per task)

Run from the project root (the folder that holds data/):
  python solver/grid_size_hypotheses.py
"""

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "grid_size_hypotheses_results.csv"

NEIGHBOURS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBOURS_8 = NEIGHBOURS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

STAGE_NAMES = {1: "sizes", 2: "colours", 3: "shapes", 4: "rows_cols"}


# ----------------------------------------------------------------------
# Load / basic measures
# ----------------------------------------------------------------------
def load_tasks(folder):
    tasks = {}
    for file_path in sorted(folder.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as f:
            tasks[file_path.stem] = json.load(f)
    return tasks


def grid_size(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


def background_colour(grid):
    counts = Counter(c for row in grid for c in row)
    return counts.most_common(1)[0][0]


def regions_of(grid, connectivity, bg):
    """Connected same-colour regions that are not the background."""
    offsets = NEIGHBOURS_8 if connectivity == 8 else NEIGHBOURS_4
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == bg:
                continue
            colour = grid[r][c]
            stack, cells = [(r, c)], []
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr][nc]
                            and grid[nr][nc] == colour):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bh, bw = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            mr, mc = min(rs), min(cs)
            form = tuple(sorted((a - mr, b - mc) for a, b in cells))
            out.append({"colour": colour, "size": len(cells),
                        "bh": bh, "bw": bw, "form": form})
    return out


# ----------------------------------------------------------------------
# Features: name -> (stage, value)
# ----------------------------------------------------------------------
def features_of(grid):
    h, w = grid_size(grid)
    feats = {}

    # Stage 1: sizes
    feats["in_h"] = (1, h)
    feats["in_w"] = (1, w)
    feats["in_area"] = (1, h * w)

    # Stage 2: colours
    bg = background_colour(grid)
    counts = Counter(c for row in grid for c in row)
    non_bg = {c: n for c, n in counts.items() if c != bg}
    feats["n_colours"] = (2, len(counts))
    feats["n_nonbg_colours"] = (2, len(non_bg))
    feats["bg_cells"] = (2, counts[bg])
    feats["nonbg_cells"] = (2, sum(non_bg.values()))
    feats["max_colour_cells"] = (2, max(non_bg.values()) if non_bg else 0)
    feats["min_colour_cells"] = (2, min(non_bg.values()) if non_bg else 0)

    # Stage 3: shapes (8-connected, same colour, background not a shape)
    shapes8 = regions_of(grid, 8, bg)
    shapes4 = regions_of(grid, 4, bg)
    feats["n_shapes8"] = (3, len(shapes8))
    feats["n_shapes4"] = (3, len(shapes4))
    feats["n_forms"] = (3, len({s["form"] for s in shapes8}))
    feats["n_solid"] = (3, sum(1 for s in shapes8 if s["size"] == s["bh"] * s["bw"]))
    feats["n_nonsolid"] = (3, sum(1 for s in shapes8 if s["size"] != s["bh"] * s["bw"]))
    if shapes8:
        big = max(shapes8, key=lambda s: s["size"])
        small = min(shapes8, key=lambda s: s["size"])
        feats["big_shape_h"] = (3, big["bh"])
        feats["big_shape_w"] = (3, big["bw"])
        feats["big_shape_cells"] = (3, big["size"])
        feats["small_shape_h"] = (3, small["bh"])
        feats["small_shape_w"] = (3, small["bw"])
        feats["small_shape_cells"] = (3, small["size"])
        rs = []
        cs = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] != bg:
                    rs.append(r)
                    cs.append(c)
        feats["content_h"] = (3, max(rs) - min(rs) + 1)
        feats["content_w"] = (3, max(cs) - min(cs) + 1)
        common = Counter(s["form"] for s in shapes8).most_common(1)[0][0]
        feats["n_most_common_form"] = (3, sum(1 for s in shapes8 if s["form"] == common))
    else:
        for k in ("big_shape_h", "big_shape_w", "big_shape_cells", "small_shape_h",
                  "small_shape_w", "small_shape_cells", "content_h", "content_w",
                  "n_most_common_form"):
            feats[k] = (3, 0)

    # Stage 4: rows and columns
    rows = [tuple(r) for r in grid]
    cols = [tuple(grid[r][c] for r in range(h)) for c in range(w)]
    feats["n_nonempty_rows"] = (4, sum(1 for r in rows if any(x != bg for x in r)))
    feats["n_nonempty_cols"] = (4, sum(1 for c in cols if any(x != bg for x in c)))
    feats["n_distinct_rows"] = (4, len(set(rows)))
    feats["n_distinct_cols"] = (4, len(set(cols)))
    return feats


# ----------------------------------------------------------------------
# Hypotheses: expression over features -> one dimension
# ----------------------------------------------------------------------
def expressions(names_by_stage, level):
    """Yield (complexity, label, steps_used, fn) for every expression allowed at `level`.
    Expressions must use at least one feature of the newest stage (so earlier
    stages are not tried twice)."""
    allowed = [n for n, s in names_by_stage.items() if s <= level]

    def newest(*names):
        return any(names_by_stage[n] == level for n in names)

    # unary: k*f + c, f // k, ceil(f / 2), square root
    for f in allowed:
        if not newest(f):
            continue
        st = {names_by_stage[f]}
        for k in (1, 2, 3, 4):
            for c in (0, -1, 1, -2, 2):
                comp = (0 if (k == 1 and c == 0) else 1) + (k > 1) + (c != 0)
                label = f"{f}" if (k == 1 and c == 0) else f"{k}*{f}{c:+d}" if c else f"{k}*{f}"
                yield (comp, label, st,
                       lambda F, f=f, k=k, c=c: k * F[f] + c)
        for k in (2, 3):
            yield (3, f"{f}//{k}", st, lambda F, f=f, k=k: F[f] // k)
        yield (3, f"ceil({f}/2)", st, lambda F, f=f: (F[f] + 1) // 2)
        yield (3, f"sqrt({f})", st,
               lambda F, f=f: math.isqrt(F[f]) if math.isqrt(F[f]) ** 2 == F[f] else None)

    # binary: f1 * f2, f1 + f2, f1 - f2
    for f1, f2 in product(allowed, allowed):
        if f1 == f2 or not newest(f1, f2):
            continue
        st = {names_by_stage[f1], names_by_stage[f2]}
        yield (4, f"{f1}*{f2}", st, lambda F, a=f1, b=f2: F[a] * F[b])
        if f1 < f2:
            yield (4, f"{f1}+{f2}", st, lambda F, a=f1, b=f2: F[a] + F[b])
        yield (4, f"{f1}-{f2}", st, lambda F, a=f1, b=f2: F[a] - F[b])


# ----------------------------------------------------------------------
# Direction labels
# ----------------------------------------------------------------------
def sign(a, b):
    return (b > a) - (b < a)


def direction_label(size_in, size_out):
    if size_out == (1, 1) and size_in != (1, 1):
        return "total_drop"
    dh, dw = sign(size_in[0], size_out[0]), sign(size_in[1], size_out[1])
    if dh == 0 and dw == 0:
        return "same"
    if dh < 0 and dw < 0:
        return "absolutely_smaller"
    if dh > 0 and dw > 0:
        return "absolutely_bigger"
    if dh > 0 and dw < 0:
        return "mixed_1"
    if dh < 0 and dw > 0:
        return "mixed_2"
    if dh == 0:
        return "mixed_3"
    return "mixed_4"


# ----------------------------------------------------------------------
# Task analysis
# ----------------------------------------------------------------------
def classify(train):
    """Rule using TRAINING grids only (the test input is not used)."""
    in_sizes = [grid_size(p["input"]) for p in train]
    out_sizes = [grid_size(p["output"]) for p in train]
    same = [i == o for i, o in zip(in_sizes, out_sizes)]
    if all(same):
        return 1, None
    if any(same):
        return 5, None
    ins_const = len(set(in_sizes)) == 1
    outs_const = len(set(out_sizes)) == 1
    if outs_const:
        return (2 if ins_const else 3), out_sizes[0]
    return (4 if ins_const else 5), None


def analyse_task(task):
    train, test = task["train"], task["test"]
    rule, const = classify(train)
    actual = [grid_size(t["output"]) if "output" in t else None for t in test]

    directions = [direction_label(grid_size(p["input"]), grid_size(p["output"])) for p in train]
    row = {"rule": rule, "stage": "-", "steps": "-", "h_hyp": "-", "w_hyp": "-",
           "survivors": 0, "predicted": [], "actual": actual, "outcome": "",
           "directions": "|".join(directions), "direction_flag": ""}

    if rule == 1:
        row["stage"] = "rule"
        row["predicted"] = [grid_size(t["input"]) for t in test]
    elif rule in (2, 3):
        row["stage"] = "rule"
        row["predicted"] = [const for _ in test]
    else:
        train_feats = [features_of(p["input"]) for p in train]
        names = {n: s for n, (s, _) in train_feats[0].items()}
        feats_values = [{n: v for n, (s, v) in tf.items()} for tf in train_feats]
        out_h = [grid_size(p["output"])[0] for p in train]
        out_w = [grid_size(p["output"])[1] for p in train]
        test_vals = [{n: v for n, (s, v) in features_of(t["input"]).items()} for t in test]

        for level in (1, 2, 3, 4):
            def fits(targets):
                hits = []
                for comp, label, steps, fn in expressions(names, level):
                    ok = True
                    for F, target in zip(feats_values, targets):
                        if fn(F) != target:
                            ok = False
                            break
                    if ok:
                        hits.append((comp, label, steps, fn))
                return sorted(hits, key=lambda x: (x[0], x[1]))

            h_hits, w_hits = fits(out_h), fits(out_w)
            if h_hits and w_hits:
                # Every surviving pair of (h, w) hypotheses must agree for a test input to be predicted.
                preds = []
                for tv in test_vals:
                    hs = {fn(tv) for _, _, _, fn in h_hits} - {None}
                    ws = {fn(tv) for _, _, _, fn in w_hits} - {None}
                    if len(hs) == 1 and len(ws) == 1:
                        preds.append((hs.pop(), ws.pop()))
                    else:
                        preds.append(None)
                row["stage"] = STAGE_NAMES[level]
                row["survivors"] = len(h_hits) * len(w_hits)
                row["h_hyp"] = h_hits[0][1]
                row["w_hyp"] = w_hits[0][1]
                row["steps"] = "+".join(sorted(
                    {STAGE_NAMES[s] for _, _, st, _ in (h_hits[0], w_hits[0]) for s in st}))
                # Direction check (flag only): training always moved the same way,
                # but the test prediction moves another way.
                if len(set(directions)) == 1 and directions[0] != "same":
                    for i, (p, t) in enumerate(zip(preds, test)):
                        if p is not None and direction_label(grid_size(t["input"]), p) != directions[0]:
                            row["direction_flag"] = "yes"
                row["predicted"] = preds
                break

    pred, act = row["predicted"], row["actual"]
    if not pred or any(p is None for p in pred):
        row["outcome"] = "no_prediction"
    elif any(a is None for a in act):
        row["outcome"] = "unchecked"
    else:
        row["outcome"] = "correct" if pred == act else "wrong"
    return row


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def print_table(name, results):
    print(f"\n=== {name}: size prediction by rule ===")
    header = f"{'Rule':<6}{'Tasks':>7}{'Correct':>9}{'Wrong':>7}{'No pred.':>10}"
    print(header)
    print("-" * len(header))
    tot = Counter()
    for rule in range(1, 6):
        rows = [r for r in results.values() if r["rule"] == rule]
        c = sum(r["outcome"] == "correct" for r in rows)
        w = sum(r["outcome"] == "wrong" for r in rows)
        n = len(rows) - c - w
        print(f"{rule:<6}{len(rows):>7}{c:>9}{w:>7}{n:>10}")
        tot["t"] += len(rows); tot["c"] += c; tot["w"] += w; tot["n"] += n
    print("-" * len(header))
    print(f"{'All':<6}{tot['t']:>7}{tot['c']:>9}{tot['w']:>7}{tot['n']:>10}")
    print(f"Correct overall: {tot['c']}/{tot['t']} ({tot['c'] / tot['t']:.1%}); "
          f"accuracy when predicted: {tot['c'] / max(1, tot['c'] + tot['w']):.1%}")

    print("\nRules 4 and 5 by the stage that found a rule:")
    by_stage = defaultdict(lambda: Counter())
    for r in results.values():
        if r["rule"] in (4, 5):
            by_stage[r["stage"]][r["outcome"]] += 1
    for stage in ["sizes", "colours", "shapes", "rows_cols", "-"]:
        if stage in by_stage:
            c = by_stage[stage]
            print(f"  {stage:<10} correct {c['correct']:>3}  wrong {c['wrong']:>3}  "
                  f"no_prediction {c['no_prediction']:>3}")


def print_directions(name, results):
    print(f"\n--- {name}: direction of change, rules 3, 4 and 5 (per training pair) ---")
    counts = Counter()
    for r in results.values():
        if r["rule"] in (3, 4, 5):
            for d in r["directions"].split("|"):
                counts[d] += 1
    for label, n in counts.most_common():
        print(f"  {label:<20}{n:>6}")
    vetoed = sum(1 for r in results.values() if r["direction_flag"])
    print(f"  tasks where the prediction points a different way than training: {vetoed}")


def print_wrong(name, results):
    wrong = [(t, r) for t, r in results.items() if r["outcome"] == "wrong"]
    print(f"\nWrong predictions in {name} ({len(wrong)}):")
    for tid, r in wrong:
        print(f"  {tid}  rule {r['rule']}  stage {r['stage']}  predicted {r['predicted']}  actual {r['actual']}")


def write_csv(all_results):
    RESULTS_DIR.mkdir(exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "task_id", "rule", "stage", "steps_used", "h_hypothesis",
                     "w_hypothesis", "survivors", "directions", "direction_flag", "predicted",
                     "actual", "outcome"])
        for ds, res in all_results.items():
            for tid, r in res.items():
                wr.writerow([ds, tid, r["rule"], r["stage"], r["steps"], r["h_hyp"],
                             r["w_hyp"], r["survivors"], r["directions"], r["direction_flag"],
                             r["predicted"], r["actual"], r["outcome"]])
    print(f"\nPer-task results written to {RESULTS_CSV}")


def main():
    all_results = {}
    for name in ("training", "evaluation"):
        tasks = load_tasks(DATA_DIR / name)
        all_results[name] = {tid: analyse_task(t) for tid, t in tasks.items()}
        print_table(name, all_results[name])
        print_directions(name, all_results[name])
        print_wrong(name, all_results[name])
    write_csv(all_results)


if __name__ == "__main__":
    main()
