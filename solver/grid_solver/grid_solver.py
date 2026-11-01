"""
Step 7: Grid solver

Reads the receipt of every task from grid_transformations (which already holds the
flags of steps 1-5) and acts on what the receipt states. Every task, and every test
input in it, gets TWO attempts (attempt_1, attempt_2). It never leaves a task empty.

How it solves
  1. Strategies are switched on by the receipt:
       transform:<name>      the candidates that fit every training pair (best first)
       partial:<name>        the closest library functions when nothing fits
       shift_consistent      the receipt says every pair moves its colours by the same shift
       fill_enclosed         the receipt says holes are filled in
       colour_map_input      same-size tasks: the colour change learned from training
       soft_local_<kind>     same-size tasks: neighbourhood rule by majority vote
       majority_output       all training outputs share a size: cell-by-cell majority
       resize_input          the sizes step predicted an output size
       copy_input            always on, the guaranteed answer
  2. Every strategy is judged on the TRAINING pairs only. Strategies that learn
     something are judged leave-one-out (learn on the other pairs, predict the one
     held back), so a rule that merely memorises scores badly.
  3. Score of a prediction = 0.55 x pairs reproduced exactly
                           + 0.35 x share of cells right
                           + 0.10 x size check (1 = the size the sizes step predicted,
                                    0 = a different size, 0.5 = no size predicted)
                           + 0.03 for each other strategy that proposed the same grid
     (training evidence only; the test output is never looked at).
  4. attempt_1 = best scoring grid, attempt_2 = best scoring DIFFERENT grid. If only
     one grid exists, a fallback (majority output, blank grid, flipped input) fills in.

Outputs
  * solver/grid_solver/results/grid_solver_submission_<dataset>.json
        {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}  (one per test input;
        the Kaggle submission layout)
  * solver/grid_solver/results/grid_solver_receipts.json   which strategy made which
        attempt, with its training evidence (no grids), for the Grid Verifier
  * solver/grid_solver/results/grid_solver_results.csv     one row per task
  * tables in the terminal. The public test outputs are used ONLY to fill the
    "correct" columns and tables (a check, not part of solving).

Usage
  python solver/grid_solver/grid_solver.py
  (run steps 1-6 first, this step reads the receipts they write)
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
OUT_CSV = RESULTS_DIR / "grid_solver_results.csv"
OUT_RECEIPTS = RESULTS_DIR / "grid_solver_receipts.json"
DATASETS = ("training", "evaluation")

sys.path.insert(0, str(ROOT / "solver" / "grid_transformations"))
try:
    from grid_transformations import (load_transformations_receipts, check_transformations_receipts,
                                      find_transformations, apply_transformation, REGISTRY, shift,
                                      learn_local_rule, neighbourhood_keys, background_of)
except (ImportError, SystemExit) as error:
    raise SystemExit(f"Could not import grid_transformations ({error}).\n"
                     "Steps 1-6 must be in their solver/ folders.")

W_EXACT, W_CELL, W_SIZE, W_VOTE, VOTE_CAP = 0.55, 0.35, 0.10, 0.03, 0.06
MAX_TRANSFORM_CANDIDATES = 6


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


def valid(pred):
    return (isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.size > 0
            and pred.shape[0] <= 30 and pred.shape[1] <= 30
            and int(pred.min()) >= 0 and int(pred.max()) <= 9)


def safe_apply(fn, train, g, target):
    try:
        pred = fn(train, g, target)
    except Exception:
        return None
    if pred is None:
        return None
    pred = np.asarray(pred, dtype=int)
    return pred if valid(pred) else None


# ----------------------------------------------------------------------
# Strategies: fn(train, grid, target_shape) -> grid or None
# ----------------------------------------------------------------------
def all_same_shape(train):
    return all(a.shape == b.shape for a, b in train)


def copy_input_fn(train, g, target):
    return g


def colour_map_fn(train, g, target):
    if not all_same_shape(train):
        return None
    votes = defaultdict(Counter)
    for a, b in train:
        for src, dst in zip(a.ravel().tolist(), b.ravel().tolist()):
            votes[src][dst] += 1
    mapping = {s: c.most_common(1)[0][0] for s, c in votes.items()}
    return np.array([[mapping.get(c, c) for c in row] for row in g.tolist()])


def soft_local_fn(kind):
    def fn(train, g, target):
        if not all_same_shape(train):
            return None
        votes = defaultdict(Counter)
        for a, b in train:
            for key, out in zip(neighbourhood_keys(a, kind), b.ravel().tolist()):
                votes[key][out] += 1
        rule = {k: c.most_common(1)[0][0] for k, c in votes.items()}
        keys = neighbourhood_keys(g, kind)
        return np.array([rule.get(k, c) for k, c in zip(keys, g.ravel().tolist())]).reshape(g.shape)
    return fn


def strict_local_fn(kind):
    def fn(train, g, target):
        rule = learn_local_rule(train, kind) if all_same_shape(train) else None
        if rule is None:
            return None
        keys = neighbourhood_keys(g, kind)
        return np.array([rule.get(k, c) for k, c in zip(keys, g.ravel().tolist())]).reshape(g.shape)
    return fn


def majority_output_fn(train, g, target):
    outputs = [b for _, b in train]
    if len({o.shape for o in outputs}) != 1:
        return None
    stack = np.stack(outputs)
    counts = np.stack([(stack == c).sum(0) for c in range(10)]).astype(float)
    counts += 0.5 * np.stack([outputs[0] == c for c in range(10)])      # ties go to the first output
    return counts.argmax(0)


def resize_input_fn(train, g, target):
    if target is None:
        return None
    bg = background_of(g)
    out = np.full(tuple(target), 0 if bg is None else bg, dtype=int)
    h, w = min(g.shape[0], out.shape[0]), min(g.shape[1], out.shape[1])
    out[:h, :w] = g[:h, :w]
    return out


def enclosed_background(g):
    """(background colour, mask of background cells that cannot reach the border)."""
    bg = background_of(g)
    if bg is None:
        return None, None
    h, w = g.shape
    reached = np.zeros(g.shape, bool)
    stack = [(y, x) for y in range(h) for x in range(w)
             if (y in (0, h - 1) or x in (0, w - 1)) and g[y, x] == bg]
    for y, x in stack:
        reached[y, x] = True
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not reached[ny, nx] and g[ny, nx] == bg:
                reached[ny, nx] = True
                stack.append((ny, nx))
    return bg, (g == bg) & ~reached


def fill_enclosed_fn(train, g, target):
    colours = Counter()
    for a, b in train:
        if a.shape != b.shape:
            return None
        _, mask = enclosed_background(a)
        if mask is None:
            return None
        colours.update(b[mask].tolist())
    if not colours:
        return None
    _, mask = enclosed_background(g)
    if mask is None or not mask.any():
        return None
    out = g.copy()
    out[mask] = colours.most_common(1)[0][0]
    return out


def build_strategies(train, receipt):
    """The strategies this task's receipt switches on, in tie-break order."""
    flags = set(receipt["flags"]) | set(receipt["sometimes"])
    strategies = []

    def add(name, fn, mode):
        strategies.append({"name": name, "fn": fn, "mode": mode, "rank": len(strategies)})

    specs, _ = find_transformations(train)
    for spec in specs[:MAX_TRANSFORM_CANDIDATES]:
        if spec["kind"] == "whole_grid":
            add(f"transform:{spec['name']}", lambda tr, g, t, s=spec: apply_transformation(s, g)[0], "fit")
        else:
            add(f"transform:{spec['name']}", strict_local_fn(spec["neighbourhood"]), "loo")

    shift_value = receipt["steps"]["rows_columns"]["consistent_shift"]
    if shift_value is not None:
        add("shift_consistent", lambda tr, g, t, s=shift_value: shift(g, s[0], s[1]), "fit")
    if "SHAPE_HOLES_FILLED_IN" in flags:
        add("fill_enclosed", fill_enclosed_fn, "loo")
    if all_same_shape(train):
        add("colour_map_input", colour_map_fn, "loo")
        for kind in ("cross_rel", "block3_rel", "cross"):
            add(f"soft_local_{kind}", soft_local_fn(kind), "loo")
    for name, _fits in receipt["partial"]:
        if name in REGISTRY:
            add(f"partial:{name}", lambda tr, g, t, n=name: REGISTRY[n](g), "fit")
    if len({b.shape for _, b in train}) == 1:
        add("majority_output", majority_output_fn, "loo")
    if any(s is not None for s in receipt["steps"]["sizes"]["predicted_output_sizes"]):
        add("resize_input", resize_input_fn, "fit")
    add("copy_input", copy_input_fn, "fit")
    return strategies


# ----------------------------------------------------------------------
# Evidence (training pairs only)
# ----------------------------------------------------------------------
def cell_accuracy(pred, target):
    return 0.0 if pred is None or pred.shape != target.shape else float((pred == target).mean())


def training_evidence(strategy, train):
    """(share of pairs reproduced exactly, mean cell accuracy) on the training pairs."""
    exact, cell = [], []
    learns = strategy["mode"] == "loo" and len(train) >= 2
    for i, (a, b) in enumerate(train):
        known = train[:i] + train[i + 1:] if learns else train
        pred = safe_apply(strategy["fn"], known, a, b.shape)
        exact.append(1.0 if pred is not None and pred.shape == b.shape and np.array_equal(pred, b) else 0.0)
        cell.append(cell_accuracy(pred, b))
    return sum(exact) / len(exact), sum(cell) / len(cell)


def size_check(pred, size):
    if size is None:
        return 0.5
    return 1.0 if pred.shape == tuple(size) else 0.0


# ----------------------------------------------------------------------
# Solve one test input
# ----------------------------------------------------------------------
def fallback_attempt(train, g, size, avoid):
    options = [("fallback:majority_output", safe_apply(majority_output_fn, train, g, size)),
               ("fallback:resize_input", safe_apply(resize_input_fn, train, g, size)),
               ("fallback:blank_grid", np.full(tuple(size) if size else g.shape,
                                               background_of(g) if background_of(g) is not None else 0)),
               ("fallback:flip_h_input", g[:, ::-1]), ("fallback:transpose_input", g.T)]
    for name, pred in options:
        if pred is not None and valid(np.asarray(pred)) and not same_grid(np.asarray(pred), avoid):
            return name, np.asarray(pred, dtype=int)
    return "fallback:repeat", avoid


def same_grid(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def solve_test_input(train, g, size, strategies, evidence):
    groups = []                                           # one entry per distinct grid
    for strategy in strategies:
        pred = safe_apply(strategy["fn"], train, g, size)
        if pred is None:
            continue
        exact, cell = evidence[strategy["name"]]
        score = W_EXACT * exact + W_CELL * cell + W_SIZE * size_check(pred, size)
        for group in groups:
            if same_grid(group["pred"], pred):
                group["supporters"].append(strategy["name"])
                break
        else:
            groups.append({"pred": pred, "name": strategy["name"], "score": score, "exact": exact,
                           "cell": cell, "size_ok": size_check(pred, size), "rank": strategy["rank"],
                           "supporters": []})
    for group in groups:
        group["final"] = group["score"] + min(W_VOTE * len(group["supporters"]), VOTE_CAP)
    groups.sort(key=lambda gr: (-gr["final"], gr["rank"]))

    attempts = groups[:2]
    if len(attempts) < 2:
        avoid = attempts[0]["pred"] if attempts else g
        name, pred = fallback_attempt(train, g, size, avoid)
        attempts.append({"pred": pred, "name": name, "final": 0.0, "exact": 0.0, "cell": 0.0,
                         "size_ok": size_check(pred, size), "supporters": []})
    return attempts, groups


# ----------------------------------------------------------------------
# One task
# ----------------------------------------------------------------------
def solve_task(task, receipt):
    train = [(to_np(p["input"]), to_np(p["output"])) for p in task["train"]]
    strategies = build_strategies(train, receipt)
    evidence = {s["name"]: training_evidence(s, train) for s in strategies}
    sizes = receipt["steps"]["sizes"]["predicted_output_sizes"]

    submission, test_receipts, predictions = [], [], []
    for k, t in enumerate(task["test"]):
        size = sizes[k] if k < len(sizes) else None
        attempts, ranked = solve_test_input(train, to_np(t["input"]), size, strategies, evidence)
        submission.append({"attempt_1": attempts[0]["pred"].tolist(), "attempt_2": attempts[1]["pred"].tolist()})
        predictions.append([a["pred"] for a in attempts])
        test_receipts.append({
            "predicted_size": size,
            "attempts": [{"strategy": a["name"], "score": round(a["final"], 3), "exact": round(a["exact"], 3),
                          "cell": round(a["cell"], 3), "size_ok": a["size_ok"],
                          "also_proposed_by": a["supporters"]} for a in attempts],
            "distinct_attempts": not same_grid(attempts[0]["pred"], attempts[1]["pred"]),
            "ranked": [{"strategy": g_["name"], "score": round(g_["final"], 3)} for g_ in ranked[:5]]})
    strategy_log = [{"strategy": s["name"], "mode": s["mode"], "exact": round(evidence[s["name"]][0], 3),
                     "cell": round(evidence[s["name"]][1], 3)} for s in strategies]
    return submission, {"test": test_receipts, "strategies": strategy_log}, predictions


def check_on_public_test(task, predictions):
    """Per test input: which attempt (if any) is exactly right. Verification only."""
    if not all("output" in t for t in task["test"]):
        return None
    results = []
    for t, attempts in zip(task["test"], predictions):
        target = to_np(t["output"])
        results.append({"a1": same_grid(attempts[0], target), "a2": same_grid(attempts[1], target),
                        "a1_shape": attempts[0].shape == target.shape,
                        "a2_shape": attempts[1].shape == target.shape})
    return results


# ----------------------------------------------------------------------
# For the Grid Verifier
# ----------------------------------------------------------------------
def load_solver_receipts(path=OUT_RECEIPTS):
    """{dataset: {task_id: receipt}}. Stops with a clear message if the file is missing."""
    if not Path(path).exists():
        raise SystemExit(f"Grid solver receipts not found: {path}\n"
                         f"Run solver/grid_solver/grid_solver.py first.")
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def check_solver_receipts(receipts, dataset, task_ids):
    """Stop if any task has no solver receipt (every task must have two attempts)."""
    have = receipts.get(dataset, {})
    missing = [t for t in task_ids if t not in have]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(f"Grid solver receipt missing for {len(missing)} of {len(task_ids)} "
                         f"{dataset} tasks: {shown}\nRe-run solver/grid_solver/grid_solver.py.")
    print(f"Grid solver receipts: {len(task_ids)}/{len(task_ids)} {dataset} tasks covered")


# ----------------------------------------------------------------------
# Tables and CSV
# ----------------------------------------------------------------------
def family(name):
    return name.split(":")[0] if ":" in name else name


def csv_row(dataset, task_id, transformation, solver, check):
    tests = solver["test"]
    join = lambda values: ";".join(str(v) for v in values)
    row = {
        "dataset": dataset, "task_id": task_id, "group": transformation["group"],
        "story": transformation["story"], "n_test_inputs": len(tests),
        "n_strategies": len(solver["strategies"]),
        "attempt_1_strategy": join(t["attempts"][0]["strategy"] for t in tests),
        "attempt_1_score": join(t["attempts"][0]["score"] for t in tests),
        "attempt_2_strategy": join(t["attempts"][1]["strategy"] for t in tests),
        "attempt_2_score": join(t["attempts"][1]["score"] for t in tests),
        "distinct_attempts": join("yes" if t["distinct_attempts"] else "no" for t in tests),
    }
    if check is None:
        row.update({"attempt_1_correct": "unknown", "attempt_2_correct": "unknown", "solved": "unknown",
                    "task_score": "", "attempt_1_shape_ok": "unknown", "solved_by": ""})
    else:
        solved = [c["a1"] or c["a2"] for c in check]
        row.update({
            "attempt_1_correct": join("yes" if c["a1"] else "no" for c in check),
            "attempt_2_correct": join("yes" if c["a2"] else "no" for c in check),
            "solved": "yes" if all(solved) else "partial" if any(solved) else "no",
            "task_score": round(sum(solved) / len(solved), 3),
            "attempt_1_shape_ok": join("yes" if c["a1_shape"] else "no" for c in check),
            "solved_by": join(t["attempts"][0 if c["a1"] else 1]["strategy"] if (c["a1"] or c["a2"]) else "-"
                              for t, c in zip(tests, check)),
        })
    return row


def print_summary(dataset, rows):
    total = len(rows)
    checked = [r for r in rows if r["solved"] != "unknown"]
    print(f"\n=== {dataset}: grid solver ({total} tasks, every task has 2 attempts) ===")
    if not checked:
        print("No public test outputs to check against.")
        return
    score = sum(r["task_score"] for r in checked)
    solved = sum(r["solved"] == "yes" for r in checked)
    first = sum(all(v == "yes" for v in r["attempt_1_correct"].split(";")) for r in checked)
    shape = sum(all(v == "yes" for v in r["attempt_1_shape_ok"].split(";")) for r in checked)
    print(f"Score (mean over tasks, either attempt): {score:.1f}/{len(checked)} = {100 * score / len(checked):.1f}%")
    print(f"  tasks fully solved        : {solved}   (attempt_1 alone solves {first})")
    print(f"  attempt_1 right size      : {shape}/{len(checked)}")
    print(f"  attempts identical        : {sum('no' in r['distinct_attempts'].split(';') for r in rows)} tasks")

    print("\nWhich strategy made the correct attempt")
    by_family = Counter()
    made = Counter()
    for r in rows:
        for name in r["attempt_1_strategy"].split(";") + r["attempt_2_strategy"].split(";"):
            made[family(name)] += 1
    for r in checked:
        for name in r["solved_by"].split(";"):
            if name != "-":
                by_family[family(name)] += 1
    header = f"{'Strategy':<22}{'Attempts made':>15}{'Correct':>10}"
    print(header)
    print("-" * len(header))
    for name, n in made.most_common():
        print(f"{name:<22}{n:>15}{by_family.get(name, 0):>10}")

    print("\nBy task group (size category | palette | shape count), groups with 20+ tasks")
    header = f"{'Group':<26}{'Tasks':>7}{'Solved':>8}{'Score':>8}"
    print(header)
    print("-" * len(header))
    by_group = defaultdict(list)
    for r in checked:
        by_group[r["group"]].append(r)
    for group, items in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        if len(items) >= 20:
            print(f"{group:<26}{len(items):>7}{sum(r['solved'] == 'yes' for r in items):>8}"
                  f"{100 * sum(r['task_score'] for r in items) / len(items):>7.1f}%")


def main():
    transformations_in = load_transformations_receipts()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, solver_receipts = [], {}
    for dataset in DATASETS:
        folder = DATA_DIR / dataset
        if not folder.is_dir():
            print(f"Note: {folder} not found, skipping {dataset}")
            continue
        tasks = load_tasks(folder)
        check_transformations_receipts(transformations_in, dataset, list(tasks))

        rows, submission, solver_receipts[dataset] = [], {}, {}
        for n, (task_id, task) in enumerate(tasks.items(), 1):
            transformation = transformations_in[dataset][task_id]
            task_submission, receipt, predictions = solve_task(task, transformation)
            submission[task_id] = task_submission
            solver_receipts[dataset][task_id] = receipt
            rows.append(csv_row(dataset, task_id, transformation, receipt, check_on_public_test(task, predictions)))
            if n % 200 == 0:
                print(f"  {dataset}: {n}/{len(tasks)} tasks solved", flush=True)

        check_solver_receipts(solver_receipts, dataset, list(tasks))
        with (RESULTS_DIR / f"grid_solver_submission_{dataset}.json").open("w", encoding="utf-8") as file:
            json.dump(submission, file)
        print_summary(dataset, rows)
        all_rows += rows

    if not all_rows:
        raise SystemExit(f"No tasks found under {DATA_DIR}. Expected data/training and data/evaluation.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with OUT_RECEIPTS.open("w", encoding="utf-8") as file:
        json.dump(solver_receipts, file)
    print(f"\nSubmission files written to {RESULTS_DIR} (grid_solver_submission_<dataset>.json)")
    print(f"Per-task results written to {OUT_CSV}")
    print(f"Solver receipts for the verifier written to {OUT_RECEIPTS}")


if __name__ == "__main__":
    main()