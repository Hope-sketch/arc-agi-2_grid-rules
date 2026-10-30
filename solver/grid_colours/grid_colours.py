"""
Step 2: Grid colours (investigation only, not used to solve yet)


Section 1: colours added / removed / maintained / other
  For every training pair, compare the SET of colours in the input with the
  SET of colours in the output:
      added              output has colours the input did not (and none dropped)
      removed            input has colours the output dropped (and none added)
      maintained         same set of colours
      added_and_removed  both happened in the same pair
  Then look at the whole task (all training pairs):
      every pair the same kind  -> that group
      anything else             -> "other"

Section 2: background colour
  Background = the most frequent colour in a grid.
  Per pair: maintained (same background in and out) or vanishes (changes).
  Per task: maintained / vanishes / other (pairs disagree).
  If two colours tie for most frequent in a grid, the background is NOT
  guessed. That pair abstains and the task is decided from the other pairs,
  as long as at least half of the pairs could be decided. Otherwise the task
  is marked "unknown".

Test check (this is the "where does it fail" part)
  Decisions come from the training pairs only. Afterwards the public test
  answers are used to ask: does the same thing also happen in the test pair?
  holds = yes, breaks = no, n/a = the task is "other"/"unknown".

Outputs
  * tables in the terminal
  * grid_colour_results.csv (one row per task; dataset + task_id come first)
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_CSV = Path("grid_colour_results.csv")

# Your hand-sorted examples from Grid_Colour.pdf (used only for a self-check).
HAND_COLOUR_GROUPS = {
    "added": "d304284e d364b489 d406998b d4a91cb9 d4f3cd78",
    "removed": "d35bdbdc d4c90558 d56f2372 d687bc17 d89b689b",
    "maintained": "d37a1ef5 d43fd935 d492a647 d4b1c2b1 d511f180",
    "other": "d2acf2cb d4469b4b d5c634a2 d5d6de2d",
}
HAND_BG_GROUPS = {
    "maintained": "a78176bb a79310a0 a834deea",
    "vanishes": "a740d043 a85d4709 a8610ef7",
    "other": "a6f40cea a87f7484",
}


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
# Colour helpers
# ----------------------------------------------------------------------
def colours_in(grid):
    """Which colours are in the grid (not how many cells of each)."""
    return {cell for row in grid for cell in row}


def background_colour(grid):
    """Most frequent colour, or None if empty or tied."""
    counts = Counter(cell for row in grid for cell in row)
    if not counts:
        return None
    ranked = counts.most_common(2)
    if len(ranked) == 2 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def colour_change(inp, out):
    cin, cout = colours_in(inp), colours_in(out)
    added = sorted(cout - cin)
    removed = sorted(cin - cout)
    if added and removed:
        kind = "added_and_removed"
    elif added:
        kind = "added"
    elif removed:
        kind = "removed"
    else:
        kind = "maintained"
    return {"kind": kind, "added": added, "removed": removed}


def background_change(inp, out):
    b_in, b_out = background_colour(inp), background_colour(out)
    if b_in is None or b_out is None:
        kind = "unknown"
    elif b_in == b_out:
        kind = "maintained"
    else:
        kind = "vanishes"
    return {"kind": kind, "bg_in": b_in, "bg_out": b_out}


# ----------------------------------------------------------------------
# Task level
# ----------------------------------------------------------------------
def colour_group(pair_changes):
    kinds = {c["kind"] for c in pair_changes}
    if len(kinds) == 1:
        kind = next(iter(kinds))
        if kind == "added_and_removed":
            return "other", "added_and_removed_in_every_pair"
        return kind, "same_in_every_pair"
    return "other", "pairs_disagree"


def background_group(pair_changes):
    """
    A pair with a tie for most frequent colour abstains (it says nothing).
    The task is decided from the other pairs, but only if at least half of
    the pairs could be decided; otherwise it stays "unknown".
    """
    total = len(pair_changes)
    known = [c["kind"] for c in pair_changes if c["kind"] != "unknown"]
    if len(known) * 2 < total:
        return "unknown", "tie_in_most_pairs"

    note = "" if len(known) == total else f" (decided from {len(known)} of {total} pairs)"
    kinds = set(known)
    if len(kinds) == 1:
        return next(iter(kinds)), "same_in_every_pair" + note
    return "other", "pairs_disagree" + note


def change_is_consistent(group, pair_changes):
    """For added/removed tasks: is it the same colour(s) in every pair?"""
    if group == "added":
        return len({tuple(c["added"]) for c in pair_changes}) == 1
    if group == "removed":
        return len({tuple(c["removed"]) for c in pair_changes}) == 1
    return None


def test_check(group, test_kinds):
    if group in ("other", "unknown", "none") or not test_kinds:
        return "n/a"
    return "holds" if set(test_kinds) == {group} else "breaks"


def analyse_task(task):
    train = task.get("train", [])
    test = [t for t in task.get("test", []) if "output" in t]

    if not train:
        return {"colour_group": "none", "colour_reason": "no_training_examples",
                "consistent": None, "colour_test": "n/a",
                "bg_group": "none", "bg_reason": "no_training_examples",
                "bg_test": "n/a", "per_pair_colours": "", "per_pair_bg": ""}

    col = [colour_change(p["input"], p["output"]) for p in train]
    bg = [background_change(p["input"], p["output"]) for p in train]
    col_test = [colour_change(t["input"], t["output"])["kind"] for t in test]
    bg_test = [background_change(t["input"], t["output"])["kind"] for t in test]

    cg, c_reason = colour_group(col)
    bgg, b_reason = background_group(bg)

    return {
        "colour_group": cg,
        "colour_reason": c_reason,
        "consistent": change_is_consistent(cg, col),
        "colour_test": test_check(cg, col_test),
        "bg_group": bgg,
        "bg_reason": b_reason,
        "bg_test": test_check(bgg, bg_test),
        "per_pair_colours": " | ".join(f"+{c['added']} -{c['removed']}" for c in col),
        "per_pair_bg": " | ".join(f"{b['bg_in']}>{b['bg_out']}" for b in bg),
    }


def analyse_dataset(tasks):
    results = []
    for task_id, task in tasks.items():
        row = analyse_task(task)
        row["task_id"] = task_id
        results.append(row)
    return results


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def print_group_table(title, results, group_key, test_key, order):
    print(f"\n{title}")
    header = f"{'Group':<14}{'Tasks':>7}{'Holds':>8}{'Breaks':>8}{'n/a':>6}{'Holds (when checked)':>24}"
    print(header)
    print("-" * len(header))
    by_group = defaultdict(list)
    for r in results:
        by_group[r[group_key]].append(r)
    for group in order:
        rows = by_group.get(group, [])
        if not rows:
            continue
        h = sum(1 for r in rows if r[test_key] == "holds")
        b = sum(1 for r in rows if r[test_key] == "breaks")
        n = len(rows) - h - b
        rate = f"{h / (h + b):.1%}" if (h + b) else "-"
        print(f"{group:<14}{len(rows):>7}{h:>8}{b:>8}{n:>6}{rate:>24}")


def print_crosstab(results):
    colour_order = ["added", "removed", "maintained", "other"]
    bg_order = ["maintained", "vanishes", "other", "unknown"]
    print("\nColour group (rows) x background group (columns):")
    print(f"{'':<14}" + "".join(f"{b:>12}" for b in bg_order))
    for cg in colour_order:
        counts = Counter(r["bg_group"] for r in results if r["colour_group"] == cg)
        print(f"{cg:<14}" + "".join(f"{counts.get(b, 0):>12}" for b in bg_order))


def print_dataset(name, results):
    print(f"\n=== {name}: colours ({len(results)} tasks) ===")
    print_group_table("Section 1: colour group (does the training pattern hold on the test pair?)",
                      results, "colour_group", "colour_test",
                      ["added", "removed", "maintained", "other"])
    reasons = Counter(r["colour_reason"] for r in results if r["colour_group"] == "other")
    if reasons:
        print("  Why 'other':", dict(reasons))
    for g in ("added", "removed"):
        rows = [r for r in results if r["colour_group"] == g]
        same = sum(1 for r in rows if r["consistent"])
        print(f"  {g}: same colour(s) {g} in every pair for {same}/{len(rows)} tasks")

    print_group_table("Section 2: background group (does the training pattern hold on the test pair?)",
                      results, "bg_group", "bg_test",
                      ["maintained", "vanishes", "other", "unknown"])
    print_crosstab(results)


def check_hand_lists(all_results):
    """Compare the script with the examples you sorted by hand in the PDF."""
    lookup = defaultdict(list)
    for dataset, results in all_results.items():
        for r in results:
            lookup[r["task_id"]].append(r)

    print("\n=== Check against your hand-sorted lists ===")
    for label, key, hand in (("Colour groups", "colour_group", HAND_COLOUR_GROUPS),
                             ("Background groups", "bg_group", HAND_BG_GROUPS)):
        print(f"\n{label}:")
        for group, ids in hand.items():
            ids = ids.split()
            ok = 0
            for task_id in ids:
                rows = lookup.get(task_id)
                if not rows:
                    print(f"  {group}: {task_id} not found in the data (check the ID)")
                    continue
                if any(r[key] == group for r in rows):
                    ok += 1
                else:
                    print(f"  {group}: {task_id} -> script says {rows[0][key]}"
                          f" ({rows[0]['colour_reason' if key == 'colour_group' else 'bg_reason']})")
            print(f"  {group}: {ok}/{len(ids)} match")


def write_csv(all_results):
    columns = ["dataset", "task_id", "colour_group", "colour_reason", "consistent",
               "colour_test", "bg_group", "bg_reason", "bg_test",
               "per_pair_colours", "per_pair_bg"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for dataset, results in all_results.items():
            for r in results:
                writer.writerow([dataset] + [r[c] for c in columns[1:]])
    print(f"\nPer-task results written to {RESULTS_CSV}")


# ----------------------------------------------------------------------
def main():
    datasets = {
        "training": load_tasks(DATA_DIR / "training"),
        "evaluation": load_tasks(DATA_DIR / "evaluation"),
    }
    all_results = {}
    for name, tasks in datasets.items():
        all_results[name] = analyse_dataset(tasks)
        print_dataset(name, all_results[name])

    check_hand_lists(all_results)
    write_csv(all_results)


if __name__ == "__main__":
    main()
