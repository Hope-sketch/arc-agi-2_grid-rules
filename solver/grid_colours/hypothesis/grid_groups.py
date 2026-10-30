"""
Step 2b: Grid groups (sort the tasks until they reach the Grid Solver)

Nothing is predicted here. Every task is put in exactly ONE group, built from
two findings that already exist:

  SIZE word   (from grid_sizes.py, training pairs only)
      STAY    rule 1  input == output in every pair
      FIXED   rule 2  inputs all same size, outputs all same size
      FUNNEL  rule 3  inputs vary, outputs all one size
      SPREAD  rule 4  inputs all same size, outputs vary
      FREE    rule 5  inputs vary and outputs vary (or pairs mixed)

  PALETTE word (from grid_colours.py, input palette vs output palette)
      KEEP    same colours in and out, in every pair
      GAIN    colours added in every pair, none dropped
      DROP    colours dropped in every pair, none added
      SWAP    colours added AND dropped in every pair
      MIXED   the pairs disagree with each other

  Group name = SIZE-PALETTE, e.g. STAY-KEEP = the grid keeps its size and
  keeps its colour palette. 5 x 5 = 25 possible groups.

Extra labels that travel with the task (not used to split groups):
  palette_rule, background group, direction labels, and the test checks of both steps.

Outputs
  * a group table per dataset in the terminal
  * results/grid_groups.csv (one row per task)

Run from the project root:
  python solver/grid_groups.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import grid_colours
import grid_sizes

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "grid_groups.csv"

SIZE_WORD = {0: "NONE", 1: "STAY", 2: "FIXED", 3: "FUNNEL", 4: "SPREAD", 5: "FREE"}
SIZE_ORDER = ["STAY", "FIXED", "FUNNEL", "SPREAD", "FREE"]
PALETTE_ORDER = ["KEEP", "GAIN", "DROP", "SWAP", "MIXED"]


def palette_word(colour_group, colour_reason):
    if colour_group == "maintained":
        return "KEEP"
    if colour_group == "added":
        return "GAIN"
    if colour_group == "removed":
        return "DROP"
    if colour_reason == "added_and_removed_in_every_pair":
        return "SWAP"
    return "MIXED"


def group_task(task):
    size = grid_sizes.analyse_task(task)
    colour = grid_colours.analyse_task(task)
    size_word = SIZE_WORD[size["rule"]]
    pal_word = palette_word(colour["colour_group"], colour["colour_reason"])
    return {
        "group": f"{size_word}-{pal_word}",
        "size_word": size_word,
        "palette_word": pal_word,
        "size_rule": size["rule"],
        "size_test_check": size["test_check"],
        "train_pairs": size["train_pairs"],
        "directions": size["directions"],
        "palette_rule": colour["palette_rule"],
        "palette_test": colour["palette_test"],
        "background_group": colour["bg_group"],
        "per_pair_colours": colour["per_pair_colours"],
    }


def print_matrix(name, results):
    print(f"\n=== {name}: groups (size word x palette word), {len(results)} tasks ===")
    header = f"{'':<9}" + "".join(f"{p:>8}" for p in PALETTE_ORDER) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    counts = Counter((r["size_word"], r["palette_word"]) for r in results.values())
    for s in SIZE_ORDER:
        row = [counts.get((s, p), 0) for p in PALETTE_ORDER]
        print(f"{s:<9}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}")
    print("-" * len(header))
    totals = [sum(counts.get((s, p), 0) for s in SIZE_ORDER) for p in PALETTE_ORDER]
    print(f"{'Total':<9}" + "".join(f"{n:>8}" for n in totals) + f"{sum(totals):>8}")

    print("\nBiggest groups:")
    by_group = Counter(r["group"] for r in results.values())
    for group, n in by_group.most_common(8):
        print(f"  {group:<14}{n:>5}  ({n / len(results):.1%})")


def write_csv(all_results):
    RESULTS_DIR.mkdir(exist_ok=True)
    cols = ["dataset", "task_id", "group", "size_word", "palette_word", "size_rule",
            "size_test_check", "palette_rule", "palette_test", "background_group",
            "train_pairs", "directions", "per_pair_colours"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for ds, res in all_results.items():
            for tid, r in sorted(res.items(), key=lambda kv: (kv[1]["group"], kv[0])):
                w.writerow([ds, tid] + [r[c] for c in cols[2:]])
    print(f"\nPer-task groups written to {RESULTS_CSV}")


def main():
    all_results = {}
    for name in ("training", "evaluation"):
        tasks = grid_sizes.load_tasks(DATA_DIR / name)
        all_results[name] = {tid: group_task(t) for tid, t in tasks.items()}
        print_matrix(name, all_results[name])
    write_csv(all_results)


if __name__ == "__main__":
    main()
