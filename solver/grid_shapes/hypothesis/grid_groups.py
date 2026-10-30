"""
Grid groups + shapes (investigation only, not used to solve yet)

Takes the groups you already have (size word + palette word, e.g. STAY-KEEP,
results/grid_groups_results.csv) and adds a SHAPE word from the shapes step
(results/grid_shapes_results.csv), giving groups like STAY-KEEP-MORE.

Shape word (from the shape COUNT rule of grid_shapes.py)
  MORE  : training outputs have more shapes than the inputs (added)
  FEWER : training outputs have fewer shapes than the inputs (removed)
  EVEN  : the same number of shapes (same)
  VARY  : the training pairs disagree (other)

What it answers
  1. Does adding the shape word split the old groups into purer groups?
     (compared with the baseline: the share of the most common shape word
     if you ignore sizes and palettes completely)
  2. Do the patterns found on the training tasks also show up on the
     evaluation tasks? (groups are judged on training only, evaluation is
     the check)

Usage
  python solver/grid_groups_shapes.py

Outputs
  * tables in the terminal
  * results/grid_groups_shapes_results.csv  (one row per task)
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

SOLVER_DIR = Path(__file__).resolve().parents[2]   # the solver/ folder

GROUPS_CSV = SOLVER_DIR / "grid_colours" / "results" / "grid_groups_results.csv"
SHAPES_CSV = SOLVER_DIR / "grid_shapes" / "results" / "grid_shapes_results.csv"
OUT_CSV = SOLVER_DIR / "grid_shapes" / "results" / "grid_groups_shapes_results.csv"

SHAPE_WORD = {"added": "MORE", "removed": "FEWER", "same": "EVEN", "other": "VARY"}
SHAPE_WORDS = ["MORE", "FEWER", "EVEN", "VARY"]

MIN_TRAIN_TASKS = 20   # smaller groups are too small to judge
MIN_EVAL_TASKS = 5     # fewer evaluation tasks than this are marked "few"


# ----------------------------------------------------------------------
# Load and join
# ----------------------------------------------------------------------
def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def join_rows():
    shapes = {(r["dataset"], r["task_id"]): r for r in read_rows(SHAPES_CSV)}
    joined, missing = [], 0
    for g in read_rows(GROUPS_CSV):
        s = shapes.get((g["dataset"], g["task_id"]))
        if s is None:
            missing += 1
            shape_word = "NONE"
            s = {}
        else:
            shape_word = SHAPE_WORD.get(s["count_group"], "VARY")
        joined.append({
            "dataset": g["dataset"],
            "task_id": g["task_id"],
            "group": g["group"],
            "shape_word": shape_word,
            "combined_group": f"{g['group']}-{shape_word}",
            "size_rule": g["size_rule"],
            "palette_rule": g["palette_rule"],
            "count_test": s.get("count_test", "n/a"),
            "form_group": s.get("form_group", "n/a"),
            "form_test": s.get("form_test", "n/a"),
            "test_vs_train_inputs": s.get("test_vs_train_inputs", "n/a"),
        })
    return joined, missing


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def share_text(part, whole):
    return f"{part / whole:.0%}" if whole else "-"


def hold_text(rows):
    holds = sum(1 for r in rows if r["count_test"] == "holds")
    breaks = sum(1 for r in rows if r["count_test"] == "breaks")
    return share_text(holds, holds + breaks)


def top_word(rows):
    word, n = Counter(r["shape_word"] for r in rows).most_common(1)[0]
    return word, n


def print_split_table(rows, dataset):
    """For each old group: how its tasks spread over the shape words."""
    baseline_word, baseline_n = top_word(rows)
    print(f"\nBaseline (ignoring sizes and palettes): {baseline_word} covers "
          f"{share_text(baseline_n, len(rows))} of {len(rows)} {dataset} tasks")

    header = (f"{'Group':<14}{'Tasks':>6}"
              + "".join(f"{w:>7}" for w in SHAPE_WORDS)
              + f"{'Top':>8}{'Share':>7}{'Shape holds':>13}")
    print(header)
    print("-" * len(header))

    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    for group, g_rows in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        counts = Counter(r["shape_word"] for r in g_rows)
        word, n = top_word(g_rows)
        small = "  (small)" if len(g_rows) < MIN_TRAIN_TASKS else ""
        print(f"{group:<14}{len(g_rows):>6}"
              + "".join(f"{counts[w]:>7}" for w in SHAPE_WORDS)
              + f"{word:>8}{share_text(n, len(g_rows)):>7}{hold_text(g_rows):>13}{small}")


def print_generalisation_table(train_rows, eval_rows):
    """Does the top shape word of a training group also lead on evaluation?"""
    print(f"\nDoes the training pattern show up on evaluation? "
          f"(training groups with {MIN_TRAIN_TASKS}+ tasks)")
    header = (f"{'Group':<14}{'Train':>6}{'Top':>8}{'Share':>7}"
              f"{'Eval':>6}{'Same word':>11}{'Share':>7}")
    print(header)
    print("-" * len(header))

    train_by = defaultdict(list)
    for r in train_rows:
        train_by[r["group"]].append(r)
    eval_by = defaultdict(list)
    for r in eval_rows:
        eval_by[r["group"]].append(r)

    for group, t_rows in sorted(train_by.items(), key=lambda kv: -len(kv[1])):
        if len(t_rows) < MIN_TRAIN_TASKS:
            continue
        word, n = top_word(t_rows)
        e_rows = eval_by.get(group, [])
        e_same = sum(1 for r in e_rows if r["shape_word"] == word)
        few = "  (few)" if len(e_rows) < MIN_EVAL_TASKS else ""
        print(f"{group:<14}{len(t_rows):>6}{word:>8}{share_text(n, len(t_rows)):>7}"
              f"{len(e_rows):>6}{e_same:>11}{share_text(e_same, len(e_rows)):>7}{few}")


# ----------------------------------------------------------------------
def write_csv(rows):
    columns = ["dataset", "task_id", "group", "shape_word", "combined_group",
               "size_rule", "palette_rule", "count_test", "form_group",
               "form_test", "test_vs_train_inputs"]
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] for c in columns])
    print(f"\nPer-task results written to {OUT_CSV}")


def main():
    rows, missing = join_rows()
    if missing:
        print(f"Warning: {missing} tasks have no row in {SHAPES_CSV.name} "
              f"(shape word NONE, left out of the tables)")
    usable = [r for r in rows if r["shape_word"] != "NONE"]
    train = [r for r in usable if r["dataset"] == "training"]
    evaluation = [r for r in usable if r["dataset"] == "evaluation"]

    print(f"\n=== training: groups + shape word ({len(train)} tasks) ===")
    print_split_table(train, "training")
    print_generalisation_table(train, evaluation)

    print(f"\n=== evaluation: groups + shape word ({len(evaluation)} tasks) ===")
    print_split_table(evaluation, "evaluation")

    write_csv(rows)


if __name__ == "__main__":
    main()