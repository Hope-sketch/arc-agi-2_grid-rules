"""
Step 2: Grid colours (investigation only, not used to solve yet)

Follows your Grid_Colour.pdf.

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

Section 1b: the colour PALETTE rule (tightened)
  For every training pair the palette of the input and the palette of the
  output are compared (kept / added / removed). Then the solver looks for ONE
  rule that explains the output palette from the input palette in EVERY
  training pair, trying the simplest first:
      same             output palette == input palette
      fixed_output     the output palette is the same set in every pair
      add_fixed        output = input + the same colour(s) every pair
      remove_fixed     output = input - the same colour(s) every pair
      swap_fixed       output = input - fixed colour(s) + fixed colour(s)
      remove_role      the dropped colour has the same ROLE every pair
                       (background / commonest non-background / rarest non-background)
      subset           output palette is always inside the input palette, no fixed rule
      superset         output palette always contains the input palette, no fixed rule
      other            nothing above fits
  The rule is then checked on the test pair (holds / breaks / n/a), and the
  test input palette is compared with the training input palettes
  (same_as_training / inside_training_colours / has_new_colours).

Section 2: background colour
  Background = the most frequent colour in a grid.
  Per pair: maintained (same background in and out) or vanishes (changes).
  Per task: maintained / vanishes / other (pairs disagree).
  If two colours tie for most frequent in a grid, the background is NOT
  guessed. That pair abstains and the task is decided from the other pairs,
  as long as at least half of the pairs could be decided. Otherwise the task
  is marked "unknown".

Section 3: colours and cells (only for tasks where the size step cannot decide)
  Tasks in size rules 4 and 5 (training grids only) get a check for a link
  between the colours and the OUTPUT SIZE. A link counts only if it holds in
  EVERY training pair. The links tried:
      area_eq_kept_cells        output area = input cells of the colours that stay
      area_eq_dropped_cells     output area = input cells of the colours that are dropped
      area_eq_nonbg_cells       output area = all input cells that are not background
      area_eq_rarest_cells      output area = cells of the rarest non-background colour
      area_eq_commonest_cells   output area = cells of the commonest non-background colour
      side_eq_n_colours         one side of the output = number of colours in the input
      side_eq_n_kept_colours    one side of the output = number of colours that stay
      side_eq_n_dropped_colours one side of the output = number of colours dropped
  Then the same link is checked on the public test pair (holds / breaks).
  The links are written to the CSV as evidence for the later steps.

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
RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "grid_colour_results.csv"

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
# Section 1b: palette rule
# ----------------------------------------------------------------------
def role_colour(grid, role):
    """The colour that has a role in this grid, or None (ties are not guessed)."""
    counts = Counter(c for row in grid for c in row)
    ranked = counts.most_common()
    if not ranked:
        return None
    if role == "background":
        return background_colour(grid)
    bg = background_colour(grid)
    others = [(c, n) for c, n in ranked if c != bg]
    if not others:
        return None
    if role == "commonest_nonbg":
        top = max(n for _, n in others)
        pick = [c for c, n in others if n == top]
    else:  # rarest_nonbg
        low = min(n for _, n in others)
        pick = [c for c, n in others if n == low]
    return pick[0] if len(pick) == 1 else None


def palette_rule(train):
    """Return (rule_name, detail, check) where check(test_in, test_out) -> bool."""
    ins = [colours_in(p["input"]) for p in train]
    outs = [colours_in(p["output"]) for p in train]
    added = [o - i for i, o in zip(ins, outs)]
    removed = [i - o for i, o in zip(ins, outs)]

    def names(cols):
        return "{" + ",".join(str(c) for c in sorted(cols)) + "}"

    if all(i == o for i, o in zip(ins, outs)):
        return "same", "output palette = input palette", \
            lambda ti, to: colours_in(ti) == colours_in(to)

    if len({frozenset(o) for o in outs}) == 1:
        fixed = outs[0]
        return "fixed_output", f"output palette is always {names(fixed)}", \
            lambda ti, to, fixed=fixed: colours_in(to) == fixed

    if len({frozenset(a) for a in added}) == 1 and not any(removed) and added[0]:
        a = added[0]
        return "add_fixed", f"input + {names(a)}", \
            lambda ti, to, a=a: colours_in(to) == colours_in(ti) | a

    if len({frozenset(r) for r in removed}) == 1 and not any(added) and removed[0]:
        r = removed[0]
        return "remove_fixed", f"input - {names(r)}", \
            lambda ti, to, r=r: colours_in(to) == colours_in(ti) - r

    if (len({frozenset(a) for a in added}) == 1 and len({frozenset(r) for r in removed}) == 1
            and added[0] and removed[0]):
        a, r = added[0], removed[0]
        return "swap_fixed", f"input - {names(r)} + {names(a)}", \
            lambda ti, to, a=a, r=r: colours_in(to) == (colours_in(ti) - r) | a

    if not any(added) and all(len(r) == 1 for r in removed):
        for role in ("background", "commonest_nonbg", "rarest_nonbg"):
            found = [role_colour(p["input"], role) for p in train]
            if all(f is not None and removed[k] == {f} for k, f in enumerate(found)):
                return "remove_role", f"the {role} colour is dropped", \
                    lambda ti, to, role=role: (role_colour(ti, role) is not None and
                                               colours_in(to) == colours_in(ti) - {role_colour(ti, role)})

    if all(o <= i for i, o in zip(ins, outs)):
        return "subset", "output palette is inside the input palette", \
            lambda ti, to: colours_in(to) <= colours_in(ti)
    if all(o >= i for i, o in zip(ins, outs)):
        return "superset", "output palette contains the input palette", \
            lambda ti, to: colours_in(to) >= colours_in(ti)
    return "other", "no palette rule fits every training pair", None


def test_input_vs_training(train, test):
    """How does the test input palette compare with the training input palettes?"""
    if not test:
        return "n/a"
    train_pal = [colours_in(p["input"]) for p in train]
    union = set().union(*train_pal) if train_pal else set()
    kinds = []
    for t in test:
        tp = colours_in(t["input"])
        if tp in train_pal:
            kinds.append("same_as_training")
        elif tp <= union:
            kinds.append("inside_training_colours")
        else:
            kinds.append("has_new_colours")
    return kinds[0] if len(set(kinds)) == 1 else "mixed"


def palette_check(train, test):
    rule, detail, check = palette_rule(train)
    if check is None or not test:
        result = "n/a"
    else:
        result = "holds" if all(check(t["input"], t["output"]) for t in test) else "breaks"
    return rule, detail, result


# ----------------------------------------------------------------------
# Section 3: colours and cells (output size)
# ----------------------------------------------------------------------
def grid_size(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


def size_rule(train):
    """Size rule from TRAINING grids only (same definition as grid_sizes.py)."""
    ins = [grid_size(p["input"]) for p in train]
    outs = [grid_size(p["output"]) for p in train]
    same = [i == o for i, o in zip(ins, outs)]
    if all(same):
        return 1
    if any(same):
        return 5
    if len(set(outs)) == 1:
        return 2 if len(set(ins)) == 1 else 3
    return 4 if len(set(ins)) == 1 else 5


def colour_cell_links(inp, out):
    """Which colour-to-size links hold for this one pair (a set of names)."""
    counts = Counter(c for row in inp for c in row)
    bg = background_colour(inp)
    if bg is None:
        bg = counts.most_common(1)[0][0]
    cin, cout = colours_in(inp), colours_in(out)
    kept = {c for c in cin & cout if c != bg}
    dropped = {c for c in cin - cout if c != bg}
    non_bg = {c: n for c, n in counts.items() if c != bg}
    h, w = grid_size(out)
    area = h * w

    links = set()
    if kept and area == sum(counts[c] for c in kept):
        links.add("area_eq_kept_cells")
    if dropped and area == sum(counts[c] for c in dropped):
        links.add("area_eq_dropped_cells")
    if non_bg and area == sum(non_bg.values()):
        links.add("area_eq_nonbg_cells")
    if non_bg and area == min(non_bg.values()):
        links.add("area_eq_rarest_cells")
    if non_bg and area == max(non_bg.values()):
        links.add("area_eq_commonest_cells")
    if len(cin) in (h, w):
        links.add("side_eq_n_colours")
    if kept and len(kept) in (h, w):
        links.add("side_eq_n_kept_colours")
    if dropped and len(dropped) in (h, w):
        links.add("side_eq_n_dropped_colours")
    return links


def colour_cell_group(train, test):
    """Links that hold in EVERY training pair, and whether they hold on the test pair."""
    per_pair = [colour_cell_links(p["input"], p["output"]) for p in train]
    common = set.intersection(*per_pair) if per_pair else set()
    test_links = [colour_cell_links(t["input"], t["output"]) for t in test]
    if not common:
        return "none", "n/a"
    if not test_links:
        return "|".join(sorted(common)), "n/a"
    holds = common & set.intersection(*test_links)
    return "|".join(sorted(common)), ("holds" if holds else "breaks")


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
                "consistent": None, "colour_test": "n/a", "size_rule": 0,
                "palette_rule": "none", "palette_detail": "", "palette_test": "n/a",
                "test_vs_train": "n/a",
                "cell_links": "n/a", "cell_links_test": "n/a",
                "bg_group": "none", "bg_reason": "no_training_examples",
                "bg_test": "n/a", "per_pair_colours": "", "per_pair_bg": ""}

    col = [colour_change(p["input"], p["output"]) for p in train]
    bg = [background_change(p["input"], p["output"]) for p in train]
    col_test = [colour_change(t["input"], t["output"])["kind"] for t in test]
    bg_test = [background_change(t["input"], t["output"])["kind"] for t in test]

    cg, c_reason = colour_group(col)
    bgg, b_reason = background_group(bg)

    p_rule, p_detail, p_test = palette_check(train, test)
    t_vs_train = test_input_vs_training(train, test)

    rule = size_rule(train)
    if rule in (4, 5):
        links, links_test = colour_cell_group(train, test)
    else:
        links, links_test = "n/a", "n/a"   # sizes step already decides rules 1-3

    return {
        "colour_group": cg,
        "colour_reason": c_reason,
        "consistent": change_is_consistent(cg, col),
        "colour_test": test_check(cg, col_test),
        "palette_rule": p_rule,
        "palette_detail": p_detail,
        "palette_test": p_test,
        "test_vs_train": t_vs_train,
        "size_rule": rule,
        "cell_links": links,
        "cell_links_test": links_test,
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
    print_palette_rules(results)
    print_cell_links(results)


def print_palette_rules(results):
    print("\nSection 1b: palette rule (one rule that explains the output palette in every training pair)")
    header = f"{'Palette rule':<14}{'Tasks':>7}{'Holds':>8}{'Breaks':>8}{'n/a':>6}{'Holds (when checked)':>24}"
    print(header)
    print("-" * len(header))
    for rule in ["same", "fixed_output", "add_fixed", "remove_fixed", "swap_fixed",
                 "remove_role", "subset", "superset", "other"]:
        rows = [r for r in results if r["palette_rule"] == rule]
        if not rows:
            continue
        h = sum(1 for r in rows if r["palette_test"] == "holds")
        b = sum(1 for r in rows if r["palette_test"] == "breaks")
        n = len(rows) - h - b
        rate = f"{h / (h + b):.1%}" if (h + b) else "-"
        print(f"{rule:<14}{len(rows):>7}{h:>8}{b:>8}{n:>6}{rate:>24}")
    counts = Counter(r["test_vs_train"] for r in results)
    print("  Test input palette vs training input palettes:", dict(counts))
    breaks = [r for r in results if r["palette_test"] == "breaks"]
    print(f"  Palette rule breaks on the test pair in {len(breaks)} tasks; "
          f"{sum(1 for r in breaks if r['test_vs_train'] == 'has_new_colours')} of them have new colours in the test input")


def print_cell_links(results):
    rows = [r for r in results if r["size_rule"] in (4, 5)]
    print(f"\nSection 3: colours and cells, size rules 4 and 5 ({len(rows)} tasks)")
    with_link = [r for r in rows if r["cell_links"] != "none"]
    holds = sum(1 for r in with_link if r["cell_links_test"] == "holds")
    breaks = sum(1 for r in with_link if r["cell_links_test"] == "breaks")
    print(f"  tasks with a colour-to-size link in every training pair: {len(with_link)}/{len(rows)}")
    print(f"  of those, the link also holds on the test pair: {holds}, breaks: {breaks}")
    counts = Counter(link for r in with_link for link in r["cell_links"].split("|"))
    for link, n in counts.most_common():
        print(f"    {link:<28}{n:>5}")


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
               "palette_rule", "palette_detail", "palette_test", "test_vs_train",
               "size_rule", "cell_links", "cell_links_test",
               "per_pair_colours", "per_pair_bg"]
    RESULTS_DIR.mkdir(exist_ok=True)
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
