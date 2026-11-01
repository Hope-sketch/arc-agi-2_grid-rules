"""
Whole-grid transformation library (used by grid_transformations.py)

The old step-6 idea: try a fixed list of whole-grid moves (flips, rotations,
mirror joins, tiling, scaling, crop, halves), optionally followed by ONE
consistent colour map learned from the training pairs. A move counts as
FOUND when it reproduces the output of EVERY training pair.

It is kept as one probe among others, because it does solve a few tasks.
"""

import numpy as np


def background(g):
    vals, counts = np.unique(g, return_counts=True)
    return vals[np.argmax(counts)]


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


def find_whole_grid(train, tests):
    """train: list of (input, output) arrays. tests: list of (input, output or None).

    Returns a dict: category (found_exact / found_with_colour_map / partial / none),
    chosen, test_correct ("" when no public test answer or nothing found).
    """
    library = fixed_transforms()
    library.update(parameterised_transforms(*train[0]))

    exact, colour_map_found, maps, best = [], [], {}, (0, 0.0, "")
    for name, fn in library.items():
        preds = [safe_apply(fn, i) for i, _ in train]
        per_pair = [p is not None and p.shape == o.shape and np.array_equal(p, o)
                    for p, (_, o) in zip(preds, train)]
        acc = float(np.mean([cell_accuracy(p, o) for p, (_, o) in zip(preds, train)]))
        best = max(best, (sum(per_pair), acc, name), key=lambda s: (s[0], s[1]))
        if all(per_pair):
            exact.append(name)
        else:
            m = learn_colour_map(preds, [o for _, o in train])
            if m is not None:
                colour_map_found.append(name)
                maps[name] = m

    chosen, kind = "", "none"
    if exact:
        chosen, kind = exact[0], "exact"
    elif colour_map_found:
        chosen, kind = colour_map_found[0], "colour_map"

    test_correct = ""
    if chosen:
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

    if kind == "exact":
        category = "found_exact"
    elif kind == "colour_map":
        category = "found_with_colour_map"
    elif best[0] > 0:
        category = "partial"
    else:
        category = "none"
    return {"category": category, "chosen": chosen, "test_correct": test_correct}