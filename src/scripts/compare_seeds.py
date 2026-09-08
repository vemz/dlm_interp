from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]

MODELS = [
    ("seed 0", "", "baseline_s0"),
    ("seed 1", "_s1", "baseline_s1_30k"),
    ("seed 2", "_s2", "baseline_s2_30k"),
]

HORIZONS = ("0", "1", "2", "4", "8", "16")


def read(tag, name):
    path = ROOT / ("results" + tag) / name
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def clears(lo, hi, direction=1):
    """Does the interval exclude zero on the predicted side?"""
    return (lo > 0) if direction > 0 else (hi < 0)


def block(title, note, rows, columns):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if note:
        print(note + "\n")
    widths = [max(len(str(r[i])) for r in [columns] + rows) for i in range(len(columns))]
    line = "  ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def verdict(hits, total, claim):
    if total == 0:
        print(f"\n  no models collected yet for: {claim}")
        return
    word = "reappears in" if hits == total else "reappears in only"
    print(f"\n  {hits} of {total} models clear zero. The claim {word} "
          f"{hits}/{total} independently trained models.")
    if hits < total:
        print("  Report the failures too. A claim that holds in two of three is a "
              "claim\n  about two of three, and that is still worth writing down.")


# ---------------------------------------------------------------- dissociation

def dissociation():
    rows, hits, total = [], 0, 0
    for label, tag, _ in MODELS:
        data = read(tag, "part1_dissociation.csv")
        if data is None:
            rows.append([label, "-", "-", "-", "not collected"])
            continue
        for rung in ("50", "none"):
            hit = next((d for d in data if d["pca_dims"] == rung), None)
            if hit is None:
                continue
            lo, hi = float(hit["boot_lo"]), float(hit["boot_hi"])
            ok = clears(lo, hi)
            rows.append([f"{label}, {rung} dims", f"{float(hit['difference']):+.4f}",
                         f"({lo:+.4f}, {hi:+.4f})", hit["p_positive"],
                         "clears" if ok else "unresolved"])
            if rung == "50":
                total += 1
                hits += int(ok)
    block("CLAIM 1 - the dissociation: readiness increment minus predictability increment",
          "Predicted sign: positive. The residual stream should add more to readiness\n"
          "than to predictability. Two rungs shown; the 50-dim rung is the one counted.",
          rows, ["model", "difference", "95% CI", "P(>0)", ""])
    verdict(hits, total, "dissociation")


# -------------------------------------------------------------------- horizon

def horizon():
    rows = []
    per_h = {h: [0, 0] for h in HORIZONS}
    for label, tag, _ in MODELS:
        data = read(tag, "rq1_horizon_bootstrap.csv")
        if data is None:
            rows.append([label, "not collected", "", "", ""])
            continue
        for h in HORIZONS:
            hit = next((d for d in data
                        if d["horizon"] == h and d["features"] == "all layers"), None)
            if hit is None:
                continue
            lo, hi = float(hit["boot_lo"]), float(hit["boot_hi"])
            ok = clears(lo, hi)
            per_h[h][1] += 1
            per_h[h][0] += int(ok)
            rows.append([label, f"h = {h}", f"{float(hit['delta_mean']):+.4f}",
                         f"({lo:+.4f}, {hi:+.4f})", "clears" if ok else "unresolved"])
    block("CLAIM 2 - the horizon: residual stream over the decoder's own signals",
          "Predicted sign: positive, decaying with h. The published horizon is four\n"
          "steps on seed 0. The question is where it lands on the other two models.",
          rows, ["model", "horizon", "delta", "95% CI", ""])
    print("\n  per-horizon replication:")
    for h in HORIZONS:
        got, tot = per_h[h]
        if tot:
            print(f"    h = {h:>2}: {got} of {tot} clear zero")
    deepest = [h for h in HORIZONS if per_h[h][1] and per_h[h][0] == per_h[h][1]]
    if deepest:
        print(f"\n  Deepest horizon clearing zero in every collected model: "
              f"h = {max(deepest, key=int)}.")
        print("  That, not seed 0's number, is what goes in the paper.")


# ---------------------------------------------------------------- depth (null)

def depth():
    rows, hits, total = [], 0, 0
    for label, tag, _ in MODELS:
        data = read(tag, "depth_part1_readiness.csv")
        if data is None:
            rows.append([label, "-", "-", "not collected"])
            continue
        hit = next((d for d in data if d["pca_dims"] == "none"), None)
        if hit is None:
            continue
        lo, hi = float(hit["boot_lo"]), float(hit["boot_hi"])
        resolved = clears(lo, hi) or clears(lo, hi, -1)
        rows.append([label, f"{float(hit['gap']):+.4f}", f"({lo:+.4f}, {hi:+.4f})",
                     "separates" if resolved else "flat"])
        total += 1
        hits += int(not resolved)
    block("CLAIM 3 - depth, uncompressed: shallow layer minus deepest layer",
          "Predicted: no separation. This is an accepted null, so replication here\n"
          "means the interval CONTAINS zero in each model, not that it clears it.",
          rows, ["model", "gap", "95% CI", ""])
    if total:
        print(f"\n  {hits} of {total} models show no separation at full dimensionality.")
        print("  A null that reproduces across three models is a much better null than")
        print("  one measured once, but it is still a null: report the interval, never")
        print("  the word 'flat' on its own.")


# ------------------------------------------------------------- training curve

def curve():
    rows = []
    for label, tag, _ in MODELS:
        data = read(tag, "checkpoint_curve_bootstrap.csv")
        if data is None:
            rows.append([label, "not collected", "", ""])
            continue
        for dims in ("50", "none"):
            early = next((d for d in data
                          if d["dims"] == dims and d["step"] == "5000"), None)
            late = next((d for d in data
                         if d["dims"] == dims and d["step"] == "30000"), None)
            if early is None or late is None:
                continue
            rows.append([f"{label}, {dims} dims",
                         f"{float(early['advantage']):+.4f}",
                         f"{float(late['advantage']):+.4f}",
                         f"{float(early['relative']):.2f} -> {float(late['relative']):.2f}"])
    block("CLAIM 4 - across training (optional; needs the intermediate checkpoints)",
          "Predicted: the absolute advantage roughly constant, the relative advantage\n"
          "falling because the output baseline catches up.",
          rows, ["model", "adv @ 5k", "adv @ 30k", "relative 5k -> 30k"])


def main():
    have = [label for label, tag, _ in MODELS
            if (ROOT / ("results" + tag)).exists()]
    print(f"models with a results directory: {', '.join(have) if have else 'none'}")
    print("\nNothing below is pooled across models and no standard error is taken over")
    print("them. Three is a replication count, not a sample.")
    dissociation()
    horizon()
    depth()
    curve()
    print("\n" + "=" * 78)
    print("How to write this up")
    print("=" * 78)
    print("  Good:  'The dissociation appears in three independently trained models,")
    print("          with a window-clustered interval clearing zero in each.'")
    print("  Bad:   'The dissociation is +0.021 +/- 0.003 across three seeds.'")
    print("  The second sentence is the sd/sqrt(8) error with a different denominator.")


if __name__ == "__main__":
    main()
