"""Does the paper's decomposition survive a stronger backbone?

The published grid ran on a fine-tuned ResNet18 and concluded that the
construction-monotonicity prior carries the structural result while structured
decoding alone does not. The obvious objection is that this holds only because
the emissions are weak. This compares the same three-arm decomposition under
ResNet50.

Two comparisons, and only the second one is the point:

  1. Did the backbone actually get better? If ResNet50's emissions are no
     stronger than ResNet18's, the experiment tests nothing and the rest of the
     output should be ignored.
  2. Does the free -> hard gap survive? That gap IS the paper's claim. It is
     reported paired per held-out site, the same way the paper reports it.

The published grid is restricted to seed 0 throughout, because the backbone grid
ran one seed and a three-seed mean is a different estimator.

    python src/experiments/compare_backbone.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]

ARMS = ["none", "free", "hard"]
METRICS = ["acc", "macro_f1", "f1@50", "edit", "violation", "onset_mae_frames"]
FLOOR = {"acc": 0.374, "macro_f1": 0.275, "edit": 0.815, "f1@50": 0.220,
         "onset_mae_frames": 37.1}


def load(path: Path, seeds: set[int] | None = None) -> dict:
    """{(arm, site): row} for temporal=none rows only."""
    out = {}
    if not path.exists():
        raise SystemExit(f"missing {path}")
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("temporal", "none") != "none":
            continue
        if seeds is not None and r.get("seed") not in seeds:
            continue
        out.setdefault((r["mode"], r["held_out"]), []).append(r)
    return out


def mean_metric(rows: dict, arm: str, key: str) -> float:
    vals = [st.mean([r[key] for r in rs if r.get(key) is not None])
            for (a, _), rs in rows.items() if a == arm]
    return st.mean(vals) if vals else float("nan")


def paired(rows: dict, arm_a: str, arm_b: str, key: str):
    """arm_b - arm_a, paired per site. Returns (mean, t, p, n, wins)."""
    sites = sorted({s for (a, s) in rows if a == arm_a} &
                   {s for (a, s) in rows if a == arm_b})
    diffs = []
    for s in sites:
        va = st.mean([r[key] for r in rows[(arm_a, s)] if r.get(key) is not None])
        vb = st.mean([r[key] for r in rows[(arm_b, s)] if r.get(key) is not None])
        diffs.append(vb - va)
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan"), len(diffs), 0
    m, sd = st.mean(diffs), st.stdev(diffs)
    t = m / (sd / math.sqrt(len(diffs))) if sd else float("inf")
    # two-sided p from the t distribution with n-1 df, not the normal tail
    p = 2 * student_t.sf(abs(t), len(diffs) - 1) if sd else 0.0
    return m, t, p, len(diffs), sum(1 for d in diffs if d > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published", default="experiments/loso/results.jsonl")
    ap.add_argument("--backbone", default="experiments/backbone/results.jsonl")
    args = ap.parse_args()

    new = load(REPO / args.backbone)
    seeds = {s for rs in new.values() for s in (r.get("seed") for r in rs)}
    old = load(REPO / args.published, seeds=seeds)

    print(f"ResNet18 (published, seeds {sorted(seeds)}): "
          f"{sum(len(v) for v in old.values())} runs")
    print(f"ResNet50 (new):                              "
          f"{sum(len(v) for v in new.values())} runs\n")

    print("=== 1. did the backbone get better? (per-arm means) ===")
    print(f"{'metric':>18}  {'arm':>5}  {'ResNet18':>9} {'ResNet50':>9}  "
          f"{'change':>8}   floor")
    for key in METRICS:
        for arm in ARMS:
            a, b = mean_metric(old, arm, key), mean_metric(new, arm, key)
            fl = FLOOR.get(key)
            flag = ""
            if fl is not None and key != "violation":
                better = (lambda x: x > fl) if key != "onset_mae_frames" else (lambda x: x < fl)
                flag = f"  {'clears' if better(b) else 'BELOW'} {fl:g}"
            print(f"{key:>18}  {arm:>5}  {a:>9.3f} {b:>9.3f}  {b - a:>+8.3f}{flag}")
        print()

    print("=== 2. does the decomposition survive? ===")
    print("    paired per held-out site, the same test the paper reports\n")
    for key in ["f1@50", "acc", "edit"]:
        print(f"  --- {key} ---")
        for label, (a, b) in [("structure alone  (none -> free)", ("none", "free")),
                              ("the prior        (free -> hard)", ("free", "hard")),
                              ("both            (none -> hard)", ("none", "hard"))]:
            for name, rows in (("ResNet18", old), ("ResNet50", new)):
                m, t, p, n, w = paired(rows, a, b, key)
                print(f"    {label:<34} {name}  {m:+.3f}  "
                      f"t={t:+5.2f}  p={p:.3f}  wins {w}/{n}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
