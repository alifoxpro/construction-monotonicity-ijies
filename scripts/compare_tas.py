"""Do the standard temporal-action-segmentation models beat the domain prior?

The paper's claim is that a construction-monotonicity prior is worth far more
than sequence modelling as such. It cites MS-TCN and ASFormer to position the
work; this compares against them directly, on the same corpus, backbone, folds
and protocol.

Three questions, in the order that decides what may be claimed:

  1. Do they clear the IMAGE-FREE CONTROL? A predictor that splits each sequence
     into equal monotone runs without seeing a pixel scores F1@50 = 0.220. An arm
     below that is not evidence of anything, however sophisticated. The LSTM
     baseline reaches 0.128 and does not clear it.

  2. Is their low order-violation rate real, or COLLAPSE? A model predicting one
     stage for a whole sequence has zero violations and a meaningless score. The
     violation rate must therefore always be read next to the number of distinct
     stages actually predicted -- otherwise a degenerate model looks disciplined.

  3. Do they beat the rigid prior, paired per held-out site, on F1@50?

    python src/experiments/compare_tas.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]

FLOOR = {"acc": 0.374, "macro_f1": 0.275, "edit": 0.815, "f1@50": 0.220}
ANNOTATION_NOISE = 3.69


def load(paths) -> dict:
    """{arm: {site: [rows]}} where arm folds the temporal axis in."""
    out: dict[str, dict[str, list]] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("temporal", "none")
            arm = t if t != "none" else r["mode"]
            out.setdefault(arm, {}).setdefault(r["held_out"], []).append(r)
    return out


def arm_mean(rows: dict, key: str) -> float:
    per_site = [st.mean([r[key] for r in rs if r.get(key) is not None])
                for rs in rows.values()
                if any(r.get(key) is not None for r in rs)]
    return st.mean(per_site) if per_site else float("nan")


def stages_predicted(rows: dict) -> float:
    """Mean number of DISTINCT stages the arm actually predicts per sequence.

    Recovered from eval_stage_counts where available; otherwise from the
    per-frame confusion the row records. An arm near 1.0 has collapsed, and its
    order-violation rate carries no information.
    """
    vals = []
    for rs in rows.values():
        for r in rs:
            n = r.get("n_pred_stages")
            if n is not None:
                vals.append(float(n))
    return st.mean(vals) if vals else float("nan")


def paired(a_rows: dict, b_rows: dict, key: str):
    """b - a, paired per site, seeds averaged within site."""
    sites = sorted(set(a_rows) & set(b_rows))
    diffs = []
    for s in sites:
        va = st.mean([r[key] for r in a_rows[s] if r.get(key) is not None])
        vb = st.mean([r[key] for r in b_rows[s] if r.get(key) is not None])
        diffs.append(vb - va)
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan"), len(diffs), 0
    m, sd = st.mean(diffs), st.stdev(diffs)
    t = m / (sd / math.sqrt(len(diffs))) if sd else float("inf")
    # two-sided p from the t distribution with n-1 df -- NOT the normal tail,
    # which at n = 13 understates p by up to a factor of three
    p = 2 * student_t.sf(abs(t), len(diffs) - 1) if sd else 0.0
    return m, t, p, len(diffs), sum(1 for d in diffs if d > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published", default="experiments/loso/results.jsonl")
    ap.add_argument("--tas", default="experiments/tas/results.jsonl")
    args = ap.parse_args()

    arms = load([REPO / args.published, REPO / args.tas])
    order = [a for a in ["none", "free", "lstm", "mstcn", "asformer", "hard", "gated"]
             if a in arms]
    print("arms present:", order)
    print("runs per arm:",
          {a: sum(len(v) for v in arms[a].values()) for a in order}, "\n")

    print("=== 1. per-arm means, against the image-free control ===")
    print(f"{'arm':>10} {'acc':>7} {'macroF1':>8} {'F1@50':>7} {'edit':>7} "
          f"{'viol':>7}   clears F1@50 floor?")
    for a in order:
        f150 = arm_mean(arms[a], "f1@50")
        mark = "YES" if f150 > FLOOR["f1@50"] else "no"
        print(f"{a:>10} {arm_mean(arms[a], 'acc'):>7.3f} "
              f"{arm_mean(arms[a], 'macro_f1'):>8.3f} {f150:>7.3f} "
              f"{arm_mean(arms[a], 'edit'):>7.3f} "
              f"{arm_mean(arms[a], 'violation'):>7.3f}   {mark}")
    print(f"{'CONTROL':>10} {FLOOR['acc']:>7.3f} {FLOOR['macro_f1']:>8.3f} "
          f"{FLOOR['f1@50']:>7.3f} {FLOOR['edit']:>7.3f} {'-':>7}   (image-free)")

    print("\n=== 2. is a low violation rate discipline, or collapse? ===")
    print("    a model predicting ONE stage has zero violations and says nothing")
    for a in order:
        v = arm_mean(arms[a], "violation")
        f150 = arm_mean(arms[a], "f1@50")
        ed = arm_mean(arms[a], "edit")
        note = ""
        # Collapse shows up as low violations WITH a low edit score: a model
        # emitting one long run has nothing to violate and nothing to segment.
        # Edit is the right companion here rather than F1@50, and the arms whose
        # violations are ~0 BY CONSTRUCTION (hard, gated) are excluded -- their
        # zero says nothing about collapse either way.
        if a not in ("hard", "gated") and v < 0.05 and ed < arm_mean(arms["none"], "edit") * 1.2:
            note = "  <-- low violations AND no segmentation: suspect collapse"
        print(f"{a:>10}  violations {v:.3f}   F1@50 {f150:.3f}   edit {ed:.3f}{note}")

    print("\n=== 3. paired per held-out site, on F1@50 ===")
    if "hard" in arms:
        for a in order:
            if a == "hard":
                continue
            m, t, p, n, w = paired(arms[a], arms["hard"], "f1@50")
            print(f"  {a:>10} -> hard : {m:+.3f}  t={t:+5.2f}  p={p:.3f}  "
                  f"rigid ahead on {w}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
