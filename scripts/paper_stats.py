"""Every paired test the paper reports, from the logged results, in one place.

Table I and the re-work sweep's per-seed tests were first computed ad hoc, with
a normal tail for p. At n = 13 sites (df = 12) or n = 5 seeds (df = 4) that
understates p by up to a factor of three, and it moved the headline accuracy
contrast from p = .064 to a reported .041. This script is now the only source
for those numbers: two-sided p from the t distribution with n - 1 df.

    python src/experiments/paper_stats.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]
DEGENERATE = {"seq0009"}          # one stage across all frames (Section IX)


def ttest(diffs):
    n = len(diffs)
    m, sd = st.mean(diffs), st.stdev(diffs)
    t = m / (sd / math.sqrt(n))
    return m, t, 2 * student_t.sf(abs(t), n - 1), n, sum(d > 0 for d in diffs)


def fmt(label, res):
    m, t, p, n, w = res
    ps = "<.001" if p < .001 else f"{p:.3f}"
    return f"  {label:<38} {m:+.4f}  t={t:+.2f}  df={n-1:<2} p={ps:<6}  wins {w}/{n}"


def load_loso():
    arms: dict[str, dict[str, list]] = {}
    for p in ["experiments/loso/results.jsonl", "experiments/tas/results.jsonl"]:
        for line in open(REPO / p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("temporal", "none")
            arm = t if t != "none" else r["mode"]
            arms.setdefault(arm, {}).setdefault(r["held_out"], []).append(r)
    return arms


def site_diffs(arms, a, b, key, exclude=()):
    out = []
    for s in sorted(set(arms[a]) & set(arms[b])):
        if s in exclude:
            continue
        va = [r[key] for r in arms[a][s] if r.get(key) is not None]
        vb = [r[key] for r in arms[b][s] if r.get(key) is not None]
        if va and vb:
            out.append(st.mean(vb) - st.mean(va))
    return out


def main():
    arms = load_loso()
    print("=== Table I: paired per held-out site, seeds averaged within site ===")
    rows = [("none", "hard", "per-frame -> hard"), ("free", "hard", "free CRF -> hard (headline)"),
            ("none", "free", "per-frame -> free CRF"), ("none", "lstm", "per-frame -> LSTM"),
            ("lstm", "hard", "LSTM -> hard"), ("hard", "gated", "hard -> gated")]
    for a, b, lab in rows:
        for key in ("acc", "edit", "f1@50"):
            print(fmt(f"{lab} [{key}]", ttest(site_diffs(arms, a, b, key))))
    for a in ("mstcn", "asformer"):
        print(fmt(f"{a} -> hard [f1@50]", ttest(site_diffs(arms, a, "hard", "f1@50"))))

    # "the three degenerate sites of Section IX": seq0009 plus the two lowest
    # sites on every arm, identified here rather than hard-coded
    per_site = {s: st.mean(r["f1@50"] for r in arms["hard"][s]) for s in arms["hard"]}
    low = sorted((v, s) for s, v in per_site.items() if s not in DEGENERATE)[:2]
    excl = DEGENERATE | {s for _, s in low}
    print(f"\n=== headline without degenerate sites {sorted(excl)} ===")
    for key in ("acc", "edit", "f1@50"):
        print(fmt(f"free CRF -> hard [{key}]", ttest(site_diffs(arms, "free", "hard", key, excl))))

    print("\n=== re-work sweep: gated - hard, paired per seed ===")
    d = json.load(open(REPO / "experiments/rework_sweep/results.json"))["rows"]
    for rate in sorted({r["rework_rate"] for r in d}):
        by = {(r["mode"], r["seed"]): r for r in d if r["rework_rate"] == rate}
        seeds = sorted(s for (m, s) in by if m == "hard" and ("gated", s) in by)
        gv = st.mean(by[("hard", s)]["gold_violation"] for s in seeds)
        for key in ("acc", "edit"):
            diffs = [by[("gated", s)][key] - by[("hard", s)][key] for s in seeds]
            print(fmt(f"events={rate} viol={gv:.4f} [{key}]", ttest(diffs)))

    print("\n=== concentration: gated - hard, paired per seed ===")
    c = json.load(open(REPO / "experiments/rework_concentration/results.json"))["rows"]
    for regime in sorted({r["regime"] for r in c}):
        for ev in sorted({r["n_events"] for r in c if r["regime"] == regime}):
            by = {(r["mode"], r["seed"]): r for r in c
                  if r["regime"] == regime and r["n_events"] == ev}
            seeds = sorted(s for (m, s) in by if m == "hard" and ("gated", s) in by)
            if len(seeds) < 2:
                continue
            gv = st.mean(by[("hard", s)]["gold_violation"] for s in seeds)
            diffs = [by[("gated", s)]["acc"] - by[("hard", s)]["acc"] for s in seeds]
            print(fmt(f"{regime} events={ev} viol={gv:.4f} [acc]", ttest(diffs)))


if __name__ == "__main__":
    main()
