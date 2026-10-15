"""How reproducible is a single run of the grid?

Reproducing one fold three times gave acc 0.125, 0.125, 0.531 from identical
code, flags and seed, while the per-epoch losses matched the published run to
four significant figures (experiments/schedule/README.md). Training reproduces;
the DECODE does not. This script measures how far that reaches by comparing an
independent replicate against the published grid, run for run.

Two questions, and they are not the same question:

  1. How far does an INDIVIDUAL run move? This is what a reader reproducing one
     number will see, and it is reported as the distribution of |replicate -
     published| over matched (arm, site, seed) triples.
  2. Does the PAPER'S CLAIM move? The headline contrast is rigid vs learnable on
     F1@50, paired per held-out site. A large per-run spread that cancels in the
     pairing is a different situation from one that flips the conclusion, and
     only the second threatens the result. Both are computed below.

    python src/experiments/compare_replicate.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]

METRICS = ["acc", "macro_f1", "f1@50", "edit", "onset_mae_frames", "violation"]


def load(path: Path) -> dict[tuple, dict]:
    """{(mode, temporal, held_out, seed): row}, last row wins."""
    out = {}
    if not path.exists():
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r.get("mode"), r.get("temporal", "none"), r.get("held_out"),
               r.get("seed"))
        out[key] = r
    return out


def paired_t(diffs: list[float]) -> tuple[float, float]:
    """Paired t statistic and two-sided p from the t distribution.

    Same test the paper reports, recomputed here so the replicate's contrast is
    compared like with like rather than against a differently computed number.
    """
    import math
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m, sd = st.mean(diffs), st.stdev(diffs)
    if sd == 0:
        return float("inf"), 0.0
    t = m / (sd / math.sqrt(n))
    # two-sided p from the t distribution with n-1 df, not the normal tail
    p = 2 * student_t.sf(abs(t), n - 1)
    return t, p


def contrast(rows: dict[tuple, dict], key: str = "f1@50"):
    """mean per-site (hard - gated), averaging seeds within site as the paper does."""
    per_site: dict[str, dict[str, list[float]]] = {}
    for (mode, temporal, site, seed), r in rows.items():
        if temporal != "none" or mode not in ("hard", "gated"):
            continue
        v = r.get(key)
        if v is None or v != v:
            continue
        per_site.setdefault(site, {}).setdefault(mode, []).append(v)
    diffs = []
    for site, d in sorted(per_site.items()):
        if "hard" in d and "gated" in d:
            diffs.append(st.mean(d["hard"]) - st.mean(d["gated"]))
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published", default="experiments/loso/results.jsonl")
    ap.add_argument("--replicate", default="experiments/stability/results.jsonl")
    args = ap.parse_args()

    pub = load(REPO / args.published)
    rep = load(REPO / args.replicate)
    shared = sorted(set(pub) & set(rep))
    if not shared:
        print("no matched runs yet — is the replicate still running?")
        return 1

    print(f"matched runs: {len(shared)} "
          f"(published {len(pub)}, replicate {len(rep)})\n")

    print("=== 1. how far does an individual run move? ===")
    print(f"{'metric':>18}  {'mean |diff|':>11} {'median':>8} {'max':>8} "
          f"{'exact':>7}")
    for m in METRICS:
        ds = []
        exact = 0
        for k in shared:
            a, b = pub[k].get(m), rep[k].get(m)
            if a is None or b is None or a != a or b != b:
                continue
            ds.append(abs(a - b))
            if a == b:
                exact += 1
        if not ds:
            continue
        print(f"{m:>18}  {st.mean(ds):>11.4f} {st.median(ds):>8.4f} "
              f"{max(ds):>8.4f} {exact:>4}/{len(ds)}")

    print("\n  runs that moved most on F1@50:")
    moved = sorted(shared, key=lambda k: -abs((pub[k].get("f1@50") or 0)
                                              - (rep[k].get("f1@50") or 0)))
    for k in moved[:6]:
        mode, _, site, seed = k
        print(f"    {mode:>6} {site} seed {seed}: "
              f"acc {pub[k]['acc']:.3f} -> {rep[k]['acc']:.3f}   "
              f"F1@50 {pub[k]['f1@50']:.3f} -> {rep[k]['f1@50']:.3f}")

    print("\n=== 2. does the paper's contrast move? ===")
    print("    (hard - gated) on F1@50, seeds averaged within site, n sites)")
    for name, rows in (("published", pub), ("replicate", rep)):
        # restrict BOTH to the seeds the replicate actually ran, or the
        # comparison silently contrasts a 3-seed mean against a 1-seed one
        seeds = {k[3] for k in rep}
        sub = {k: v for k, v in rows.items() if k[3] in seeds}
        ds = contrast(sub)
        if not ds:
            print(f"  {name:>10}: no complete pairs")
            continue
        t, p = paired_t(ds)
        ahead = sum(1 for d in ds if d > 0)
        print(f"  {name:>10}: mean diff {st.mean(ds):+.4f}  "
              f"rigid ahead on {ahead}/{len(ds)} sites  t = {t:.2f}  p ~ {p:.3f}")

    print("\n  A conclusion that survives here survived independent execution.")
    print("  A per-run spread that is large while the contrast holds means the")
    print("  noise is common to both arms and cancels in the pairing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
