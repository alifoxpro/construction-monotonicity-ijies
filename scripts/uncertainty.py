"""Uncertainty, site sensitivity and multiplicity — IJIES reviewer 2, point 5.

THE OBJECTION, verbatim: "Inference rests on only thirteen sites and a
substantial family of correlated comparisons. Specify the primary contrasts and
outcomes, distinguish exploratory tests, and report uncertainty intervals and
sensitivity to individual sites rather than relying on a collection of nominal
p-values. Account for the shared training data across leave-one-site-out folds
when interpreting uncertainty. Excluding three poorly performing sites should
remain a clearly justified sensitivity analysis, not stronger confirmatory
evidence. The manuscript appropriately avoids equating a nonsignificant gate
deficit with demonstrated inferiority; apply the same caution to claims of
equivalence and to the synthetic break-even interval."

Six separate demands, answered in order below.

(1) PRIMARY VERSUS EXPLORATORY. The study was not pre-registered, so this is a
designation made at revision time and it is labelled as such rather than
presented as a prior plan. One contrast is confirmatory: free CRF -> rigid on
F1@50, which isolates the construction prior from structured decoding and is the
claim the paper rests on. Everything else is exploratory, including every
accuracy and edit-score test, every baseline comparison and every onset test.
Holm's correction is applied across the confirmatory set, and across the
exploratory family separately so a reader can see what nominal significance is
worth there.

(2) UNCERTAINTY INTERVALS. Each contrast gets a 95% t interval over the 13 sites
AND a bootstrap interval resampling sites with replacement, which does not
assume the per-site differences are normal. At n = 13 the two can disagree, and
where they do the bootstrap is the one to read.

(3) DISTRIBUTION-FREE CONFIRMATION. An exact sign test accompanies every
contrast. It discards the magnitudes and asks only how many sites moved in the
claimed direction, so it is immune to the one-site outliers that a mean of 13
numbers is not.

(4) SENSITIVITY TO INDIVIDUAL SITES. Each site is dropped in turn and the
contrast recomputed, which gives the range the estimate takes over the 13
leave-one-out analyses and says whether any single site is carrying the result.

(5) THE SHARED TRAINING DATA. This is the sharpest part of the objection. The 13
per-site differences are NOT independent: any two folds train on 11 of the same
12 sites, a 92% overlap, so the differences are positively correlated and the
paired t-test -- which assumes independence -- reports a p that is too small. The
correlation cannot be estimated from a single grid, so we do not invent a number
for it. We report instead the TIPPING POINT: under an exchangeable correlation
rho, the variance of the mean is inflated by [1 + (n-1)rho], and we solve for the
rho at which each contrast stops being significant. A contrast that needs a large
rho to overturn is robust to the dependence; one that falls at rho = 0.02 was
never significant in any meaningful sense.

(6) THE THREE-SITE EXCLUSION. Two of the three excluded sites were chosen as the
two WORST sites on the very outcome being tested. Selecting on the outcome
inflates the contrast mechanically, so that analysis cannot be confirmatory
whatever it shows, and the script reports the size of the inflation.

    python src/experiments/uncertainty.py
    python src/experiments/uncertainty.py --boot 50000
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

from scipy.stats import binomtest
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

#: Each arm is read from ONE authoritative grid, never pooled across grids.
#: This matters: experiments/loso_softcheck holds a 4-run spot check that
#: includes two extra `gated` runs, and letting those into the published gated
#: arm moved the rigid-to-gated contrast from -0.0945 to -0.0762. An arm is
#: defined by the grid it was run in, so the mapping is explicit here.
ARM_SOURCE = {
    "none": "experiments/loso/results.jsonl",
    "free": "experiments/loso/results.jsonl",
    "hard": "experiments/loso/results.jsonl",
    "gated": "experiments/loso/results.jsonl",
    "none+lstm": "experiments/loso/results.jsonl",
    "none+mstcn": "experiments/tas/results.jsonl",
    # The ASFormer arm was re-run after the implementation was corrected against
    # the authors' released code (asformer_equivalence.py). The published arm in
    # experiments/tas is the weaker reimplementation and is kept as the record
    # of what was submitted; every contrast below uses the corrected one.
    "none+asformer": "experiments/tas_asformer_fixed/results.jsonl",
    # The fixed-soft arm was run in a later session, in a grid of its own that
    # also re-ran the rigid arm. A soft comparison must use THAT grid's rigid
    # arm, not the published one, or it mixes two sessions in one contrast.
    "soft": "experiments/loso_learnability/results.jsonl",
    "hard@learnability": "experiments/loso_learnability/results.jsonl",
}

#: one stage across every frame, so it admits no segment structure at all
DEGENERATE = {"seq0009"}

#: (arm a, arm b, metric, label) — b minus a. The FIRST entry is confirmatory.
CONTRASTS = [
    ("free", "hard", "f1@50", "CONFIRMATORY: construction prior (free CRF → rigid)"),
    ("none", "free", "f1@50", "structured decoding alone (per-frame → free CRF)"),
    ("hard", "gated", "f1@50", "learnability (rigid → gated)"),
    ("hard@learnability", "soft", "f1@50", "penalty strength (rigid → fixed soft)"),
    ("none", "hard", "f1@50", "per-frame → rigid (the abstract's comparison)"),
    ("none+lstm", "hard", "f1@50", "temporal model → rigid"),
    ("none+mstcn", "hard", "f1@50", "MS-TCN → rigid"),
    ("none+asformer", "hard", "f1@50", "ASFormer → rigid"),
    ("free", "hard", "acc", "construction prior, on accuracy"),
    ("free", "hard", "edit", "construction prior, on edit score"),
    ("hard", "gated", "acc", "learnability, on accuracy"),
    ("hard", "gated", "edit", "learnability, on edit score"),
]


def load() -> dict[str, dict[str, list[dict]]]:
    """arm -> site -> runs, each arm taken only from its authoritative grid."""
    arms: dict[str, dict[str, list[dict]]] = {}
    cache: dict[str, list[dict]] = {}
    for arm, rel in ARM_SOURCE.items():
        if rel not in cache:
            p = REPO / rel
            cache[rel] = []
            if p.exists():
                for line in p.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cache[rel].append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        want = arm.split("@")[0]
        for r in cache[rel]:
            t = r.get("temporal", "none")
            name = r["mode"] if t == "none" else f"{r['mode']}+{t}"
            if name == want:
                arms.setdefault(arm, {}).setdefault(r["held_out"], []).append(r)
    return arms


def site_diffs(arms, a: str, b: str, key: str,
               exclude: set[str] = frozenset()) -> dict[str, float]:
    """site -> (b minus a), seeds averaged within site first (Section 6.2)."""
    out = {}
    if a not in arms or b not in arms:
        return out
    for s in sorted(set(arms[a]) & set(arms[b])):
        if s in exclude:
            continue
        va = [r[key] for r in arms[a][s] if r.get(key) is not None]
        vb = [r[key] for r in arms[b][s] if r.get(key) is not None]
        if va and vb:
            out[s] = st.mean(vb) - st.mean(va)
    return out


# --------------------------------------------------------------------------- #
#  the five quantities the reviewer asks for
# --------------------------------------------------------------------------- #
def t_test(d: list[float]) -> tuple[float, float, float, float, float]:
    """(mean, t, p, ci_lo, ci_hi) — paired t over sites, 95% interval."""
    n = len(d)
    m, sd = st.mean(d), st.stdev(d)
    se = sd / math.sqrt(n)
    t = m / se if se else float("inf")
    p = 2 * student_t.sf(abs(t), n - 1)
    half = student_t.ppf(0.975, n - 1) * se
    return m, t, p, m - half, m + half


def bootstrap_ci(d: list[float], draws: int, rng: random.Random
                 ) -> tuple[float, float]:
    """Percentile interval, resampling SITES with replacement.

    Does not assume the per-site differences are normal, which at n = 13 is an
    assumption worth not making.
    """
    n = len(d)
    means = sorted(st.mean(rng.choices(d, k=n)) for _ in range(draws))
    lo = means[int(0.025 * draws)]
    hi = means[min(draws - 1, int(0.975 * draws))]
    return lo, hi


def sign_test(d: list[float]) -> tuple[int, int, float]:
    """(wins, non-ties, exact two-sided p). Magnitudes discarded entirely."""
    wins = sum(1 for x in d if x > 0)
    ties = sum(1 for x in d if x == 0)
    n = len(d) - ties
    if n == 0:
        return wins, 0, 1.0
    return wins, n, float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)


def jackknife(diffs: dict[str, float]) -> tuple[float, float, str, str, int]:
    """Drop each site in turn. (min mean, max mean, site at min, site at max,
    how many of the leave-one-out analyses stay significant at .05)."""
    sites = sorted(diffs)
    rows = []
    for s in sites:
        rest = [diffs[k] for k in sites if k != s]
        m, _, p, _, _ = t_test(rest)
        rows.append((m, p, s))
    rows.sort()
    sig = sum(1 for m, p, _ in rows if p < 0.05)
    return rows[0][0], rows[-1][0], rows[0][2], rows[-1][2], sig


def rho_tipping_point(d: list[float]) -> float:
    """The exchangeable correlation at which this contrast stops being significant.

    With an exchangeable correlation rho among the n per-site differences,
    Var(mean) = sigma^2/n * [1 + (n-1)rho], so the test statistic shrinks by
    sqrt(1 + (n-1)rho). We solve for the rho that puts p at exactly .05, by
    bisection on a monotone function. nan if the contrast is not significant at
    rho = 0; 1.0 if even perfect correlation leaves it significant (it cannot).
    """
    n = len(d)
    m, t0, p0, _, _ = t_test(d)
    if p0 >= 0.05:
        return float("nan")

    def p_at(rho: float) -> float:
        t = t0 / math.sqrt(1.0 + (n - 1) * rho)
        return 2 * student_t.sf(abs(t), n - 1)

    lo, hi = 0.0, 1.0
    if p_at(hi) < 0.05:
        return 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if p_at(mid) < 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, family-wise over `pairs`."""
    order = sorted(pairs, key=lambda kv: kv[1])
    k = len(order)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (k - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[name] = running
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=20000,
                    help="bootstrap resamples (sites, with replacement)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="experiments/uncertainty.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arms = load()
    print(f"arms available: {', '.join(sorted(arms))}\n")

    rows = []
    for a, b, key, label in CONTRASTS:
        d = site_diffs(arms, a, b, key)
        if len(d) < 3:
            print(f"skipped (needs both arms): {label} [{key}]")
            continue
        vals = [d[s] for s in sorted(d)]
        m, t, p, lo, hi = t_test(vals)
        blo, bhi = bootstrap_ci(vals, args.boot, rng)
        wins, nt, sp = sign_test(vals)
        jmin, jmax, smin, smax, jsig = jackknife(d)
        rho = rho_tipping_point(vals)
        rows.append({"a": a, "b": b, "metric": key, "label": label,
                     "n_sites": len(d), "mean": m, "t": t, "p": p,
                     "ci_lo": lo, "ci_hi": hi, "boot_lo": blo, "boot_hi": bhi,
                     "wins": wins, "n_nontied": nt, "sign_p": sp,
                     "jack_min": jmin, "jack_max": jmax,
                     "jack_min_site": smin, "jack_max_site": smax,
                     "jack_significant": jsig, "rho_tipping": rho,
                     "per_site": d})

    # ---- (1) + (2) + (3) the table -----------------------------------------
    print("CONTRASTS — paired over held-out sites, seeds averaged within site")
    print("CI is the 95% t interval; boot is the percentile interval from "
          f"{args.boot} resamples of SITES.\n")
    hdr = (f"{'contrast':46s} {'n':>3s} {'mean':>8s} {'95% t CI':>18s} "
           f"{'95% boot CI':>18s} {'p':>7s} {'sign p':>7s} {'wins':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:46]:46s} {r['n_sites']:3d} {r['mean']:+8.4f} "
              f"[{r['ci_lo']:+7.4f},{r['ci_hi']:+7.4f}] "
              f"[{r['boot_lo']:+7.4f},{r['boot_hi']:+7.4f}] "
              f"{r['p']:7.3f} {r['sign_p']:7.3f} "
              f"{r['wins']:2d}/{r['n_nontied']:<3d}")

    # ---- (4) site sensitivity ----------------------------------------------
    print("\n\nSENSITIVITY TO INDIVIDUAL SITES — drop one site, recompute")
    hdr = (f"{'contrast':46s} {'range of the mean':>26s} "
           f"{'driven by':>11s} {'still sig':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label'][:46]:46s} "
              f"[{r['jack_min']:+8.4f} … {r['jack_max']:+8.4f}] "
              f"{r['jack_min_site']:>11s} "
              f"{r['jack_significant']:4d}/{r['n_sites']:<5d}")
    print("\n'driven by' is the site whose removal LOWERS the contrast most, "
          "i.e. the\nsite most favourable to the claim. 'still sig' counts how "
          "many of the 13\nleave-one-out analyses keep p < .05.")

    # ---- (5) the shared training data --------------------------------------
    print("\n\nTHE SHARED TRAINING DATA ACROSS FOLDS")
    print("Any two folds train on 11 of the same 12 sites (92% overlap), so the")
    print("13 per-site differences are positively correlated and the paired t")
    print("p-value is too small. The correlation is not estimable from one grid,")
    print("so instead of inventing it we report the correlation that would be")
    print("needed to remove significance: with an exchangeable rho, Var(mean) is")
    print("inflated by [1 + 12·rho].\n")
    print(f"{'contrast':46s} {'nominal p':>10s} {'rho needed to reach p = .05':>29s}")
    print("-" * 87)
    for r in rows:
        rho = r["rho_tipping"]
        txt = ("not significant at rho = 0" if rho != rho
               else f"{rho:.3f}")
        print(f"{r['label'][:46]:46s} {r['p']:10.3f} {txt:>29s}")

    # ---- (1) multiplicity ---------------------------------------------------
    conf = [rows[0]] if rows else []
    expl = rows[1:]
    print("\n\nMULTIPLICITY")
    print(f"Confirmatory set: {len(conf)} contrast. Exploratory family: "
          f"{len(expl)} contrasts in this script, and the paper reports further "
          f"\nexploratory tests over the synthetic sweep and the onset metric.")
    if expl:
        adj = holm([(r["label"], r["p"]) for r in expl])
        surviving = [(k, v) for k, v in adj.items() if v < 0.05]
        print(f"\nHolm over the exploratory family ({len(expl)} tests): "
              f"{len(surviving)} of {len(expl)} survive at .05")
        for r in expl:
            a = adj[r["label"]]
            print(f"  {r['label'][:46]:46s} p {r['p']:.3f} → Holm {a:.3f}"
                  f"{'  *' if a < 0.05 else ''}")
        for r in expl:
            r["holm_p"] = adj[r["label"]]
    if conf:
        print(f"\nThe confirmatory contrast needs no correction (family of one): "
              f"p = {conf[0]['p']:.3f}")

    # ---- (6) the outcome-dependent exclusion -------------------------------
    print("\n\nTHE THREE-SITE EXCLUSION IS NOT CONFIRMATORY")
    full = site_diffs(arms, "free", "hard", "f1@50")
    per_site_hard = {s: st.mean(r["f1@50"] for r in arms["hard"][s])
                     for s in arms["hard"]}
    low = sorted((v, s) for s, v in per_site_hard.items()
                 if s not in DEGENERATE)[:2]
    excl = DEGENERATE | {s for _, s in low}
    kept = site_diffs(arms, "free", "hard", "f1@50", excl)
    m_full, _, p_full, lo_f, hi_f = t_test([full[s] for s in sorted(full)])
    m_cut, _, p_cut, lo_c, hi_c = t_test([kept[s] for s in sorted(kept)])
    print(f"excluded: {sorted(excl)}")
    print(f"  {sorted(DEGENERATE)[0]} carries one stage across every frame, so it "
          f"admits no segment\n  structure; that exclusion is a property of the "
          f"site, decided before any test.")
    print(f"  {', '.join(sorted(s for _, s in low))} were chosen as the two "
          f"LOWEST-scoring sites on F1@50 —\n  the very outcome under test. "
          f"Selecting on the outcome inflates the contrast\n  mechanically, so "
          f"this row is a sensitivity analysis and cannot be evidence.")
    print(f"\n  all 13 sites : {m_full:+.4f} [{lo_f:+.4f}, {hi_f:+.4f}], "
          f"p = {p_full:.3f}")
    print(f"  10 sites     : {m_cut:+.4f} [{lo_c:+.4f}, {hi_c:+.4f}], "
          f"p = {p_cut:.3f}")
    print(f"  inflation    : {m_cut - m_full:+.4f} "
          f"({100 * (m_cut / m_full - 1):+.0f}% of the full-corpus estimate)")

    # ---- (7) caution about equivalence -------------------------------------
    print("\n\nWHAT A NON-SIGNIFICANT CONTRAST DOES NOT ESTABLISH")
    gate = next((r for r in rows
                 if (r["a"], r["b"], r["metric"]) == ("hard", "gated", "f1@50")),
                None)
    if gate:
        worst = min(abs(gate["boot_lo"]), abs(gate["boot_hi"]))
        big = max(abs(gate["boot_lo"]), abs(gate["boot_hi"]))
        print(f"rigid → gated on F1@50: {gate['mean']:+.4f}, p = {gate['p']:.3f}, "
              f"bootstrap interval [{gate['boot_lo']:+.4f}, {gate['boot_hi']:+.4f}].")
        print(f"The paper is right not to read this as the gate being worse. It is")
        print(f"equally wrong to read it as the two being equivalent: the interval")
        print(f"still admits a difference as large as {big:.4f} in F1@50, which is")
        print(f"{big / 0.313 * 100:.0f}% of the rigid arm's own score, and excludes")
        print(f"nothing smaller than {worst:.4f}. Equivalence would require an")
        print(f"interval inside a margin stated in advance; no such margin was set,")
        print(f"so neither direction is established.")

    out = Path(REPO / args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"bootstrap_draws": args.boot, "seed": args.seed,
         "contrasts": [{k: v for k, v in r.items() if k != "per_site"}
                       for r in rows],
         "per_site_differences": {r["label"]: r["per_site"] for r in rows},
         "exclusion_sensitivity": {
             "excluded": sorted(excl),
             "outcome_dependent": sorted(s for _, s in low),
             "all_sites": {"mean": m_full, "p": p_full,
                           "ci": [lo_f, hi_f], "n": len(full)},
             "reduced": {"mean": m_cut, "p": p_cut,
                         "ci": [lo_c, hi_c], "n": len(kept)}}},
        indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
