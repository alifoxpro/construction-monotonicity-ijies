"""The corrected ASFormer arm against the published one — IJIES reviewer 2, point 4.

The ASFormer module was rewritten to match the authors' released implementation
(see src/experiments/asformer_equivalence.py and the note at the head of the
ASFormer section in src/models/tas_encoders.py). This script compares the
corrected arm against the one the submitted paper reports, and prints the
specific numbers in the manuscript that the correction changes.

IT REFUSES TO PRINT A PAPER-READY TABLE UNTIL ALL 39 CELLS ARE IN. A mean over
a partial grid is not a smaller version of the result, it is a different
quantity: the folds are not interchangeable, and the sites that happen to have
finished first are whichever ones the loop reached, not a sample of anything.
Quoting one in a paper would be fabrication, so `--allow-partial` exists only
for watching progress and stamps every line it prints.

    python src/experiments/asformer_rerun_compare.py
    python src/experiments/asformer_rerun_compare.py --allow-partial
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

PUBLISHED = "experiments/tas/results.jsonl"
CORRECTED = "experiments/tas_asformer_fixed/results.jsonl"
GRID = "experiments/loso/results.jsonl"
N_CELLS = 39

METRICS = ("acc", "macro_f1", "f1@10", "f1@25", "f1@50", "edit", "violation")


def load(rel: str, arm: str | None = None) -> list[dict]:
    p = REPO / rel
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("temporal", "none")
        name = r["mode"] if t == "none" else f"{r['mode']}+{t}"
        if arm is None or name == arm:
            out.append(r)
    return out


def per_site(rows: list[dict], key: str) -> dict[str, float]:
    acc = collections.defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            acc[r["held_out"]].append(v)
    return {s: st.mean(v) for s, v in acc.items()}


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, int]:
    """(mean of b - a, two-sided p, n sites) over the sites both cover."""
    sites = sorted(set(a) & set(b))
    d = [b[s] - a[s] for s in sites]
    n = len(d)
    if n < 2:
        return (st.mean(d) if d else float("nan"), float("nan"), n)
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        return m, (0.0 if m else 1.0), n
    t = m / (sd / math.sqrt(n))
    return m, float(2 * student_t.sf(abs(t), n - 1)), n


def fmt_p(p: float) -> str:
    if p != p:
        return "—"
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--json", default="experiments/asformer_rerun_compare.json")
    args = ap.parse_args()

    pub = load(PUBLISHED, "none+asformer")
    cor = load(CORRECTED, "none+asformer")
    print(f"published arm : {len(pub)} cells   ({PUBLISHED})")
    print(f"corrected arm : {len(cor)} cells   ({CORRECTED})")
    if len(cor) < N_CELLS:
        print(f"\nThe corrected arm is INCOMPLETE: {len(cor)} of {N_CELLS} "
              f"cells.")
        if not args.allow_partial:
            print("Refusing to report means over a partial grid. The folds are "
                  "not\ninterchangeable, so a mean over whichever sites finished "
                  "first is a\ndifferent quantity, not a preview of the result. "
                  "Re-run when the arm\ncompletes, or pass --allow-partial to "
                  "watch progress.")
            done = sorted({r["held_out"] for r in cor})
            print(f"\nsites with any cell done ({len(done)}): "
                  f"{', '.join(done) if done else 'none'}")
            return 1
        print("--allow-partial: every figure below is PROVISIONAL and must not "
              "be quoted.\n")

    stamp = "" if len(cor) >= N_CELLS else "   [PARTIAL — DO NOT QUOTE]"

    # ---- per-site, published vs corrected ----------------------------------
    print(f"\nPER-SITE F1@50, published vs corrected{stamp}\n")
    pf, cf = per_site(pub, "f1@50"), per_site(cor, "f1@50")
    pa, ca = per_site(pub, "acc"), per_site(cor, "acc")
    print(f"{'site':10s} {'pub F1@50':>10s} {'new F1@50':>10s} {'Δ':>8s}   "
          f"{'pub acc':>8s} {'new acc':>8s} {'Δ':>8s}")
    for s in sorted(set(pf) & set(cf)):
        print(f"{s:10s} {pf[s]:10.3f} {cf[s]:10.3f} {cf[s] - pf[s]:+8.3f}   "
              f"{pa[s]:8.3f} {ca[s]:8.3f} {ca[s] - pa[s]:+8.3f}")

    # ---- arm means and the paired test -------------------------------------
    print(f"\n\nARM MEANS, run mean over cells{stamp}\n")
    print(f"{'metric':12s} {'published':>10s} {'corrected':>10s} {'Δ':>9s} "
          f"{'paired p':>9s} {'sites':>6s}")
    summary = {}
    for k in METRICS:
        pv = [r[k] for r in pub if r.get(k) is not None]
        cv = [r[k] for r in cor if r.get(k) is not None]
        if not pv or not cv:
            continue
        d, p, n = paired(per_site(pub, k), per_site(cor, k))
        summary[k] = {"published": st.mean(pv), "corrected": st.mean(cv),
                      "delta": d, "p": p, "n_sites": n}
        print(f"{k:12s} {st.mean(pv):10.3f} {st.mean(cv):10.3f} {d:+9.4f} "
              f"{fmt_p(p):>9s} {n:6d}")

    # onset is reported with its coverage, so it is computed the paper's way
    from data.sequence_dataset import list_sequence_dirs, scored_stages
    scored = {d.name: len(scored_stages(d))
              for d in list_sequence_dirs(REPO / "seq_data")}
    for name, rows in (("published", pub), ("corrected", cor)):
        fin = [r["onset_mae_frames"] for r in rows
               if r.get("onset_mae_frames") is not None
               and not math.isnan(r["onset_mae_frames"])]
        cov = st.mean(1.0 - r.get("onset_missing", 0) / scored[r["held_out"]]
                      for r in rows if scored.get(r["held_out"]))
        summary.setdefault("onset", {})[name] = {
            "run_mean": st.mean(fin) if fin else float("nan"),
            "cells_contributing": len(fin), "coverage": cov}
        print(f"onset ({name:9s}) {st.mean(fin) if fin else float('nan'):8.2f} "
              f"frames over {len(fin)} cells at coverage {cov:.2f}")

    # ---- the contrasts the paper quotes ------------------------------------
    print(f"\n\nCONTRASTS THE MANUSCRIPT QUOTES{stamp}\n")
    hard = load(GRID, "hard")
    ctr = {}
    for label, key in (("F1@50", "f1@50"), ("accuracy", "acc")):
        for which, rows in (("published", pub), ("corrected", cor)):
            d, p, n = paired(per_site(rows, key), per_site(hard, key))
            ctr[f"{which}_{key}"] = {"delta": d, "p": p, "n": n}
            print(f"  rigid minus ASFormer ({which:9s}) on {label:8s}: "
                  f"{d:+.4f}, p = {fmt_p(p)}, n = {n}")

    # ---- what has to change in the paper -----------------------------------
    if len(cor) >= N_CELLS:
        print("\n\nNUMBERS IN THE MANUSCRIPT THAT THIS CHANGES\n")
        old_f1, new_f1 = st.mean(r["f1@50"] for r in pub), st.mean(r["f1@50"] for r in cor)
        old_ac, new_ac = st.mean(r["acc"] for r in pub), st.mean(r["acc"] for r in cor)
        old_vi, new_vi = st.mean(r["violation"] for r in pub), st.mean(r["violation"] for r in cor)
        old_ed, new_ed = st.mean(r["edit"] for r in pub), st.mean(r["edit"] for r in cor)
        print(f"  Section 7.4 text : ASFormer F1@50 {old_f1:.3f} -> {new_f1:.3f}")
        print(f"  Table 4, F1@50   : {old_f1:.3f} -> {new_f1:.3f}")
        print(f"  Table 4, accuracy: {old_ac:.3f} -> {new_ac:.3f}")
        print(f"  Table 4, edit    : {old_ed:.3f} -> {new_ed:.3f}")
        print(f"  Table 4, violat. : {old_vi:.4f} -> {new_vi:.4f}")
        pc = ctr["published_f1@50"]; cc = ctr["corrected_f1@50"]
        print(f"  Section 7.4 test : rigid over ASFormer {pc['delta']:+.4f} "
              f"(p = {fmt_p(pc['p'])}) -> {cc['delta']:+.4f} "
              f"(p = {fmt_p(cc['p'])})")
        print(f"  Supplement S9/S10: every ASFormer row")

    out = Path(REPO / args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"complete": len(cor) >= N_CELLS, "n_published": len(pub),
         "n_corrected": len(cor), "metrics": summary, "contrasts": ctr},
        indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
