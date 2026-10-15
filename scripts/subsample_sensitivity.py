"""Does the headline survive a different subsample? — IJIES reviewer 2, point 6.

THE OBJECTION, verbatim: "Examine sensitivity to the 64-frame subsampling, since
the raw sequences have very different lengths and editing rates."

The corpus runs from 19 to 830 labelled frames. At --max-len 64 the shortest
sites keep every frame and the longest keep about one frame in thirteen, so the
thinning is wildly uneven across sites, and a result that depends on it is a
result about the evaluation rather than about the decoder.

We re-ran the confirmatory contrast -- free CRF against the rigid prior -- at
max-len 32 and 128 over the same 13 folds and 3 seeds, everything else held
fixed. 128 exceeds the labelled length of nine of the thirteen sites, so for most
of the corpus it is a no-thinning setting rather than a second thinning, which
is what makes it the informative direction: 32 only asks whether MORE thinning
changes the answer, while 128 asks whether the thinning was causing it.

    python src/experiments/subsample_sensitivity.py
"""
from __future__ import annotations

import argparse
import collections
import csv
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

from data.sequence_dataset import list_sequence_dirs  # noqa: E402

#: max-len -> results file. 64 is the published grid.
SOURCES = {
    32: "experiments/subsample_sensitivity/len32/results.jsonl",
    64: "experiments/loso/results.jsonl",
    128: "experiments/subsample_sensitivity/len128/results.jsonl",
}
METRICS = ("f1@50", "acc", "edit")


def load(rel: str, arm: str) -> dict[str, list[float]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    p = REPO / rel
    if not p.exists():
        return {}
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r.get("temporal", "none")
        if (r["mode"] if t == "none" else f"{r['mode']}+{t}") == arm:
            out[r["held_out"]].append(r)
    return out


def site_means(rows: dict[str, list[dict]], key: str) -> dict[str, float]:
    return {s: st.mean(r[key] for r in v if r.get(key) is not None)
            for s, v in rows.items()
            if any(r.get(key) is not None for r in v)}


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, int, int]:
    """(mean of b - a, two-sided p, n sites, wins) over the sites both cover."""
    sites = sorted(set(a) & set(b))
    d = [b[s] - a[s] for s in sites]
    n = len(d)
    if n < 2:
        return (st.mean(d) if d else float("nan"), float("nan"), n, 0)
    m, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        return m, (0.0 if m else 1.0), n, sum(1 for x in d if x > 0)
    t = m / (sd / math.sqrt(n))
    return (m, float(2 * student_t.sf(abs(t), n - 1)), n,
            sum(1 for x in d if x > 0))


def fmt_p(p: float) -> str:
    if p != p:
        return "—"
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="experiments/subsample_sensitivity/summary.json")
    args = ap.parse_args()

    # ---- how much thinning each setting actually applies -------------------
    lengths = {}
    for d in list_sequence_dirs(REPO / "seq_data"):
        n = 0
        with open(d / "labels.csv", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].strip().lower() not in {"filename", "file", ""} \
                        and int(row[1]) >= 0:
                    n += 1
        lengths[d.name] = n
    print("THINNING APPLIED BY EACH SETTING\n")
    print(f"{'max-len':>8s} {'sites kept whole':>17s} {'frames scored':>14s} "
          f"{'worst thinning':>15s}")
    for L in sorted(SOURCES):
        whole = sum(1 for n in lengths.values() if n <= L)
        scored = sum(min(n, L) for n in lengths.values())
        worst = max(n / min(n, L) for n in lengths.values())
        print(f"{L:8d} {whole:10d} of 13 {scored:14d} {worst:14.1f}x")
    print(f"\ncorpus: {len(lengths)} sites, {sum(lengths.values())} labelled "
          f"frames, {min(lengths.values())} to {max(lengths.values())} per site")

    # ---- the contrast at each length ---------------------------------------
    print("\n\nCONFIRMATORY CONTRAST AT EACH SUBSAMPLE LENGTH")
    print("free CRF -> rigid prior, paired per held-out site, seeds averaged "
          "within site\n")
    results = {}
    for key in METRICS:
        print(f"  --- {key} ---")
        print(f"  {'max-len':>8s} {'free':>8s} {'rigid':>8s} {'diff':>9s} "
              f"{'p':>8s} {'wins':>7s}")
        for L in sorted(SOURCES):
            free, hard = load(SOURCES[L], "free"), load(SOURCES[L], "hard")
            if not free or not hard:
                print(f"  {L:8d}   (missing)")
                continue
            fa, ha = site_means(free, key), site_means(hard, key)
            m, p, n, wins = paired(fa, ha)
            results.setdefault(key, {})[L] = {
                "free": st.mean(fa.values()), "rigid": st.mean(ha.values()),
                "diff": m, "p": p, "n_sites": n, "wins": wins}
            mark = "   <- published" if L == 64 else ""
            print(f"  {L:8d} {st.mean(fa.values()):8.3f} "
                  f"{st.mean(ha.values()):8.3f} {m:+9.4f} {fmt_p(p):>8s} "
                  f"{wins:3d}/{n:<3d}{mark}")
        print()

    # ---- verdict -----------------------------------------------------------
    print("\nWHAT THIS SHOWS")
    f1 = results.get("f1@50", {})
    if len(f1) == 3:
        signs = {L: (v["diff"] > 0) for L, v in f1.items()}
        sig = {L: (v["p"] < 0.05) for L, v in f1.items()}
        print(f"  direction on F1@50 : " +
              ", ".join(f"{L}: {'+' if signs[L] else '-'}" for L in sorted(f1)))
        print(f"  significant at .05 : " +
              ", ".join(f"{L}: {'yes' if sig[L] else 'no'}" for L in sorted(f1)))
        print(f"  size               : " +
              ", ".join(f"{L}: {f1[L]['diff']:+.3f}" for L in sorted(f1)))
        if all(signs.values()) and all(sig.values()):
            print("\n  The contrast keeps its sign and its significance at every")
            print("  length, including 128, where nine of the thirteen sites are")
            print("  scored on every labelled frame. The result is therefore not")
            print("  an artifact of the 64-frame subsample.")
        elif all(signs.values()):
            print("\n  The contrast keeps its SIGN at every length but not its")
            print("  significance everywhere. It is not reversed by the choice of")
            print("  subsample, but its strength depends on it, and that")
            print("  dependence belongs in the paper.")
        else:
            print("\n  The contrast CHANGES SIGN with the subsample length. The")
            print("  published result is then a property of the evaluation as")
            print("  much as of the decoder, and the paper must say so.")

    out = Path(REPO / args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lengths": lengths, "contrast": results},
                              indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
