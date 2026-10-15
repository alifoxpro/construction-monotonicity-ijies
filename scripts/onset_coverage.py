"""Onset coverage for results already on disk — IJIES reviewer 2, point 6.

The reviewer's objection is that stage-onset error averages only the stages a
model actually predicted, so a missed stage REMOVES a term rather than adding a
penalty. A model can therefore post a better onset error by predicting less, and
a model that predicts nothing posts NaN and then disappears from any mean taken
over runs.

Coverage is recoverable from the stored records without retraining anything:
every run already carries `onset_missing` and `eval_stage_counts`, so

    coverage = 1 - onset_missing / (number of gold stages in the held-out site)

This reports, per arm: the published onset error, the coverage it was computed
at, how many runs contributed no onset at all (NaN, silently dropped from the
mean), and the error recomputed over only those sites where EVERY arm covered
every stage -- the matched-set comparison the reviewer asks for.

    python src/experiments/onset_coverage.py                      # paper grid
    python src/experiments/onset_coverage.py --results <path>     # any grid
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

ARM_ORDER = ("none", "free", "hard", "soft", "gated")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


#: site -> number of stages in the gold sequence the evaluation actually scores.
#: NOT len(eval_stage_counts), which histograms every labelled frame: seq0005
#: carries stage 1 on 5 of 830 frames and no subsampled position lands on them,
#: so that stage is in the record but not in the scored gold. Counting it would
#: charge the model for missing a stage it was never shown.
_SCORED: dict[str, int] = {}


def scored_stage_count(site: str) -> int:
    if not _SCORED:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from data.sequence_dataset import list_sequence_dirs, scored_stages
        for d in list_sequence_dirs(_P(__file__).resolve().parents[2] / "seq_data"):
            _SCORED[d.name] = len(scored_stages(d))
    return _SCORED.get(site, 0)


def coverage_of(rec: dict) -> float:
    n_gold = scored_stage_count(rec.get("held_out", ""))
    if not n_gold:
        return float("nan")
    return 1.0 - rec.get("onset_missing", 0) / n_gold


def arm_label(rec: dict) -> str:
    t = rec.get("temporal", "none")
    return rec["mode"] if t == "none" else f"{rec['mode']}+{t}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/loso/results.jsonl")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    rows = load(path)
    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_arm[arm_label(r)].append(r)

    print(f"{path}  —  {len(rows)} runs, {len(by_arm)} arms\n")

    # ---- per-arm, as published -------------------------------------------
    print("AS PUBLISHED (onset error averaged over whatever each arm covered)")
    print(f"{'arm':14s} {'runs':>5s} {'onset':>8s} {'coverage':>9s} "
          f"{'NaN runs':>9s} {'missed stages':>14s}")
    for arm in sorted(by_arm, key=lambda a: (ARM_ORDER.index(a.split('+')[0])
                                             if a.split('+')[0] in ARM_ORDER else 9, a)):
        rs = by_arm[arm]
        finite = [r for r in rs if not math.isnan(r.get("onset_mae_frames", float("nan")))]
        nan_n = len(rs) - len(finite)
        cov = [coverage_of(r) for r in rs]
        cov = [c for c in cov if not math.isnan(c)]
        missed = sum(r.get("onset_missing", 0) for r in rs)
        onset = st.mean(r["onset_mae_frames"] for r in finite) if finite else float("nan")
        print(f"{arm:14s} {len(rs):5d} {onset:8.2f} {st.mean(cov) if cov else float('nan'):9.3f} "
              f"{nan_n:9d} {missed:14d}")

    # ---- matched set: sites where every arm covered every stage ----------
    per_site: dict[tuple, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        per_site[(r["held_out"], r["seed"])][arm_label(r)] = r
    arms = sorted(by_arm)
    full = [k for k, d in per_site.items()
            if len(d) == len(arms) and all(coverage_of(v) == 1.0 for v in d.values())]
    partial = [k for k, d in per_site.items() if len(d) == len(arms)]

    print(f"\nMATCHED SET: {len(full)} of {len(partial)} (site, seed) cells have "
          f"full coverage in EVERY arm")
    if not full:
        print("  no cell is fully covered by every arm — the published onset means "
              "are therefore not computed over a common set of stages")
        return 0
    print(f"{'arm':14s} {'onset (matched)':>16s} {'onset (published)':>18s} {'delta':>8s}")
    for arm in sorted(by_arm, key=lambda a: (ARM_ORDER.index(a.split('+')[0])
                                             if a.split('+')[0] in ARM_ORDER else 9, a)):
        m = st.mean(per_site[k][arm]["onset_mae_frames"] for k in full)
        rs = [r for r in by_arm[arm] if not math.isnan(r.get("onset_mae_frames", float("nan")))]
        p = st.mean(r["onset_mae_frames"] for r in rs) if rs else float("nan")
        print(f"{arm:14s} {m:16.2f} {p:18.2f} {m - p:+8.2f}")

    # ---- the sites that vanish -------------------------------------------
    print("\nSITES CONTRIBUTING NO ONSET AT ALL (dropped from every mean):")
    gone = collections.Counter()
    for r in rows:
        if math.isnan(r.get("onset_mae_frames", float("nan"))):
            gone[(r["held_out"], arm_label(r))] += 1
    if not gone:
        print("  none")
    else:
        for (site, arm), n in sorted(gone.items()):
            print(f"  {site:10s} {arm:14s} {n} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
