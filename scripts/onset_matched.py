"""Stage-onset error on a MATCHED set — IJIES reviewer 2, point 6.

The published onset error averages only the stages a model predicted, so arms
with different coverage are not comparable: the rigid decoder's 4.40 frames is
computed over 61% of gold stages while the image-free control's 7.46 is computed
over 100%. Comparing them directly rewards dropping the hard stages.

This restricts every comparison to cells where the arm under test covered EVERY
gold stage, and scores the control on exactly those sites. The control needs no
restriction because it always covers everything by construction.

WHAT IT CANNOT DO. The reviewer asks for a matched set of STAGES; this matches
at the level of (site, seed) cells, because run_loso stores metrics and not
decoded tracks, and re-running to recover them does not reproduce the published
decode exactly (see dump_predictions.py). Cell-level matching is therefore the
strongest statement available from the published runs, and it is strictly
conservative: it discards whole cells where stage-level matching would have kept
the stages both arms did cover.

    python src/experiments/onset_matched.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.sequence_dataset import list_sequence_dirs          # noqa: E402
from metrics import compute_all                               # noqa: E402
from monotone_floor import load_labels, subsample, uniform_path  # noqa: E402

ARMS = ("none", "free", "hard", "gated")


def arm_label(r: dict) -> str:
    t = r.get("temporal", "none")
    return r["mode"] if t == "none" else f"{r['mode']}+{t}"


def coverage_of(r: dict) -> float:
    n = len(r.get("eval_stage_counts") or {})
    return float("nan") if not n else 1.0 - r.get("onset_missing", 0) / n


def penalised_onset(r: dict, penalty: float) -> float:
    """Onset error with every MISSED stage charged `penalty` frames.

    The published error divides by the stages a model kept, so dropping a stage
    removes a term instead of adding one. This divides by the stages the GOLD
    has, and charges the stages the model never predicted. It is recoverable
    from the stored record because mean x covered = the sum of the errors:

        (onset_mae * covered + missing * penalty) / n_gold

    `penalty` is stated, not tuned. Two are reported: the scored sequence length
    (the worst onset an in-range prediction can have), and the image-free
    control's error on that site (missing a stage scores no better than knowing
    nothing about the images).
    """
    n_gold = len(r.get("eval_stage_counts") or {})
    missing = r.get("onset_missing", 0)
    if not n_gold:
        return float("nan")
    covered = n_gold - missing
    mae = r.get("onset_mae_frames", float("nan"))
    total = 0.0 if (covered == 0 or math.isnan(mae)) else mae * covered
    return (total + missing * penalty) / n_gold


def control_per_site(data: str, num_stages: int, max_len: int) -> dict[str, float]:
    """Image-free uniform control's onset error, per site, on the scored length."""
    out = {}
    for d in list_sequence_dirs(data):
        g = load_labels(d)
        if len(g) < 2:
            continue
        g = subsample(g, max_len)
        p = uniform_path(len(g), list(range(num_stages)))
        out[d.name] = compute_all([p], [g])["onset_mae_frames"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/loso/results.jsonl")
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--num-stages", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--json", default=None, help="write the table as JSON")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.results).open(encoding="utf-8") if l.strip()]
    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[arm_label(r)].append(r)

    ctrl = control_per_site(args.data, args.num_stages, args.max_len)
    print(f"image-free control: {len(ctrl)} sites, "
          f"mean onset {st.mean(ctrl.values()):.2f} frames at coverage 1.00\n")

    report = {}
    print(f"{'arm':12s} {'cells':>6s} {'full-cov':>9s} {'onset pub':>10s} "
          f"{'onset matched':>14s} {'control same':>13s} {'arm - ctrl':>11s}")
    for arm in sorted(by_arm, key=lambda a: (ARMS.index(a.split('+')[0])
                                             if a.split('+')[0] in ARMS else 9, a)):
        rs = by_arm[arm]
        finite = [r for r in rs if not math.isnan(r.get("onset_mae_frames", float("nan")))]
        pub = st.mean(r["onset_mae_frames"] for r in finite) if finite else float("nan")

        full = [r for r in rs if coverage_of(r) == 1.0]
        if not full:
            print(f"{arm:12s} {len(rs):6d} {0:9d} {pub:10.2f} "
                  f"{'n/a':>14s} {'n/a':>13s} {'n/a':>11s}")
            continue
        matched = st.mean(r["onset_mae_frames"] for r in full)
        ctrl_same = st.mean(ctrl[r["held_out"]] for r in full if r["held_out"] in ctrl)
        diffs = [r["onset_mae_frames"] - ctrl[r["held_out"]]
                 for r in full if r["held_out"] in ctrl]
        delta = st.mean(diffs)
        print(f"{arm:12s} {len(rs):6d} {len(full):9d} {pub:10.2f} "
              f"{matched:14.2f} {ctrl_same:13.2f} {delta:+11.2f}")
        report[arm] = {
            "cells": len(rs), "cells_full_coverage": len(full),
            "onset_published": round(pub, 3),
            "onset_matched": round(matched, 3),
            "control_on_same_cells": round(ctrl_same, 3),
            "arm_minus_control": round(delta, 3),
            "sites_full_coverage": sorted({r["held_out"] for r in full}),
        }

    print("\n'onset pub'     = as published: averaged over whatever each arm covered")
    print("'onset matched' = cells where the arm covered EVERY gold stage")
    print("'control same'  = the image-free control scored on those same cells")
    print("'arm - ctrl'    = paired difference; negative means the arm is better")

    print("")
    print("NOTE: 'onset matched' conditions on the arm's OWN success -- it keeps")
    print("only the cells where that arm covered everything -- so it is not an")
    print("unbiased estimate either. The penalised rows below avoid both biases")
    print("by charging missed stages instead of discarding them.")
    print("")
    print(f"{'arm':12s} {'coverage':>9s} {'pen@len':>9s} {'pen@ctrl':>9s} {'control':>9s}")
    for arm in sorted(by_arm, key=lambda a: (ARMS.index(a.split('+')[0])
                                             if a.split('+')[0] in ARMS else 9, a)):
        rs = by_arm[arm]
        cov = [c for c in (coverage_of(r) for r in rs) if not math.isnan(c)]
        pen_len = st.mean(penalised_onset(r, args.max_len) for r in rs)
        pen_ctl = st.mean(penalised_onset(r, ctrl.get(r["held_out"], args.max_len))
                          for r in rs)
        c_len = st.mean(ctrl[r["held_out"]] for r in rs if r["held_out"] in ctrl)
        print(f"{arm:12s} {st.mean(cov):9.3f} {pen_len:9.2f} {pen_ctl:9.2f} {c_len:9.2f}")
        report.setdefault(arm, {}).update({
            "coverage": round(st.mean(cov), 3),
            "onset_penalised_at_seq_len": round(pen_len, 3),
            "onset_penalised_at_control": round(pen_ctl, 3),
            "control_onset_same_sites": round(c_len, 3),
        })
    print("")
    print(f"'pen@len'  = missed stages charged the scored sequence length ({args.max_len} frames)")
    print("'pen@ctrl' = missed stages charged the image-free control's error on that site")
    print("'control'  = the image-free control, which covers every stage so penalties")
    print("             leave it unchanged. This is the comparison that survives.")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
