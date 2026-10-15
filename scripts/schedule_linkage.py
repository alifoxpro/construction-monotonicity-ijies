"""Translate stage-onset error from frames into days of project time.

Why this exists
---------------
Stage-onset MAE is reported in frames. A frame count means nothing to a project
manager: the question they ask is "when did the envelope actually start, and how
far out was the model?" -- in days. Converting requires a mapping from frame
index to calendar date, and for edited time-lapse that mapping is neither
available nor linear.

seq0010 is the exception. Its source footage carries the project week number and
an ISO date burnt into every frame, recorded in seq_data_raw/seq0010/timeline.json.
That gives an exact frame -> date map for one site, and lets us do two things:

  1. Report onset error in DAYS for that site.
  2. Measure how badly a linear frames -> days conversion would mislead. This
     matters beyond one sequence: any study that reports temporal error on
     time-lapse footage without a timestamp is implicitly assuming uniformity.

    python src/experiments/schedule_linkage.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_timeline(seq: str) -> list[tuple[int, date, int]]:
    """[(frame, date, week)] sorted by frame."""
    j = json.loads((REPO / "seq_data_raw" / seq / "timeline.json").read_text(encoding="utf-8"))
    out = [(a["frame"], date.fromisoformat(a["date"]), a["week"]) for a in j["anchors"]]
    return sorted(out)


def frame_to_date(frame: float, anchors) -> date:
    """Piecewise-linear interpolation between read anchors.

    Outside the anchor range we clamp rather than extrapolate: the footage
    genuinely has no dated frames there (title card, closing sequence).
    """
    if frame <= anchors[0][0]:
        return anchors[0][1]
    if frame >= anchors[-1][0]:
        return anchors[-1][1]
    for (f0, d0, _), (f1, d1, _) in zip(anchors, anchors[1:]):
        if f0 <= frame <= f1:
            if f1 == f0:
                return d0
            span = (d1 - d0).days
            return d0 + (d1 - d0) * 0 + __import__("datetime").timedelta(
                days=round(span * (frame - f0) / (f1 - f0)))
    return anchors[-1][1]


def uniformity_report(anchors) -> dict:
    """How far from linear is this time-lapse? Days per frame, segment by segment."""
    rates = []
    for (f0, d0, _), (f1, d1, _) in zip(anchors, anchors[1:]):
        if f1 == f0:
            continue
        rates.append(((f0, f1), (d1 - d0).days / (f1 - f0)))
    vals = [r for _, r in rates]
    moving = [r for r in vals if r > 0]      # segments where the project advanced
    total_days = (anchors[-1][1] - anchors[0][1]).days
    total_frames = anchors[-1][0] - anchors[0][0]
    linear = total_days / total_frames
    return {
        "segments": rates,
        "min_days_per_frame": min(vals),
        "max_days_per_frame": max(vals),
        "stalled_segments": sum(1 for r in vals if r == 0),
        # ratio over MOVING segments only: including a stalled segment makes the
        # ratio infinite and says nothing
        "ratio_moving": max(moving) / min(moving) if moving else float("nan"),
        "linear_days_per_frame": linear,
        # worst-case error factor from assuming linearity, in each direction
        "linear_overstates_by": linear / min(moving) if moving else float("nan"),
        "linear_understates_by": max(vals) / linear if linear else float("nan"),
        "total_days": total_days,
        "total_frames": total_frames,
    }


def onsets(labels: list[int], n_stages: int = 5) -> dict[int, int]:
    return {s: next((i for i, v in enumerate(labels) if v == s), None)
            for s in range(n_stages)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="seq0010")
    ap.add_argument("--pred", default="experiments/schedule/pred_seq0010.json",
                    help="per-frame predictions dumped by dump_predictions.py")
    args = ap.parse_args()

    anchors = load_timeline(args.seq)
    u = uniformity_report(anchors)

    print(f"=== {args.seq}: how non-linear is the time-lapse? ===")
    print(f"  spans {u['total_days']} days over {u['total_frames']} frames")
    print(f"  a LINEAR conversion would assume {u['linear_days_per_frame']:.2f} days/frame")
    print(f"  actual range: {u['min_days_per_frame']:.2f} to {u['max_days_per_frame']:.2f} "
          f"days/frame, with {u['stalled_segments']} segment(s) stalled at zero")
    print(f"  among moving segments the rate varies {u['ratio_moving']:.0f}x")
    print(f"  assuming linearity overstates elapsed time by up to "
          f"{u['linear_overstates_by']:.1f}x and understates it by up to "
          f"{u['linear_understates_by']:.1f}x")
    print("\n  slowest and fastest segments:")
    for (f0, f1), r in sorted(u["segments"], key=lambda x: x[1])[:3]:
        print(f"    frames {f0:3d}-{f1:3d}: {r:6.2f} days/frame")
    for (f0, f1), r in sorted(u["segments"], key=lambda x: x[1])[-3:]:
        print(f"    frames {f0:3d}-{f1:3d}: {r:6.2f} days/frame")

    pred_path = REPO / args.pred
    if not pred_path.exists():
        print(f"\n(no predictions at {args.pred}; run dump_predictions.py to get "
              f"onset error in days)")
        return 0

    d = json.loads(pred_path.read_text(encoding="utf-8"))
    gold, pred = d["gold"], d["pred"]
    go_eval, po_eval = onsets(gold), onsets(pred)

    # pred/gold are positions in the SUBSAMPLED sequence the evaluation saw; the
    # date anchors are RAW frame numbers. Dating a position as if it were a frame
    # number -- which the first version did -- reads the wrong date whenever the
    # sequence was subsampled. Map position -> assessable index -> raw frame.
    raw_labels = [int(r.split(",")[1]) for r in (REPO / "seq_data_raw" / args.seq
                  / "labels.csv").read_text(encoding="utf-8").split("\n") if r.strip()]
    raw_of_assessable = [i for i, v in enumerate(raw_labels) if v >= 0]
    raw_of_pos = [raw_of_assessable[i] for i in d["assessable_index"]]

    def to_raw(o):
        return {s: (raw_of_pos[p] if p is not None else None) for s, p in o.items()}

    go, po = to_raw(go_eval), to_raw(po_eval)
    print(f"\n  onsets mapped from {len(pred)} evaluated positions to raw frames "
          f"(evaluation onset error is in positions; dating uses raw frames)")

    print(f"\n=== onset error in DAYS (arm: {d.get('mode', '?')}, "
          f"seed {d.get('seed', '?')}) ===")

    # A prediction that emits only one stage has exactly one onset, which then
    # matches trivially and reports as zero error. That is a COLLAPSE, not a
    # result, and must never be averaged in silently.
    predicted_stages = sorted({s for s in po if po[s] is not None})
    gold_stages = sorted({s for s in go if go[s] is not None})
    if len(predicted_stages) < len(gold_stages):
        print(f"  WARNING: the model emitted only stage(s) {predicted_stages} "
              f"where the ground truth has {gold_stages}.")
        print(f"  Onsets for the missing stages do not exist, so any mean over "
              f"the surviving ones is not an error rate. Reported below, not "
              f"aggregated.")

    print(f"{'stage':>6}  {'true onset':>22}  {'predicted':>22}  {'error':>9}")
    errs_days, errs_frames = [], []
    for s in range(5):
        if go[s] is None:
            print(f"{s:>6}  {'(absent from ground truth)':>22}")
            continue
        if po[s] is None:
            print(f"{s:>6}  frame {go[s]:>3} = {frame_to_date(go[s], anchors)}  "
                  f"{'NEVER PREDICTED':>22}")
            continue
        gd, pd = frame_to_date(go[s], anchors), frame_to_date(po[s], anchors)
        dd = (pd - gd).days
        errs_days.append(abs(dd))
        errs_frames.append(abs(po[s] - go[s]))
        print(f"{s:>6}  frame {go[s]:>3} = {gd}  frame {po[s]:>3} = {pd}  "
              f"{dd:+5d} d   ({po_eval[s] - go_eval[s]:+d} evaluated positions, "
              f"{po[s] - go[s]:+d} raw frames)")

    if errs_days and len(predicted_stages) == len(gold_stages):
        print(f"\n  mean absolute onset error: {st.mean(errs_frames):.1f} frames "
              f"= {st.mean(errs_days):.0f} DAYS")
        lin = st.mean(errs_frames) * u["linear_days_per_frame"]
        print(f"  the same error read off a LINEAR assumption: {lin:.0f} days")
        if st.mean(errs_days) > 0:
            print(f"  the linear reading is wrong by a factor of "
                  f"{lin / st.mean(errs_days):.1f}")
    elif errs_days:
        print(f"\n  NOT aggregated: the prediction collapsed to "
              f"{len(predicted_stages)} of {len(gold_stages)} stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
