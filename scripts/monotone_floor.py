"""
The circularity control: how much of the segmental gain is definitional?

The objection, which a reviewer will raise
------------------------------------------
The corpus ground truth is perfectly monotone. Hard decoding forces a monotone
output. Segmental metrics (edit, F1@k) measure segment structure. So the paper
injects the true structure and then measures structural similarity --- the gain
could be tautological rather than evidence that the model learned anything.

The control
-----------
If the gain were purely definitional, a decoder that outputs a monotone path
while knowing NOTHING about the images would already collect it. So we score
image-free monotone predictors on the real corpus:

  uniform      : split each sequence into K equal parts, assign stages 0..K-1
  uniform-obs  : as above but over only the stages that occur in that sequence
                 (a stronger opponent: it is told which stages are present)
  constant     : predict a single stage everywhere (trivially monotone, 1 segment)

These need no training and use no pixels. Whatever they score is the floor that
monotonicity alone buys; the trained models must be judged against that floor,
not against zero.

    python src/experiments/monotone_floor.py
"""

from __future__ import annotations

import argparse
import json
import csv
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sequence_dataset import list_sequence_dirs        # noqa: E402
from metrics import compute_all                             # noqa: E402
from models.sequence_constraint import order_violation_rate  # noqa: E402


def load_labels(seq_dir: Path) -> list[int]:
    out = []
    with open(seq_dir / "labels.csv", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            s = int(row[1])
            if s >= 0:
                out.append(s)
    return out


def uniform_path(T: int, stages: list[int]) -> list[int]:
    """Monotone path: T frames split into len(stages) equal runs."""
    if not stages:
        return [0] * T
    k = len(stages)
    out = []
    for i in range(T):
        out.append(stages[min(k - 1, i * k // T)])
    return out


def subsample(seq: list[int], max_len: int) -> list[int]:
    """The evaluation view of a sequence: SequenceFrameDataset's even subsampling."""
    if len(seq) <= max_len:
        return list(seq)
    step = (len(seq) - 1) / (max_len - 1)
    return [seq[round(i * step)] for i in range(max_len)]


def load_runs(paths) -> dict:
    """{arm: {site: [rows]}} from LOSO results files."""
    arms: dict = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            t = r.get("temporal", "none")
            arm = t if t != "none" else r["mode"]
            arms.setdefault(arm, {}).setdefault(r["held_out"], []).append(r)
    return arms


def site_means(rows: dict, key: str) -> dict:
    out = {}
    for s, rs in rows.items():
        v = [r[key] for r in rs if r.get(key) is not None and r[key] == r[key]]
        if v:
            out[s] = st.mean(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--num-stages", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=64,
                    help="must match the --max-len the trained runs were scored at")
    args = ap.parse_args()

    dirs = [d for d in list_sequence_dirs(args.data)]
    full = {d.name: load_labels(d) for d in dirs}
    full = {k: v for k, v in full.items() if len(v) >= 2}
    # Score the controls on exactly what the models are scored on: each sequence
    # subsampled to --max-len, one sequence at a time, then averaged over sites.
    # The first version scored them on full-length sequences, pooled -- so its
    # onset floor (37.1 frames) was in raw frames while every model's onset MAE
    # is in subsampled frames, up to 13x smaller on the longest sequence.
    golds = {k: subsample(v, args.max_len) for k, v in full.items()}

    makers = {
        "uniform (all K stages)":
            lambda g: uniform_path(len(g), list(range(args.num_stages))),
        "uniform (stages present)":
            lambda g: uniform_path(len(g), sorted(set(g))),
        "constant (majority stage)":
            lambda g: [Counter(g).most_common(1)[0][0]] * len(g),
    }
    keys = ["acc", "macro_f1", "edit", "f1@10", "f1@25", "f1@50", "onset_mae_frames",
            "onset_coverage"]

    print(f"{len(golds)} sequences, {sum(len(g) for g in full.values())} labelled frames, "
          f"scored at max_len={args.max_len} "
          f"({sum(len(g) for g in golds.values())} frames)")
    print(f"gold order-violation rate: {order_violation_rate(list(full.values())):.4f}\n")
    print(f"{'image-free predictor':<28} " + " ".join(f"{k[:7]:>7}" for k in keys))
    print("-" * 86)
    per_site = {}
    for name, make in makers.items():
        rows = {}
        for s, g in golds.items():
            p = make(g)
            m = compute_all([p], [g])
            m["acc"] = sum(a == b for a, b in zip(p, g)) / len(g)
            rows[s] = m
        per_site[name] = rows
        print(f"{name:<28} " + " ".join(
            f"{st.mean(r[k] for r in rows.values() if r[k] == r[k]):>7.3f}" for k in keys))

    # ---- every trained arm against the control, paired per held-out site ----
    from scipy.stats import ttest_1samp, wilcoxon
    ctrl = per_site["uniform (all K stages)"]
    repo = Path(__file__).resolve().parents[2]
    arms = load_runs([repo / "experiments/loso/results.jsonl",
                      repo / "experiments/tas/results.jsonl"])
    order = [a for a in ["none", "free", "lstm", "mstcn", "asformer", "hard", "gated"]
             if a in arms]
    print("\n=== trained arm minus image-free control, paired per held-out site ===")
    print("    (onset: negative = arm better; sites where the arm predicts no onset"
          " are dropped)")
    for key in ["acc", "edit", "f1@10", "f1@25", "f1@50", "onset_mae_frames"]:
        print(f"  --- {key} ---")
        for a in order:
            m = site_means(arms[a], key)
            ss = sorted(s for s in m if s in ctrl and ctrl[s][key] == ctrl[s][key])
            d = [m[s] - ctrl[s][key] for s in ss]
            t = ttest_1samp(d, 0.0).pvalue
            try:
                w = wilcoxon(d).pvalue
            except ValueError:
                w = float("nan")
            better = sum(x < 0 for x in d) if key == "onset_mae_frames" else sum(x > 0 for x in d)
            print(f"    {a:>9}: arm {st.mean(m[s] for s in ss):6.3f}  control "
                  f"{st.mean(ctrl[s][key] for s in ss):6.3f}  diff {st.mean(d):+6.3f}  "
                  f"t-test p {t:.3f}  Wilcoxon p {w:.3f}  arm better on {better}/{len(d)}")
    print("\nAn arm 'clears' the control only if the paired difference is significant;")
    print("a higher mean with the arm ahead on half the sites is not a clearance.")


if __name__ == "__main__":
    main()
