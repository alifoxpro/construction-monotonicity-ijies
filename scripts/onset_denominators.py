"""Every onset number in the paper, with its denominators — IJIES reviewer 1, point 7.

THE OBJECTION, verbatim: "Explain why the rigid decoder's onset error is 4.40
frames in Table 2 and the Conclusion but 4.65 in Sections 5.3 and 7.1.1. The
latter uses twelve comparable sites, which may explain the difference, but the
relationship to the full-corpus estimate and to the 1.11-frame annotation
disagreement is not documented. Give the exact site and stage denominators for
each value and label subset estimates explicitly."

Three numbers are in play and no two of them are averages of the same thing.

  4.40  the RUN mean: every run of the arm that produced any onset at all,
        averaged over runs. Runs that produced none are NaN and silently vanish.
  4.65  the SITE mean: seeds averaged within site first, then averaged over the
        sites that produced an onset. seq0009 produces none under any arm.
  1.11  the ANNOTATOR mean: per-sequence onset disagreement between the two
        annotators, averaged over sequences.

The first two differ because of which runs and sites survive. The third differs
from both for a reason the manuscript never states: it is computed over nearly
every stage, while the decoder's is computed over the 61% of stages the decoder
actually predicted. A stage the decoder never predicts contributes no term to
its error, so the two means are not taken over a common set of stages, and the
gap between them is not a clean measure of headroom.

It also includes a sequence the decoder figure excludes. seq0009 carries one
stage across every frame: both annotators place that single onset at frame 0, so
it contributes exactly 0.00 to the annotator mean, while contributing nothing at
all to the decoder mean. Removing it from the annotator mean for comparability
moves 1.11 to 1.20.

This script prints each value with the site count, the run count, the stage-term
count and the coverage it was computed at, so that every onset figure in the
paper can be traced to a denominator.

    python src/experiments/onset_denominators.py
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from data.sequence_dataset import list_sequence_dirs, scored_stages  # noqa: E402

RAW = REPO / "seq_data_raw"
MAX_LEN = 64

ARM_SOURCE = {
    "none": "experiments/loso/results.jsonl",
    "free": "experiments/loso/results.jsonl",
    "hard": "experiments/loso/results.jsonl",
    "gated": "experiments/loso/results.jsonl",
    "none+lstm": "experiments/loso/results.jsonl",
    "none+mstcn": "experiments/tas/results.jsonl",
    "none+asformer": "experiments/tas/results.jsonl",
}
ARM_ORDER = ["none", "free", "hard", "gated", "none+lstm",
             "none+mstcn", "none+asformer"]


def read_labels(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            rows.append((row[0].strip(), int(row[1])))
    return rows


def subsample_idx(n: int, max_len: int) -> list[int]:
    if n <= max_len:
        return list(range(n))
    step = (n - 1) / (max_len - 1)
    return [round(i * step) for i in range(max_len)]


def onsets(stages: list[int]) -> dict[int, int]:
    """stage -> first index at which it appears."""
    out: dict[int, int] = {}
    for i, s in enumerate(stages):
        out.setdefault(s, i)
    return out


def load_arms() -> dict[str, dict[str, list[dict]]]:
    arms: dict[str, dict[str, list[dict]]] = {}
    cache: dict[str, list[dict]] = {}
    for arm, rel in ARM_SOURCE.items():
        if rel not in cache:
            p = REPO / rel
            cache[rel] = ([json.loads(l) for l in p.open(encoding="utf-8")
                           if l.strip()] if p.exists() else [])
        for r in cache[rel]:
            t = r.get("temporal", "none")
            if (r["mode"] if t == "none" else f"{r['mode']}+{t}") == arm:
                arms.setdefault(arm, {}).setdefault(r["held_out"], []).append(r)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--json", default="experiments/onset_denominators.json")
    args = ap.parse_args()

    dirs = list_sequence_dirs(REPO / args.data)
    sites = [d.name for d in dirs]
    scored = {d.name: len(scored_stages(d)) for d in dirs}
    arms = load_arms()

    # ---------------------------------------------------------------- decoder
    print("DECODER ONSET ERROR — the same arm under two estimators\n")
    hdr = (f"{'arm':14s} {'run mean':>9s} {'runs':>8s} {'site mean':>10s} "
           f"{'sites':>7s} {'stage terms':>12s} {'missing':>8s} {'coverage':>9s}")
    print(hdr)
    print("-" * len(hdr))
    table = {}
    for arm in [a for a in ARM_ORDER if a in arms]:
        per_site = arms[arm]
        runs = [r for s in per_site for r in per_site[s]]
        finite = [r for r in runs
                  if r.get("onset_mae_frames") is not None
                  and not math.isnan(r["onset_mae_frames"])]
        run_mean = st.mean(r["onset_mae_frames"] for r in finite) if finite else float("nan")

        site_vals = {}
        for s, rs in per_site.items():
            f = [r["onset_mae_frames"] for r in rs
                 if r.get("onset_mae_frames") is not None
                 and not math.isnan(r["onset_mae_frames"])]
            if f:
                site_vals[s] = st.mean(f)
        site_mean = st.mean(site_vals.values()) if site_vals else float("nan")

        # SCORED stages, not the record's eval_stage_counts: that histograms
        # every labelled frame, and a stage too short to survive the 64-frame
        # subsample appears in it without being in the gold the model saw.
        terms = sum(scored[r["held_out"]] for r in runs)
        missing = sum(r.get("onset_missing", 0) for r in runs)
        cov = st.mean(1.0 - r.get("onset_missing", 0) / scored[r["held_out"]]
                      for r in runs if scored.get(r["held_out"]))
        table[arm] = {"run_mean": run_mean, "n_runs_total": len(runs),
                      "n_runs_contributing": len(finite),
                      "site_mean": site_mean, "n_sites_total": len(per_site),
                      "n_sites_contributing": len(site_vals),
                      "stage_terms": terms, "missing_terms": missing,
                      "coverage": cov,
                      "sites_dropped": sorted(set(per_site) - set(site_vals))}
        print(f"{arm:14s} {run_mean:9.2f} {len(finite):3d}/{len(runs):<4d} "
              f"{site_mean:10.2f} {len(site_vals):2d}/{len(per_site):<4d} "
              f"{terms:12d} {missing:8d} {cov:9.3f}")

    h = table.get("hard", {})
    if h:
        print(f"\nThe two figures the reviewer asks about are both the rigid arm:")
        print(f"  4.40 = the RUN mean  : {h['run_mean']:.2f} over "
              f"{h['n_runs_contributing']} of {h['n_runs_total']} runs "
              f"({h['n_runs_total'] - h['n_runs_contributing']} produced no onset)")
        print(f"  4.65 = the SITE mean : {h['site_mean']:.2f} over "
              f"{h['n_sites_contributing']} of {h['n_sites_total']} sites "
              f"(dropped: {', '.join(h['sites_dropped']) or 'none'})")
        print(f"  both are computed over {h['stage_terms'] - h['missing_terms']} "
              f"of {h['stage_terms']} gold stage onsets, a coverage of "
              f"{h['coverage']:.2f}")
        print(f"\nNeither is a full-corpus estimate. Both are subset estimates and")
        print(f"should be labelled as such wherever they appear.")

    # -------------------------------------------------------------- annotator
    print("\n\nANNOTATOR DISAGREEMENT — on the frames the evaluation actually saw\n")
    print("Computed on annotator A's assessable frames, subsampled to 64 exactly")
    print("as the evaluation subsamples them, so the units are the decoder's.\n")
    hdr = (f"{'site':10s} {'frames':>7s} {'stages A':>9s} {'shared':>7s} "
           f"{'coverage':>9s} {'onset disagreement':>19s}")
    print(hdr)
    print("-" * len(hdr))
    per_seq, per_onset, rows = [], [], {}
    for site in sites:
        a_raw = [s for _, s in read_labels(RAW / site / "labels.csv")]
        b_path = RAW / site / "labels.B.csv"
        if not b_path.exists():
            continue
        b_raw = [s for _, s in read_labels(b_path)]
        # A's assessable positions, then A's 64-frame subsample of them. B is
        # read at the SAME raw frames, not at B's own assessable subsample:
        # the question is what a second reader would have said about the frames
        # the model was scored on.
        assess = [i for i, s in enumerate(a_raw) if s >= 0]
        keep = [assess[p] for p in subsample_idx(len(assess), MAX_LEN)]
        a_sub = [a_raw[i] for i in keep]
        b_sub = [b_raw[i] if i < len(b_raw) else -1 for i in keep]
        oa, ob = onsets(a_sub), onsets([s for s in b_sub])
        shared = [s for s in oa if s >= 0 and s in ob]
        diffs = [abs(oa[s] - ob[s]) for s in shared]
        n_a = len([s for s in oa if s >= 0])
        cov = len(shared) / n_a if n_a else float("nan")
        m = st.mean(diffs) if diffs else float("nan")
        per_seq.append(m)
        per_onset += diffs
        rows[site] = {"frames": len(keep), "stages_a": n_a,
                      "shared": len(shared), "coverage": cov, "mean": m}
        print(f"{site:10s} {len(keep):7d} {n_a:9d} {len(shared):7d} "
              f"{cov:9.3f} {m:19.2f}")

    deg = [s for s in rows if rows[s]["stages_a"] == 1]
    matched = [rows[s]["mean"] for s in rows if s not in deg]
    print(f"\n  per sequence, all {len(per_seq)} sequences : "
          f"{st.mean(per_seq):.2f}   <- the paper's 1.11")
    print(f"  per sequence, excluding {', '.join(deg) or 'none'}      : "
          f"{st.mean(matched):.2f}   <- comparable with the decoder's 4.65")
    print(f"  per individual onset, n = {len(per_onset)}        : "
          f"{st.mean(per_onset):.2f}")
    print(f"  coverage of the annotator comparison       : "
          f"{st.mean(r['coverage'] for r in rows.values()):.3f}")

    # ------------------------------------------------------------- the caveat
    print("\n\nWHY THE TWO ARE NOT A LIKE-FOR-LIKE HEADROOM MEASURE")
    if h:
        print(f"  annotator disagreement is computed at coverage "
              f"{st.mean(r['coverage'] for r in rows.values()):.2f};")
        print(f"  the rigid decoder's error at coverage {h['coverage']:.2f}.")
        print(f"  A stage the decoder never predicts contributes no term to its")
        print(f"  error, so the decoder is scored on an easier subset of stages")
        print(f"  than the annotators are. The difference between "
              f"{st.mean(matched):.2f} and {h['site_mean']:.2f}")
        print(f"  frames is therefore an upper bound on the headroom, not an")
        print(f"  estimate of it, and the sign of the bias is known: correcting")
        print(f"  for coverage can only widen the decoder's error, never shrink")
        print(f"  it.")
    print(f"\n  seq0009 carries one stage, so both annotators place its single")
    print(f"  onset at frame 0 and it contributes exactly 0.00 to the annotator")
    print(f"  mean, while contributing nothing to the decoder mean. That alone")
    print(f"  moves the annotator figure from {st.mean(per_seq):.2f} to "
          f"{st.mean(matched):.2f}.")

    out = Path(REPO / args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "max_len": MAX_LEN, "arms": table,
        "annotator": {"per_site": rows,
                      "per_sequence_all": st.mean(per_seq),
                      "per_sequence_matched": st.mean(matched),
                      "degenerate_sites": deg,
                      "per_onset": st.mean(per_onset),
                      "n_onsets": len(per_onset),
                      "coverage": st.mean(r["coverage"] for r in rows.values())},
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
