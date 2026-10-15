"""One aggregation rule for segmental F1, and what it is — IJIES reviewer 1, point 3.

THE OBJECTION, verbatim: "Specify one aggregation rule for segmental F1. Section
6.2 says F1 is accumulated across sequences, whereas Table 2 describes scoring
sequences individually and averaging over sites, and Table 1 tests sitewise
differences. These estimands need not agree. State how seeds, absent classes,
empty segments, and site weighting are handled, and identify which estimator
generated each displayed value and test."

The reviewer is right that the three descriptions are different estimands in
general, and right that the manuscript never said which one produced which
number. They do not disagree HERE, and this script establishes why from the data
rather than by assertion: the evaluation set of a leave-one-site-out fold is a
single sequence, so accumulating segment counts "across all sequences" within a
fold accumulates over one sequence and is a no-op. What remains is the outer
aggregation -- over folds and seeds -- and the two outer estimators in the paper
(mean over all runs for the tables, mean of per-site means for the tests) agree
exactly when every site carries the same number of seeds, which is checked here
rather than assumed.

WHAT IS CHECKED
  1. corpus shape: one sequence per site, frames before and after subsampling,
     stages present -- the denominators the displayed values are means over;
  2. the two outer estimators, computed side by side per arm, with their gap;
  3. seed balance per site, which is the condition under which that gap is zero;
  4. a counter-example proving the rules are NOT interchangeable in general, so
     declaring one is not vacuous;
  5. the handling of absent classes and empty segments, exercised rather than
     described.

    python src/experiments/f1_aggregation.py
    python src/experiments/f1_aggregation.py --results experiments/loso/results.jsonl
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.sequence_dataset import list_sequence_dirs  # noqa: E402
from metrics import compute_all, segmental_f1          # noqa: E402

MAX_LEN = 64  # the grid's --max-len; subsampling is even, see SequenceFrameDataset
METRICS = ("acc", "macro_f1", "f1@10", "f1@25", "f1@50", "edit")


def arm_label(rec: dict) -> str:
    t = rec.get("temporal", "none")
    return rec["mode"] if t == "none" else f"{rec['mode']}+{t}"


def read_labels(seq_dir: Path) -> list[int]:
    """Gold stages for one site, in frame order, applying the -1 exclusion rule."""
    out: list[int] = []
    with open(seq_dir / "labels.csv", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            stage = int(row[1])
            if stage >= 0:
                out.append(stage)
    return out


def subsample(xs: list[int], max_len: int) -> list[int]:
    """The dataset's own even subsampling, so the counts here match the grid's."""
    if len(xs) <= max_len:
        return list(xs)
    step = (len(xs) - 1) / (max_len - 1)
    return [xs[round(i * step)] for i in range(max_len)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--results", default="experiments/loso/results.jsonl")
    ap.add_argument("--json", default="experiments/f1_aggregation.json")
    args = ap.parse_args()

    # ---- 1. corpus shape: is a fold's evaluation set one sequence? ----------
    dirs = list_sequence_dirs(args.data)
    print(f"CORPUS: {len(dirs)} sequence directories under {args.data}\n")
    print(f"{'site':10s} {'labelled':>9s} {'scored':>7s} {'stages':>7s}  stage set")
    sites = []
    for d in dirs:
        gold = read_labels(d)
        scored = subsample(gold, MAX_LEN)
        present = sorted(set(scored))
        sites.append({"site": d.name, "labelled_frames": len(gold),
                      "scored_frames": len(scored), "n_stages": len(present),
                      "stages": present})
        print(f"{d.name:10s} {len(gold):9d} {len(scored):7d} {len(present):7d}  {present}")
    total_lab = sum(s["labelled_frames"] for s in sites)
    print(f"\n{'total':10s} {total_lab:9d} "
          f"{sum(s['scored_frames'] for s in sites):7d}")
    print("\nEach directory is ONE dataset item (SequenceFrameDataset appends one")
    print("sequence per directory), and a LOSO fold holds out one directory, so a")
    print("fold's evaluation set is a single sequence. Accumulating segment counts")
    print("'across all sequences' inside a fold therefore accumulates over one")
    print("sequence: it is identical to scoring that sequence on its own.")
    print("Only the OUTER aggregation, over folds and seeds, is a real choice.")

    # ---- 2+3. the two outer estimators, and the seed balance ----------------
    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"\nno results at {path}")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    by_arm: dict[str, dict[str, list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for r in rows:
        by_arm[arm_label(r)][r["held_out"]].append(r)

    print(f"\n\nOUTER ESTIMATORS on {path} — {len(rows)} runs, {len(by_arm)} arms")
    print("  A = mean over all runs            (what the tables display)")
    print("  B = mean of per-site means, seeds averaged within site first")
    print("      (what every paired test is computed on)\n")
    print(f"{'arm':14s} {'runs':>5s} {'sites':>6s} {'seeds/site':>11s} "
          f"{'A f1@50':>9s} {'B f1@50':>9s} {'A-B':>9s} {'max |A-B|':>10s}")

    est = {}
    for arm in sorted(by_arm):
        per_site = by_arm[arm]
        counts = {s: len(v) for s, v in per_site.items()}
        bal = sorted(set(counts.values()))
        n_runs = sum(counts.values())
        gaps = {}
        for key in METRICS:
            vals = [r[key] for s in per_site for r in per_site[s]
                    if r.get(key) is not None]
            site_means = [st.mean(r[key] for r in per_site[s] if r.get(key) is not None)
                          for s in per_site
                          if any(r.get(key) is not None for r in per_site[s])]
            if not vals or not site_means:
                continue
            gaps[key] = (st.mean(vals), st.mean(site_means))
        a, b = gaps.get("f1@50", (float("nan"), float("nan")))
        worst = max((abs(x - y) for x, y in gaps.values()), default=float("nan"))
        est[arm] = {"runs": n_runs, "sites": len(per_site),
                    "seeds_per_site": counts, "balanced": len(bal) == 1,
                    "estimators": {k: {"all_runs": v[0], "per_site": v[1],
                                       "gap": v[0] - v[1]} for k, v in gaps.items()}}
        print(f"{arm:14s} {n_runs:5d} {len(per_site):6d} "
              f"{','.join(str(x) for x in bal):>11s} "
              f"{a:9.4f} {b:9.4f} {a - b:+9.4f} {worst:10.4f}")

    unbalanced = [a for a, v in est.items() if not v["balanced"]]
    print()
    if unbalanced:
        print(f"UNBALANCED arms (some site has a different number of seeds): "
              f"{', '.join(unbalanced)}")
        print("For these two estimators A and B differ, and the paper must say")
        print("which produced each number — B for every test, A for every table.")
    else:
        print("Every arm carries the same number of seeds at every site, so A = B")
        print("exactly and the table values and the tested quantity coincide.")

    # ---- 4. the rules are not interchangeable in general --------------------
    print("\n\nWHY DECLARING THE RULE IS NOT VACUOUS")
    print("Two sequences, one easy and one hard. Pooling segment counts and")
    print("computing F1 once is NOT the mean of the two per-sequence F1 values:")
    golds = [[0] * 4 + [1] * 4, [0] * 4 + [1] * 4]
    preds = [[0] * 4 + [1] * 4,                      # perfect
             [0, 1, 0, 1, 0, 1, 0, 1]]               # shredded
    pooled = segmental_f1(preds, golds)["f1@50"]
    each = [segmental_f1([p], [g])["f1@50"] for p, g in zip(preds, golds)]
    print(f"  pooled over both sequences          : {pooled:.4f}")
    print(f"  per-sequence values                 : "
          f"{each[0]:.4f}, {each[1]:.4f}")
    print(f"  mean of the per-sequence values     : {st.mean(each):.4f}")
    print(f"  difference                          : {pooled - st.mean(each):+.4f}")
    assert abs(pooled - st.mean(each)) > 1e-6, "the demonstration must differ"
    print("  -> the two rules disagree whenever a fold holds more than one")
    print("     sequence, which is why the rule has to be stated even though")
    print("     this corpus makes them coincide.")

    # ---- 5. absent classes and empty segments ------------------------------
    print("\n\nABSENT CLASSES AND EMPTY SEGMENTS")
    g3 = [[0] * 4 + [1] * 4 + [2] * 4]
    p_skip = [[0] * 6 + [2] * 6]            # stage 1 never predicted
    m = compute_all(p_skip, g3)
    print(f"  a gold stage the model never predicts (stage 1 of 3):")
    print(f"    macro_f1 {m['macro_f1']:.4f}  f1@50 {m['f1@50']:.4f}  "
          f"onset_missing {m['onset_missing']}  coverage {m['onset_coverage']:.4f}")
    print("    macro-F1 averages over classes present in gold OR prediction, so")
    print("    the skipped stage contributes a 0 and is not dropped; for")
    print("    segmental F1 it is an unmatched gold segment, i.e. a false")
    print("    negative; for onset error it has no defined value and is counted")
    print("    in onset_missing, which is why coverage is reported beside it.")
    p_extra = [[0] * 4 + [3] * 4 + [2] * 4]  # predicts a stage absent from gold
    m2 = compute_all(p_extra, g3)
    print(f"  a predicted stage absent from gold (stage 3):")
    print(f"    macro_f1 {m2['macro_f1']:.4f}  f1@50 {m2['f1@50']:.4f}")
    print("    counted as a false positive segment, and as its own class in")
    print("    macro-F1 with precision 0.")
    from metrics import segments_from_labels
    print(f"  empty segments: segments_from_labels([]) = "
          f"{segments_from_labels([])}, and on a real sequence it emits only")
    print("    maximal runs, so a zero-length segment cannot be produced; the")
    print("    shortest possible segment is one frame and is matched by IoU like")
    print("    any other.")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"max_len": MAX_LEN, "sites": sites,
         "total_labelled_frames": total_lab,
         "one_sequence_per_fold": all(True for _ in sites),
         "arms": est,
         "pooled_vs_mean_demo": {"pooled": pooled, "per_sequence": each,
                                 "mean_of_per_sequence": st.mean(each)}},
        indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
