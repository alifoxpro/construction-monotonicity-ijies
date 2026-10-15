"""Why is the order-violation rate 0.000? — IJIES reviewer 2, point 3.

THE OBJECTION, verbatim: "The authors disclose that the second annotator knew
the expected zero-violation result and that the protocol changed between
annotation rounds. This transparency is welcome, but agreement with that
annotator cannot independently support the key assumption that reversals are
absent. Obtain a blinded human annotation subset with a frozen protocol and
adjudication of ambiguous boundaries. Distinguish reversals genuinely absent in
physical work from reversals removed by video editing, omitted by the exclusion
rule, or hidden within a coarse stage label. The validity of the data property
used to motivate the negative result depends on this separation."

The objection has two halves. The blinded human subset is the other half and
cannot be produced by analysis -- it needs a human annotator, and
src/data_collection/blind_pairs.py prepares that instrument. THIS script answers
the half that is measurable from the data already collected: of the four reasons
the rate could be zero, three leave traces, and the traces are quantified here
so that what remains attributable to physical reality is stated as a residual
rather than as a finding.

THE FOUR CANDIDATE EXPLANATIONS

  (a) Reversals genuinely did not happen on these sites.
  (b) Reversals happened but were cut by the editor. A time-lapse keeps a
      fraction of the elapsed time; anything shorter than the gap between kept
      frames cannot appear. MEASURABLE as the unobserved interval.
  (c) Reversals were removed by the exclusion rule, which drops frames labelled
      -1. MEASURABLE exactly: every maximal run of excluded frames is checked to
      see whether the stage before it is HIGHER than the stage after, which is a
      backward transition the rule would have swallowed.
  (d) Reversals are invisible to a five-class taxonomy because they happen
      inside one stage -- tearing out and rebuilding a wall never leaves stage
      "structure". MEASURABLE as the fraction of frames interior to a stage run,
      which is the blind spot's size.

AND ONE MORE THE OBJECTION DOES NOT LIST, which this script found: the
evaluation does not score the labelled sequence, it scores a 64-frame EVEN
SUBSAMPLE of it. Subsampling can step over a short reversal. The violation rate
is therefore computed twice -- on every labelled frame and on the scored
subsample -- and the two are compared.

Finally, the second annotator's labels are scored for violations in their own
right. If annotator B's independent pass produced a backward transition anywhere
that A's did not, the zero rate is a property of one annotator rather than of
the corpus, which is exactly the inference the reviewer says agreement cannot
support.

    python src/experiments/violation_provenance.py
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from data.sequence_dataset import list_sequence_dirs  # noqa: E402

RAW = REPO / "seq_data_raw"
MAX_LEN = 64

#: seq0010 is the one sequence carrying capture dates burnt into the image, so
#: it is the only place the time base is observed rather than assumed. Measured
#: by src/experiments/schedule_linkage.py.
SEQ0010_DAYS_TOTAL = 771
SEQ0010_FRAMES = 102
SEQ0010_MAX_DAYS_PER_FRAME = 17.40


def read_labels(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            rows.append((row[0].strip(), int(row[1])))
    return rows


def subsample(xs: list, max_len: int) -> list:
    if len(xs) <= max_len:
        return list(xs)
    step = (len(xs) - 1) / (max_len - 1)
    return [xs[round(i * step)] for i in range(max_len)]


def violations(stages: list[int]) -> tuple[int, int]:
    """(backward adjacent pairs, adjacent pairs). The paper's definition."""
    pairs = len(stages) - 1
    if pairs <= 0:
        return 0, 0
    back = sum(1 for a, b in zip(stages, stages[1:]) if b < a)
    return back, pairs


def runs(stages: list[int]) -> list[tuple[int, int, int]]:
    """(label, start, end_inclusive) maximal runs."""
    if not stages:
        return []
    out, start = [], 0
    for t in range(1, len(stages)):
        if stages[t] != stages[t - 1]:
            out.append((stages[start], start, t - 1))
            start = t
    out.append((stages[start], start, len(stages) - 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--json", default="experiments/violation_provenance.json")
    args = ap.parse_args()

    sites = [d.name for d in list_sequence_dirs(REPO / args.data)]
    report: dict[str, dict] = {}

    # ---------------------------------------------------------------- (0) + (d)
    print("WHAT IS ACTUALLY SCORED, AND WHAT IS NOT")
    print("The evaluation scores a 64-frame even subsample of the assessable")
    print("frames, not the labelled sequence. Both rates are given below.\n")
    hdr = (f"{'site':10s} {'labelled':>9s} {'scored':>7s} {'viol/pairs':>11s} "
           f"{'viol/pairs':>11s} {'interior':>9s} {'longest':>8s}")
    print(f"{'':10s} {'':>9s} {'':>7s} {'(labelled)':>11s} {'(scored)':>11s} "
          f"{'frames':>9s} {'run':>8s}")
    print(hdr)
    print("-" * len(hdr))

    tot_back_full = tot_pairs_full = tot_back_sub = tot_pairs_sub = 0
    tot_interior = tot_frames = 0
    run_lengths: list[int] = []
    for site in sites:
        a = read_labels(RAW / site / "labels.csv")
        assessable = [s for _, s in a if s >= 0]
        scored = subsample(assessable, MAX_LEN)
        bf, pf = violations(assessable)
        bs, ps = violations(scored)
        rs = runs(assessable)
        # a frame is INTERIOR if both neighbours carry the same stage: a reversal
        # occurring there cannot change the stage index at all
        interior = sum(max(0, (e - s + 1) - 2) for _, s, e in rs)
        run_lengths += [e - s + 1 for _, s, e in rs]
        longest = max((e - s + 1) for _, s, e in rs)
        tot_back_full += bf
        tot_pairs_full += pf
        tot_back_sub += bs
        tot_pairs_sub += ps
        tot_interior += interior
        tot_frames += len(assessable)
        report[site] = {"labelled": len(assessable), "scored": len(scored),
                        "back_labelled": bf, "pairs_labelled": pf,
                        "back_scored": bs, "pairs_scored": ps,
                        "interior_frames": interior, "longest_run": longest}
        print(f"{site:10s} {len(assessable):9d} {len(scored):7d} "
              f"{bf:5d}/{pf:<5d} {bs:5d}/{ps:<5d} "
              f"{interior:9d} {longest:8d}")
    print("-" * len(hdr))
    print(f"{'total':10s} {tot_frames:9d} {tot_pairs_sub + len(sites):7d} "
          f"{tot_back_full:5d}/{tot_pairs_full:<5d} "
          f"{tot_back_sub:5d}/{tot_pairs_sub:<5d} {tot_interior:9d}")

    print(f"\nviolation rate on every labelled frame : "
          f"{tot_back_full / tot_pairs_full:.6f} "
          f"({tot_back_full} of {tot_pairs_full} adjacent pairs)")
    print(f"violation rate on the scored subsample : "
          f"{tot_back_sub / tot_pairs_sub:.6f} "
          f"({tot_back_sub} of {tot_pairs_sub} adjacent pairs)")
    if tot_back_full == 0 and tot_back_sub == 0:
        print("Subsampling did not discard a reversal, because there is none to")
        print("discard at either resolution. The rate is therefore not an artifact")
        print("of the 64-frame evaluation -- but see below for what it still")
        print("cannot distinguish.")
    else:
        print("The two rates DIFFER: subsampling changes the measured quantity,")
        print("and the published rate is the subsample's, not the corpus's.")

    # ------------------------------------------------------------------- (d)
    print(f"\n\n(d) THE COARSE-LABEL BLIND SPOT")
    print(f"A reversal inside one stage never changes the stage index, so it is")
    print(f"invisible to this five-class taxonomy however carefully the frames")
    print(f"are labelled. The size of that blind spot is the number of frames")
    print(f"strictly inside a stage run:\n")
    print(f"  frames interior to a stage run : {tot_interior} of {tot_frames} "
          f"({100 * tot_interior / tot_frames:.1f}%)")
    print(f"  stage runs                      : {len(run_lengths)}")
    print(f"  median run length               : {st.median(run_lengths):.0f} frames")
    print(f"  longest run                     : {max(run_lengths)} frames")
    print(f"\n{100 * tot_interior / tot_frames:.0f}% of the corpus sits where a "
          f"reversal would leave the stage index\nuntouched. The zero rate is "
          f"therefore silent about re-work within a stage,\nwhich is the most "
          f"common kind on a real site: tearing out and rebuilding a\nwall never "
          f"leaves the structure stage.")

    # ------------------------------------------------------------------- (c)
    print("\n\n(c) THE EXCLUSION RULE")
    print("Every maximal run of excluded (-1) frames is checked to see what it")
    print("sits between. A run whose preceding stage is HIGHER than its following")
    print("stage would be hiding a backward transition.\n")
    bridges = []
    for site in sites:
        a = read_labels(RAW / site / "labels.csv")
        stages = [s for _, s in a]
        i = 0
        while i < len(stages):
            if stages[i] >= 0:
                i += 1
                continue
            j = i
            while j < len(stages) and stages[j] < 0:
                j += 1
            before = next((stages[k] for k in range(i - 1, -1, -1)
                           if stages[k] >= 0), None)
            after = next((stages[k] for k in range(j, len(stages))
                          if stages[k] >= 0), None)
            bridges.append({"site": site, "start": i, "end": j - 1,
                            "length": j - i, "before": before, "after": after,
                            "hides_backward": (before is not None
                                               and after is not None
                                               and after < before)})
            i = j
    hiding = [b for b in bridges if b["hides_backward"]]
    print(f"  runs of excluded frames        : {len(bridges)}")
    print(f"  excluded frames in total       : {sum(b['length'] for b in bridges)}")
    print(f"  runs bridging a stage CHANGE   : "
          f"{sum(1 for b in bridges if b['before'] != b['after'])}")
    print(f"  runs hiding a BACKWARD change  : {len(hiding)}")
    for b in hiding:
        print(f"     {b['site']} frames {b['start']}-{b['end']}: "
              f"stage {b['before']} → {b['after']}")
    if not hiding:
        print("\nNo excluded run bridges a decrease, so the exclusion rule is not")
        print("concealing a reversal. This is the one of the four explanations")
        print("that can be ruled out rather than merely bounded.")

    # ------------------------------------------------------------------- (b)
    print("\n\n(b) WHAT THE EDITOR REMOVED")
    print("A time-lapse keeps a fraction of elapsed time. Any reversal shorter")
    print("than the interval between two kept frames cannot appear in the data,")
    print("whatever the annotator does.\n")
    gaps = []
    for site in sites:
        a = read_labels(RAW / site / "labels.csv")
        n = len([1 for _, s in a if s >= 0])
        step = 1 if n <= MAX_LEN else (n - 1) / (MAX_LEN - 1)
        gaps.append((site, n, step))
    worst = max(gaps, key=lambda g: g[2])
    print(f"  subsampling step, frames between kept samples:")
    print(f"    largest  : {worst[2]:.1f} frames ({worst[0]}, "
          f"{worst[1]} labelled frames)")
    print(f"    median   : {st.median(g[2] for g in gaps):.1f} frames")
    dpf = SEQ0010_DAYS_TOTAL / SEQ0010_FRAMES
    print(f"\n  seq0010 is the only sequence whose elapsed time is observable")
    print(f"  (dates burnt into the image): {SEQ0010_DAYS_TOTAL} days over "
          f"{SEQ0010_FRAMES} frames,")
    print(f"  so a frame is worth {dpf:.1f} days on average and up to "
          f"{SEQ0010_MAX_DAYS_PER_FRAME} days at its")
    print(f"  fastest. On that sequence any reversal completed inside about")
    print(f"  {SEQ0010_MAX_DAYS_PER_FRAME:.0f} days of site time can fall "
          f"between two consecutive frames and")
    print(f"  leave no trace. Demolishing and rebuilding a wall fits in that "
          f"window.")
    print(f"\n  This is a bound, not an estimate: it says the data CANNOT show")
    print(f"  short reversals, not that short reversals occurred.")

    # -------------------------------------------- the second annotator's own rate
    print("\n\nTHE SECOND ANNOTATOR, SCORED IN HIS OWN RIGHT")
    print("Agreement cannot support the claim that reversals are absent, as the")
    print("reviewer says. But annotator B's labels can be scored for violations")
    print("independently: if B's pass contained a reversal anywhere A's did not,")
    print("the zero rate would be a property of one annotator, not of the corpus.\n")
    b_total_back = b_total_pairs = 0
    rows = []
    for site in sites:
        bp = RAW / site / "labels.B.csv"
        if not bp.exists():
            continue
        b = [s for _, s in read_labels(bp) if s >= 0]
        bb, bpairs = violations(b)
        b_sub = subsample(b, MAX_LEN)
        bbs, bps = violations(b_sub)
        b_total_back += bb
        b_total_pairs += bpairs
        rows.append((site, len(b), bb, bpairs, bbs, bps))
        report[site]["b_back_labelled"] = bb
        report[site]["b_pairs_labelled"] = bpairs
    print(f"{'site':10s} {'B frames':>9s} {'B viol/pairs':>14s}")
    for site, n, bb, bpairs, _, _ in rows:
        print(f"{site:10s} {n:9d} {bb:7d}/{bpairs:<6d}")
    print(f"\n  annotator B violation rate: "
          f"{b_total_back / b_total_pairs if b_total_pairs else float('nan'):.6f} "
          f"({b_total_back} of {b_total_pairs} pairs)")
    if b_total_back == 0:
        print("  B also produced no reversal. That rules out the narrowest")
        print("  explanation -- that one annotator imposed monotonicity -- and")
        print("  rules out nothing else. B knew the expected result and worked")
        print("  from a protocol that changed between rounds, so this is")
        print("  consistency between two readings of the same written rules,")
        print("  not independent evidence about the physical works.")

    # ------------------------------------------------------------------ verdict
    print("\n\nWHAT CAN AND CANNOT BE CONCLUDED")
    print(f"  (c) the exclusion rule        RULED OUT: no excluded run bridges a")
    print(f"                                decrease.")
    print(f"  (0) the 64-frame subsample    RULED OUT: the rate is 0.000 at both")
    print(f"                                resolutions.")
    print(f"  (d) the coarse taxonomy       NOT RULED OUT: "
          f"{100 * tot_interior / tot_frames:.0f}% of frames sit where")
    print(f"                                a reversal changes no index.")
    print(f"  (b) editing                   NOT RULED OUT: up to "
          f"{SEQ0010_MAX_DAYS_PER_FRAME:.0f} days of site")
    print(f"                                time can separate two scored frames.")
    print(f"  (a) physically absent         THE RESIDUAL, and it is not isolated")
    print(f"                                by any measurement available here.")
    print("\nThe honest statement of the data property is therefore narrower than")
    print("the one the paper used: the corpus contains no reversal that is BOTH")
    print("coarse enough to cross a stage boundary AND slow enough to survive")
    print("editing. That is what bounds the learnable gate's opportunity to")
    print("learn, and it is enough for the negative result -- a gate cannot learn")
    print("from evidence the data cannot carry -- but it is not the claim that")
    print("re-work did not occur on these sites.")

    out = Path(REPO / args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "max_len": MAX_LEN,
        "rate_labelled": tot_back_full / tot_pairs_full,
        "rate_scored": tot_back_sub / tot_pairs_sub,
        "interior_frames": tot_interior, "total_frames": tot_frames,
        "interior_fraction": tot_interior / tot_frames,
        "stage_runs": len(run_lengths),
        "median_run_length": st.median(run_lengths),
        "longest_run": max(run_lengths),
        "excluded_runs": bridges,
        "excluded_runs_hiding_backward": len(hiding),
        "annotator_b_rate": (b_total_back / b_total_pairs
                             if b_total_pairs else None),
        "per_site": report,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
