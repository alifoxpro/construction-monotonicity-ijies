"""What the blinded pairwise pass does and does not establish.

`blind_pairs.py score` reports the headline counts. This script asks the three
questions that decide whether those counts mean anything, because the headline
"10 of 20 planted reversals detected" is ambiguous on its own: 50% could mean
the annotators are blind to construction order, or that they can read it
perfectly but answer with a positional bias.

  1. CAN THEY READ CONSTRUCTION ORDER AT ALL? Measured on the boundary pairs
     shown in their true order, where the correct answer is "right".
  2. CAN THEY DETECT A REVERSAL? Measured on the planted half, where the frames
     were swapped and the correct answer is "left".
  3. DO THE APPARENT REVERSALS TRACK THE BUILDING OR THE SCREEN? A reversal
     call on an interior pair is only evidence if its rate does not depend on
     which way the pair happened to be displayed.

Question 3 is the one that matters. Items were assigned to swapped and
unswapped halves at random, so a real effect must appear at the same rate in
both; a rate that differs between them is a property of the display, not of the
site.

    python src/data_collection/blind_pairs_analysis.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "blind_study"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)


def directional(items, ans, swapped: bool):
    """(reversal calls, forward calls) among items where a direction was given."""
    rev = fwd = 0
    for k in items:
        a = ans.get(k["id"])
        if a in (None, "same", "cannot"):
            continue
        backwards = (a == "right") if swapped else (a == "left")
        rev += backwards
        fwd += not backwards
    return rev, fwd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", default=str(STUDY / "answers.csv"))
    ap.add_argument("--json", default=str(STUDY / "analysis.json"))
    args = ap.parse_args()

    key = json.loads((STUDY / "key.json").read_text(encoding="utf-8"))["items"]
    ans = {r["item_id"]: r["answer"]
           for r in csv.DictReader(open(args.answers, encoding="utf-8"))}
    out: dict = {"n_items": len(key), "n_answered": len(ans)}
    print(f"{len(ans)} of {len(key)} items answered\n")

    # ---- 1 and 2: the boundary stratum ------------------------------------
    print("1 & 2. BOUNDARY PAIRS — annotator A recorded a forward stage change\n")
    for swapped, label, truth in (
            (False, "shown in true order", "right"),
            (True, "shown swapped (planted reversal)", "left")):
        sel = [k for k in key if k["stratum"] == "boundary"
               and k["swapped"] == swapped]
        c = collections.Counter(ans.get(k["id"]) for k in sel)
        n_dir = c["left"] + c["right"]
        correct = c[truth]
        p = binomtest(correct, n_dir, 0.5).pvalue if n_dir else float("nan")
        out["true_order" if not swapped else "planted"] = {
            "n": len(sel), "answers": dict(c), "directional": n_dir,
            "correct": correct, "p": float(p)}
        print(f"  {label}  (correct = {truth})")
        print(f"    n = {len(sel)}, answers {dict(c)}")
        print(f"    correct on directional calls: {correct}/{n_dir}"
              f"{f' ({100*correct/n_dir:.0f}%), p = {p:.3f}' if n_dir else ''}")
        print()

    # ---- response bias -----------------------------------------------------
    allc = collections.Counter(ans.values())
    n_dir = allc["left"] + allc["right"]
    p_bias = binomtest(allc["right"], n_dir, 0.5).pvalue
    out["response_bias"] = {"right": allc["right"], "left": allc["left"],
                            "p": float(p_bias)}
    print("RESPONSE BIAS — presentation order was randomised, so a 'right'")
    print("preference across all 200 items is a property of the annotator\n")
    print(f"  right {allc['right']}, left {allc['left']} of {n_dir} "
          f"directional calls, p = {p_bias:.4f}\n")

    # ---- 3: do interior reversal calls track the display? ------------------
    print("3. INTERIOR PAIRS — both frames in one stage, so annotator A's")
    print("   taxonomy could not have recorded a reversal between them\n")
    sw = [k for k in key if k["stratum"] == "interior" and k["swapped"]]
    no = [k for k in key if k["stratum"] == "interior" and not k["swapped"]]
    rs, fs = directional(sw, ans, True)
    rn, fn = directional(no, ans, False)
    _, p_fisher = fisher_exact([[rs, fs], [rn, fn]])
    out["interior"] = {"swapped": {"reversal": rs, "forward": fs},
                       "not_swapped": {"reversal": rn, "forward": fn},
                       "fisher_p": float(p_fisher)}
    print(f"   shown in true order : {rn} reversal of {rn+fn} directional "
          f"({100*rn/(rn+fn):.0f}%)")
    print(f"   shown swapped       : {rs} reversal of {rs+fs} directional "
          f"({100*rs/(rs+fs):.0f}%)")
    print(f"   Fisher exact on the difference: p = {p_fisher:.4f}\n")

    # ---- verdict -----------------------------------------------------------
    # bool(), not the bare comparison: SciPy returns numpy scalars, and a
    # numpy.bool_ cannot be serialised to JSON.
    reads_order = bool(out["true_order"]["p"] < 0.05)
    detects = bool(out["planted"]["p"] < 0.05)
    display_dependent = bool(p_fisher < 0.10)
    print("VERDICT\n")
    print(f"  reads construction order when shown the true order : "
          f"{'YES' if reads_order else 'NO'} "
          f"({out['true_order']['correct']}/{out['true_order']['directional']}, "
          f"p = {out['true_order']['p']:.3f})")
    print(f"  detects a planted reversal significantly            : "
          f"{'YES' if detects else 'NO'} "
          f"({out['planted']['correct']}/{out['planted']['directional']}, "
          f"p = {out['planted']['p']:.3f})")
    print(f"  apparent interior reversals depend on display order : "
          f"{'YES' if display_dependent else 'NO'} (p = {p_fisher:.3f})")
    print()
    if reads_order and not detects:
        print("  The instrument is not blind to construction order -- it reads")
        print("  it almost perfectly when the true order is shown. What it does")
        print("  NOT do is detect a reversal at a rate distinguishable from")
        print("  chance, and the corpus offers only 40 boundary pairs, so the")
        print("  planted control is 20 items and cannot carry that test.")
    if display_dependent:
        print()
        print("  The interior 'reversals' are therefore not usable as evidence:")
        print("  their rate depends on which way the pair was displayed, which")
        print("  a real reversal in the building cannot do.")
    out["verdict"] = {"reads_order": reads_order, "detects_planted": detects,
                      "display_dependent": display_dependent}
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n",
                               encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
