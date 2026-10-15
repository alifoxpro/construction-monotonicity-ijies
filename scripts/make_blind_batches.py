"""Split the blinded pairwise task into briefs for independent annotators.

Each brief carries the task rules and the image paths for its own items, and
NOTHING ELSE. That restriction is the whole point of the exercise: the second
annotation pass reported in the paper was invalidated because the annotator had
read the manuscript and knew the corpus violation rate was reported as 0.000, so
a replacement pass that leaks the same fact is worth nothing.

WHAT IS DELIBERATELY WITHHELD FROM THE BRIEF

  * the expected answer, the corpus violation rate, and the existence of any
    expectation at all;
  * the frozen protocol document itself, whose own preamble states the 0.000
    figure in its second paragraph while explaining why the previous pass was
    defective -- so the rules are reproduced here instead of referenced;
  * which items are planted reversals, which stratum an item came from, and
    which site a pair belongs to;
  * the original annotator's labels.

The item identifiers are the hashes from the sealed key, so an annotator cannot
sort items back into sequence order, and the briefs are written in the shuffled
order the sampler produced.

    python src/data_collection/make_blind_batches.py --batches 10
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "blind_study"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

RULES = """\
## The task

You will be shown pairs of photographs. Both frames in a pair come from the same
building site, photographed at two different moments. For each pair, answer ONE
question:

> Which frame shows the building at a LATER point in its construction?

Four answers are allowed, and all four are legitimate:

| Answer | Use it when |
|---|---|
| `left` | the LEFT frame shows work the right frame has not reached |
| `right` | the RIGHT frame shows work the left frame has not reached |
| `same` | both frames show the same phase of work; no ordering is visible |
| `cannot` | the view, framing, lighting or occlusion prevents a judgement |

## How to judge "later"

Judge by the PHYSICAL STATE OF THE WORKS, not by image quality, weather,
season, time of day, or how much of the building is visible.

Ordered by construction sequence, later means any of:

1. ground prepared -> footings or foundation present
2. foundation -> load-bearing frame, walls or roof structure going up
3. frame -> building being closed: sheathing, weather barrier, windows, doors,
   roof covering, cladding
4. closed -> interior and final works, external finishes, site cleared

If the two frames differ WITHIN one of those phases -- more courses of block,
more cladding fixed, scaffolding struck -- that is still a later frame, and you
should say so.

### Work can also go backwards

Answer that a frame is later even when that means the building went BACKWARDS:
work removed, an element demolished, a wall opened up again, scaffolding
re-erected over finished work. Do not reinterpret such a case as a different
viewpoint of the same state unless you actually believe that is what it is.

### These are NOT changes in state

* A closer or wider shot of the same work.
* A different elevation of the same building at the same phase.
* Temporary covering, sheeting or weather protection over finished work.
* Deliveries, plant or materials on site that are not yet installed.

## When to answer `cannot`

Use it freely. It is not a failure. Use it when the works are not visible
(interior close-up, sky, signage, a title card), when the two frames show
different parts of the site with no shared reference, or when you can see a
difference but cannot tell which direction it runs.

Do not use it merely because the difference is small. If you can tell, say so.

## Rules

* Judge each pair ON ITS OWN. The pairs are unrelated to each other and are in
  no particular order.
* Use ONLY the two images. Do not open, read or search for any other file in
  this repository, and do not look for labels, annotations, documentation or
  source code.
* There is no expected distribution of answers. Answer what you see.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--out", default="blind_study/batches")
    args = ap.parse_args()

    items = json.loads((STUDY / "task.json").read_text(encoding="utf-8"))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("batch_*.md"):
        old.unlink()

    size = -(-len(items) // args.batches)        # ceiling division
    written = 0
    for b in range(args.batches):
        chunk = items[b * size:(b + 1) * size]
        if not chunk:
            continue
        lines = [f"# Pairwise judgement task — batch {b + 1}", "", RULES, "",
                 "## Your items", "",
                 f"There are {len(chunk)} pairs below. For each one, open the "
                 f"two images and decide.", ""]
        for it in chunk:
            # Anonymised copies, not the originals: the real paths carry the
            # site name and the frame number, which would hand the annotator
            # the ordering outright (see anonymise_blind_frames.py).
            left = (STUDY / "frames" / f"{it['id']}_L.jpg").resolve()
            right = (STUDY / "frames" / f"{it['id']}_R.jpg").resolve()
            lines += [f"### {it['id']}", "",
                      f"- LEFT : `{left}`",
                      f"- RIGHT: `{right}`", ""]
        lines += ["## How to report", "",
                  "Return ONLY a CSV block, one line per item, in this exact "
                  "form, with no commentary:", "", "```", "item_id,answer"]
        lines += [f"{it['id']},<left|right|same|cannot>" for it in chunk[:2]]
        lines += ["...", "```", ""]
        path = out / f"batch_{b + 1:02d}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        written += len(chunk)
        print(f"  {path.name}: {len(chunk)} items")

    print(f"\n{written} items across {args.batches} briefs in {out}")
    print("Each brief contains the task rules and its own image paths only:")
    print("no expected answer, no protocol preamble, no strata, no site names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
