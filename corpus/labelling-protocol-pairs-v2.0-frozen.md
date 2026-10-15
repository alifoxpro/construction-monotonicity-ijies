# Pairwise Reversal Protocol, v2.0 — FROZEN

Status: **frozen 2026-10-14.** This document does not change once annotation
begins. If a case arises that it does not cover, the annotator answers
`cannot tell` and the item goes to adjudication; the protocol is not amended
mid-study. A change to any rule below starts a new version and a new study.

This protocol exists because of a defect in the previous one. The second
annotation pass over the main corpus (Section 5.3 of the paper) used protocol
v1.1 for four sequences and v1.2 for nine, and the annotator had read the
manuscript and so knew the corpus violation rate was reported as 0.000. Two
readings of a written rule by a party who knows the expected answer cannot
establish that reversals are absent. This protocol is written to remove both
defects: one frozen version for every item, and a task whose expected answer
cannot be inferred from the manuscript.

## 1. The task

You are shown two frames, LEFT and RIGHT, from the same construction site. You
answer one question:

> Which frame shows the building at a **later** point in construction?

Four answers are allowed, and all four are legitimate outcomes:

| Answer | Use it when |
|---|---|
| `LEFT is later` | the left frame shows work the right frame has not reached |
| `RIGHT is later` | the right frame shows work the left frame has not reached |
| `Same stage` | both frames show the same phase of work; no ordering is visible |
| `Cannot tell` | the view, framing, lighting or occlusion prevents a judgement |

**The pairs are not all in true order.** Half are deliberately shown swapped. An
answer that implies the building went backwards is therefore sometimes exactly
correct, and you should give it whenever that is what you see. There is no
answer distribution you are expected to produce.

## 2. What you must not do

* Do not open `key.json`, `task.json`, `adjudication.json`, any `labels.csv`,
  any `labels.B.csv`, or the manuscript, until you have finished and saved.
* Do not try to work out which sequence a pair came from, or to put pairs in
  order with each other. Each item is independent.
* Do not revise an earlier answer after forming a theory about the study. Going
  back to correct a misclick is fine; going back to make answers consistent with
  a pattern you have noticed is not.

## 3. How to judge "later"

Judge by the **physical state of the works**, not by image quality, weather,
season, time of day, or how much of the building is visible.

Ordered by construction sequence, later means any of:

1. ground prepared → footings or foundation present
2. foundation → load-bearing frame, walls or roof structure going up
3. frame → building being closed: sheathing, windows, doors, roof covering,
   cladding
4. closed → interior and final works, external finishes, site cleared

If the two frames differ **within** one of those phases — more courses of block,
more cladding fixed, scaffolding struck — that is still a later frame, and you
should say so. This is the main thing the previous protocol could not record,
and it is the main thing this study is for.

### Reversal

Answer that a frame is later even when doing so means the building went
backwards: work removed, an element demolished, a wall opened up again,
scaffolding re-erected over finished work. Do not reinterpret a reversal as a
different viewpoint of the same state unless you actually believe that is what
it is.

### Not a reversal

* A closer or wider shot of the same work.
* A different elevation of the same building at the same phase.
* Temporary covering, sheeting or weather protection over finished work.
* Deliveries, plant or materials on site that are not yet installed.

## 4. When to answer `Cannot tell`

Use it freely. It is not a failure, and an item answered this way is adjudicated
rather than discarded. Use it when:

* the works are not visible (interior close-up, sky, signage, a title card);
* the two frames show different parts of the site with no shared reference;
* you can see a difference but cannot tell which direction it runs.

Do not use it merely because the difference is small. If you can tell, say so.

## 5. Adjudication

Three classes of item are re-read by a third party under this same frozen
protocol:

1. items answered `Cannot tell`;
2. items where the blind answer contradicts the original annotator's labels;
3. **items where the planted direction was not detected** — a pair built from a
   forward stage boundary and shown swapped, which the annotator did not call
   backwards.

Class 3 is the one that decides whether the study is interpretable at all. If
planted reversals are not detected, the instrument lacks the sensitivity to find
real ones, and a null result carries no information. That check is reported
before any finding.

The adjudicator sees this protocol and the image pair. The adjudicator does not
see the blind annotator's answer, the original labels, or which class the item
came from. The adjudicator's answer replaces the blind answer for that item, and
the replacement is recorded, not silently applied.

## 6. What this study cannot establish

A reversal that the video editor cut from the footage is in no frame, so no
annotator can find it. This protocol bounds the reversals that are **visible in
the retained frames**; it says nothing about work that happened between two
retained frames. On the one sequence where elapsed time is observable, up to
17.4 days of site time separates consecutive frames. That limit is permanent and
is reported alongside any result from this study, not designed around.

## 7. Running it

```
python src/data_collection/blind_pairs.py prepare --n 200
python src/data_collection/blind_pairs.py score --answers blind_study/answers.csv
```

`prepare` writes `blind_study/task.html`, a single local page with no network
access. Open it, answer, and click **Save answers.csv**. Progress is kept in the
browser, so the task can be done over several sittings. The sealed key is
written at the same time and must not be opened until scoring.
